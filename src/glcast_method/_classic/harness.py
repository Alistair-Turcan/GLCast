"""Recovered numerical method implementation."""

import json



import os



from typing import Dict, Iterable, List, Optional, Sequence, Tuple



import numpy as np



import pandas as pd



try:
    import celer
except Exception:  # pragma: no cover
    celer = None



def build_design_for_state(target_wide, feature_wides, state, start_date, end_date,
                           feature_lags, ar_lags, unit):
    """One lagged design matrix for one state over [start_date, end_date]."""
    max_lag = max(max(feature_lags), max(ar_lags))
    start_ext = start_date - pd.Timedelta(max_lag, unit=unit)

    idx = target_wide.loc[start_ext:end_date].index
    if state not in target_wide.columns or len(idx) == 0:
        return None

    y_series = target_wide.loc[idx, state].copy()

    X_blocks: List[pd.Series] = []
    feature_names: List[str] = []
    group_names: List[str] = []

    for sig_name, wide in feature_wides.items():
        if state not in wide.columns:
            continue
        s = wide.loc[idx, state].copy()
        for lag in feature_lags:
            X_blocks.append(s.shift(lag))
            feature_names.append(f"{sig_name}_lag{lag}")
        group_names.append(sig_name)

    for lag in ar_lags:
        X_blocks.append(y_series.shift(lag))
        feature_names.append(f"AR_lag{lag}")
    group_names.append("AR")

    if len(X_blocks) == 0:
        return None

    X_df = pd.concat(X_blocks, axis=1)
    X_df.columns = feature_names

    aligned = pd.concat([y_series.rename("y"), X_df], axis=1)
    aligned = aligned.loc[start_date:end_date].dropna()
    if len(aligned) == 0:
        return None

    y = aligned["y"].to_numpy(dtype=float)
    X = aligned.drop(columns=["y"]).to_numpy(dtype=float)
    dates = pd.to_datetime(aligned.index)
    return X, y, dates, feature_names, group_names



def compute_walkforward_schedule(dates, train_size, test_size, retrain_every, unit="D",
                                 start_minus_one=True):
    """Rolling sequence of training end dates.

    start_minus_one=True (spec footing): first te = series_min + (train_size - 1),
    so the train window [te-(train_size-1), te] starts exactly at series_min. This
    reproduces the spec's stated origins (COVID Sundays 2020-09-27 etc. from a
    2020-04-01 series start). start_minus_one=False recovers the literal recovered
    code (series_min + train_size).
    """
    dates = pd.to_datetime(pd.Index(dates)).sort_values()
    off = int(train_size) - 1 if start_minus_one else int(train_size)
    start = dates.min() + pd.Timedelta(off, unit=unit)
    end = dates.max() - pd.Timedelta(test_size, unit=unit)
    if start >= end:
        return []
    start = dates[dates.get_indexer([start], method="nearest")[0]]
    train_ends: List[pd.Timestamp] = []
    cur = start
    while cur <= end:
        train_ends.append(cur)
        cur = cur + pd.Timedelta(retrain_every, unit=unit)
    return train_ends



def extend_indices(g_indices, group_size):
    g_indices = np.asarray(g_indices, dtype=int)
    return (g_indices[:, None] * group_size + np.arange(group_size)[None, :]).ravel().astype(int)



def shrink_indices(v_indices, group_size):
    v_indices = np.asarray(v_indices, dtype=int)
    return np.unique(v_indices // group_size).astype(int)



def standardize_xy(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=0)
    X_std[X_std == 0] = 1.0
    Xz = (X - X_mean) / X_std
    y_mean = y.mean()
    y_std = y.std(ddof=0)
    if y_std == 0:
        y_std = 1.0
    yz = (y - y_mean) / y_std
    return Xz, yz, X_mean, X_std, y_mean, y_std



def recover_coefficients(unit_coef, X_std, y_std):
    X_std = np.asarray(X_std, dtype=float)
    return (y_std / X_std) * np.asarray(unit_coef, dtype=float)



def recover_intercept(X, y, coef_original, X_mean=None, y_mean=None):
    if X_mean is None:
        X_mean = np.mean(X, axis=0)
    if y_mean is None:
        y_mean = np.mean(y)
    return float(y_mean - X_mean.dot(coef_original))



def SIS(X, y, nSel):
    X = np.asarray(X)
    y = np.asarray(y)
    y_std = y.std()
    if y_std == 0:
        return np.array([], dtype=int)
    corrs = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        xj = X[:, j]
        sx = xj.std()
        if sx == 0:
            corrs[j] = 0.0
        else:
            corrs[j] = abs(np.cov(xj, y, bias=True)[0, 1] / (sx * y_std))
    order = np.argsort(corrs)[::-1]
    return order[:nSel].astype(int)



def _group_soft_threshold(beta_group, lam):
    norm = np.linalg.norm(beta_group, 2)
    if norm == 0 or norm <= lam:
        return np.zeros_like(beta_group)
    return (1.0 - lam / norm) * beta_group



def _soft_threshold(x, lam):
    """Elementwise soft threshold (L1 prox)."""
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)



def _fit_grpLasso_fista(Xz, yz, alpha, group_size, weights=None, max_iter=3000, tol=1e-6,
                        sparse_l1=0.0):
    """Group Lasso via FISTA. If sparse_l1 > 0, adds an elementwise L1 prox after the
    group soft-threshold -> sparse group lasso (group-level + within-group sparsity).
    sparse_l1 is the L1 penalty weight (same standardization scale as alpha); it may be
    a scalar or a per-column vector of length p (e.g. lag-decay weights)."""
    Xz = np.asarray(Xz, dtype=float)
    yz = np.asarray(yz, dtype=float)
    n, p = Xz.shape
    n_groups = p // group_size
    if weights is None:
        weights_arr = np.ones(n_groups, dtype=float)
    else:
        weights_arr = np.asarray(weights, dtype=float)
        if weights_arr.size == 1:
            weights_arr = np.repeat(float(weights_arr), n_groups)
    sl1 = np.asarray(sparse_l1, dtype=float)
    has_l1 = bool(np.any(sl1 > 0.0))
    spectral = np.linalg.norm(Xz, ord=2)
    L = (spectral ** 2) / max(n, 1)
    L = max(L, 1e-8)
    step = 1.0 / L
    beta = np.zeros(p, dtype=float)
    z = beta.copy()
    t = 1.0
    for _ in range(max_iter):
        grad = Xz.T @ (Xz @ z - yz) / max(n, 1)
        beta_next = z - step * grad
        for g in range(n_groups):
            j0 = g * group_size
            j1 = j0 + group_size
            beta_next[j0:j1] = _group_soft_threshold(beta_next[j0:j1], step * alpha * weights_arr[g])
        if has_l1:
            beta_next = _soft_threshold(beta_next, step * sl1)
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = beta_next + ((t - 1.0) / t_next) * (beta_next - beta)
        if np.linalg.norm(beta_next - beta) <= tol * max(1.0, np.linalg.norm(beta)):
            beta = beta_next
            break
        beta = beta_next
        t = t_next
    return beta



def fit_grpLasso(Xz, yz, alpha, group_size, weights=None, sparse_l1=0.0):
    if bool(np.any(np.asarray(sparse_l1, dtype=float) > 0.0)):
        # celer has no sparse-group-lasso; use FISTA with both prox operators
        coef = _fit_grpLasso_fista(Xz, yz, alpha=alpha, group_size=group_size,
                                   weights=weights, sparse_l1=sparse_l1)
        return None, coef
    if celer is not None:
        model = celer.GroupLasso(groups=group_size, alpha=alpha, tol=1e-4,
                                 fit_intercept=False, weights=weights)
        model.fit(Xz, yz)
        return model, model.coef_.copy()
    coef = _fit_grpLasso_fista(Xz, yz, alpha=alpha, group_size=group_size, weights=weights)
    return None, coef



def check_KKT(Xz, yz, coef_unit_full, active_g_indices, group_size, alpha, weights=None, tol=1e-4):
    residual = yz - Xz.dot(coef_unit_full)
    p_groups = Xz.shape[1] // group_size
    stats = np.zeros(p_groups)
    for g in range(p_groups):
        j0 = g * group_size
        j1 = j0 + group_size
        Xg = Xz[:, j0:j1]
        stats[g] = np.linalg.norm(Xg.T.dot(residual), 2) / Xg.shape[0]
    thresh = alpha if weights is None else alpha * np.asarray(weights)
    violated = np.where(stats > (thresh + tol))[0]
    violated = list(set(violated) - set(np.asarray(active_g_indices, dtype=int)))
    violated.sort()
    return violated



def SIS_grpLasso_KKT(X_expanded, y, group_size, alpha, weights=None, sis_ratio=1.0, max_iter=50,
                     sparse_l1=0.0):
    n, p_exp = X_expanded.shape
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
    Xz, yz, X_mean, X_std, y_mean, y_std = standardize_xy(X_expanded, y)
    # Mixed group sizes (e.g. aux groups with L lags + AR group with a different
    # count) break the uniform-group reshape used below. Pad the standardized
    # design with zero columns up to a multiple of group_size: a zero column is
    # penalized to exactly zero coefficient under the L2 group penalty, so the
    # fit, KKT checks, and group norms are unchanged. Padded columns are sliced
    # off before returning. When p_exp % group_size == 0 this is a no-op and the
    # code path is identical to the published configs.
    p_pad = int(np.ceil(p_exp / group_size) * group_size)
    if p_pad != p_exp:
        Xz = np.hstack([Xz, np.zeros((n, p_pad - p_exp))])
    p_groups = p_pad // group_size
    if weights is not None and len(weights) < p_groups:
        # pad weights for zero-column filler groups (never selected; value irrelevant)
        weights = np.concatenate([weights, np.ones(p_groups - len(weights))])
    # Per-column sparse_l1 (e.g. lag-decay): pad in lockstep with the zero columns.
    if np.ndim(sparse_l1) > 0:
        sparse_l1 = np.asarray(sparse_l1, dtype=float)
        assert sparse_l1.size == p_exp, "per-column sparse_l1 must match design width"
        if p_pad != p_exp:
            sparse_l1 = np.concatenate([sparse_l1, np.zeros(p_pad - p_exp)])
    coef_unit_full = np.zeros(p_pad, dtype=float)
    X3 = Xz.reshape(n, p_groups, group_size)
    X_group = np.sqrt(np.mean(X3 ** 2, axis=2))
    nSel = int(max(1, min(p_groups, np.floor((len(y) / group_size - 1) * sis_ratio))))
    violated_g = list(SIS(X_group, yz, nSel=nSel))
    if len(violated_g) == 0:
        coef = recover_coefficients(coef_unit_full[:p_exp], X_std, y_std)
        intercept = float(y_mean)
        return np.array([], dtype=int), coef, intercept, coef_unit_full[:p_exp].copy()
    active_g = np.array([], dtype=int)
    it = 0
    while len(violated_g) > 0:
        it += 1
        if it > max_iter:
            break
        active_g = np.array(sorted(set(active_g) | set(violated_g)), dtype=int)
        SIS_v = extend_indices(active_g, group_size)
        # subset weights to the active groups so per-group penalties stay aligned
        # with the column subset Xz[:, SIS_v] (no-op for uniform weights)
        w_sub = weights[active_g] if weights is not None else None
        # subset per-column sparse_l1 to the active columns (no-op for scalar)
        sl1_sub = sparse_l1[SIS_v] if np.ndim(sparse_l1) > 0 else sparse_l1
        _, coef_unit_sub = fit_grpLasso(Xz[:, SIS_v], yz, alpha=alpha,
                                        group_size=group_size, weights=w_sub,
                                        sparse_l1=sl1_sub)
        coef_unit_full[:] = 0.0
        coef_unit_full[SIS_v] = coef_unit_sub
        violated_g = check_KKT(Xz, yz, coef_unit_full, active_g, group_size, alpha, weights=weights)
    coef = recover_coefficients(coef_unit_full[:p_exp], X_std, y_std)
    intercept = recover_intercept(X_expanded, y, coef, X_mean=X_mean, y_mean=y_mean)
    active_v = np.where(coef != 0)[0]
    active_g_final = shrink_indices(active_v, group_size)
    return active_g_final, coef, intercept, coef_unit_full[:p_exp].copy()



def q_aggregate_weights(y_val, Yhat_val, total_step=10, selection=False, eps=1e-3, tau=1.0):
    y = np.asarray(y_val, float).reshape(-1, 1)
    XB = np.asarray(Yhat_val, float)
    _, M = XB.shape
    if M == 0:
        return np.array([], dtype=float)
    if M == 1:
        return np.array([1.0], dtype=float)
    tau = float(max(tau, 1e-12))
    if selection:
        mse = np.mean((y - XB) ** 2, axis=0)
        khat = int(np.argmin(mse))
        theta_hat = np.zeros(M, dtype=float)
        theta_hat[khat] = 1.0
        return theta_hat
    rss = np.sum((y - XB) ** 2, axis=0)
    a = -0.5 * rss / tau
    a = a - np.max(a)
    theta_hat = np.exp(a)
    theta_hat /= theta_hat.sum()
    theta_old = theta_hat.copy()
    beta_pred = XB @ theta_hat
    for _ in range(int(total_step)):
        Xbeta = beta_pred.reshape(-1, 1)
        adj = np.sum((Xbeta - XB) ** 2, axis=0) / (8.0 * tau)
        a = -0.5 * rss / tau + adj
        a = a - np.max(a)
        theta_hat = np.exp(a)
        theta_hat /= theta_hat.sum()
        beta_new = (XB @ theta_hat) * 0.25 + 0.75 * beta_pred
        if np.sum(np.abs(theta_hat - theta_old)) < eps:
            beta_pred = beta_new
            break
        theta_old = theta_hat.copy()
        beta_pred = beta_new
    return theta_hat



def estimate_R_hat_from_series(d0, dl, s):
    """Screened discrepancy: sum of squared differences over the s largest-|delta| entries."""
    if hasattr(d0, "index") and hasattr(dl, "index"):
        common = d0.index.intersection(dl.index)
        if len(common) == 0:
            return np.inf
        delta = (dl.loc[common] - d0.loc[common]).to_numpy()
        m = len(delta)
    else:
        d0 = np.asarray(d0, float).ravel()
        dl = np.asarray(dl, float).ravel()
        m = min(len(d0), len(dl))
        if m == 0:
            return np.inf
        delta = dl[:m] - d0[:m]
    s = int(min(s, m))
    if s <= 0:
        return 0.0
    idx = np.argpartition(np.abs(delta), -s)[-s:]
    return float(np.sum(delta[idx] ** 2))



def rank_candidate_states(d_by_state, target_state, all_states, s):
    """Return donor states sorted by R-hat ascending (excluding target)."""
    d0 = d_by_state[target_state]
    scores = []
    for st in all_states:
        if st == target_state or st not in d_by_state:
            continue
        scores.append((st, estimate_R_hat_from_series(d0, d_by_state[st], s=s)))
    scores.sort(key=lambda x: x[1])
    return [st for st, _ in scores]



def predict_named(X, feat_names_ref, coef_series, intercept):
    w = coef_series.reindex(feat_names_ref).fillna(0.0).to_numpy()
    return np.asarray(X) @ w + intercept


