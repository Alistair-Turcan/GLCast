"""Screened group and sparse-group lasso fitting."""

import numpy as np

from ._design import standardize_xy

try:
    import celer
except ModuleNotFoundError as exc:
    if exc.name != "celer":
        raise
    celer = None


def extend_indices(group_indices, group_size):
    indices = np.asarray(group_indices, dtype=int)
    return (indices[:, None] * group_size + np.arange(group_size)).ravel()


def shrink_indices(column_indices, group_size):
    return np.unique(np.asarray(column_indices, dtype=int) // group_size)


def SIS(X, y, nSel):
    """Rank screening features by absolute correlation with the target."""
    y_std = np.std(y)
    if y_std == 0:
        return np.array([], dtype=int)
    correlations = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        x_std = np.std(X[:, j])
        if x_std:
            correlations[j] = abs(np.cov(X[:, j], y, bias=True)[0, 1] / (x_std * y_std))
    return np.argsort(correlations)[::-1][:nSel].astype(int)


def _group_soft_threshold(values, penalty):
    norm = np.linalg.norm(values)
    if norm == 0 or norm <= penalty:
        return np.zeros_like(values)
    return (1.0 - penalty / norm) * values


def _soft_threshold(values, penalty):
    return np.sign(values) * np.maximum(np.abs(values) - penalty, 0.0)


def _fit_grpLasso_fista(
    Xz, yz, alpha, group_size, weights=None, max_iter=10000, tol=1e-6, sparse_l1=0.0
):
    """Minimize squared loss plus group L2 and optional per-column L1 penalties.

    The sparse-group proximal operator applies elementwise soft thresholding
    before group shrinkage. Xz and yz must already be standardized.
    """
    Xz, yz = np.asarray(Xz, dtype=float), np.asarray(yz, dtype=float)
    n, p = Xz.shape
    n_groups = p // group_size
    group_weights = (
        np.ones(n_groups) if weights is None else np.asarray(weights, dtype=float)
    )
    if group_weights.size == 1:
        group_weights = np.full(n_groups, group_weights.item())
    l1 = np.asarray(sparse_l1, dtype=float)
    step = 1.0 / max(np.linalg.norm(Xz, ord=2) ** 2 / max(n, 1), 1e-8)
    beta = np.zeros(p)
    z, t = beta.copy(), 1.0
    for _ in range(max_iter):
        gradient = Xz.T @ (Xz @ z - yz) / max(n, 1)
        beta_next = _soft_threshold(z - step * gradient, step * l1)
        for group in range(n_groups):
            columns = slice(group * group_size, (group + 1) * group_size)
            beta_next[columns] = _group_soft_threshold(
                beta_next[columns], step * alpha * group_weights[group]
            )
        if np.linalg.norm(beta_next - beta) <= tol * max(1.0, np.linalg.norm(beta)):
            return beta_next
        # Restart acceleration when momentum points against the proximal step.
        # This avoids long oscillations with highly correlated lagged predictors.
        if np.dot(z - beta_next, beta_next - beta) > 0:
            t_next = 1.0
            z = beta_next.copy()
        else:
            t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = beta_next + ((t - 1.0) / t_next) * (beta_next - beta)
        beta, t = beta_next, t_next
    raise RuntimeError(f"Group lasso did not converge within {max_iter} iterations")


def fit_grpLasso(Xz, yz, alpha, group_size, weights=None, sparse_l1=0.0):
    """Use celer for ordinary group lasso when installed, otherwise FISTA."""
    if celer is not None and not np.any(np.asarray(sparse_l1) > 0):
        model = celer.GroupLasso(
            groups=group_size,
            alpha=alpha,
            tol=1e-4,
            fit_intercept=False,
            weights=weights,
        )
        model.fit(Xz, yz)
        return model, model.coef_.copy()
    return None, _fit_grpLasso_fista(
        Xz, yz, alpha, group_size, weights, sparse_l1=sparse_l1
    )


def check_KKT(
    Xz,
    yz,
    coefficients,
    active_groups,
    group_size,
    alpha,
    weights=None,
    tol=1e-4,
    sparse_l1=0.0,
):
    """Find inactive groups that violate the group/sparse-group KKT condition."""
    residual = yz - Xz @ coefficients
    gradient = Xz.T @ residual / Xz.shape[0]
    gradient = _soft_threshold(gradient, np.asarray(sparse_l1, dtype=float))
    scores = np.linalg.norm(gradient.reshape(-1, group_size), axis=1)
    threshold = alpha if weights is None else alpha * np.asarray(weights)
    violated = np.flatnonzero(scores > threshold + tol)
    return sorted(set(violated) - set(active_groups))


def SIS_grpLasso_KKT(
    X_expanded,
    y,
    group_size,
    alpha,
    weights=None,
    sis_ratio=1.0,
    max_iter=50,
    sparse_l1=0.0,
):
    """Fit screened groups, add KKT violations, and return original-scale coefficients.

    A short final group is padded with zeros internally. Returns active group
    indices, original-scale coefficients, intercept, and standardized coefficients.
    """
    X_expanded, y = np.asarray(X_expanded, dtype=float), np.asarray(y, dtype=float)
    n, width = X_expanded.shape
    if (
        not isinstance(group_size, int)
        or isinstance(group_size, bool)
        or group_size < 1
    ):
        raise ValueError("group_size must be a positive integer")
    Xz, yz, X_mean, X_std, y_mean, y_std = standardize_xy(X_expanded, y)
    padded_width = int(np.ceil(width / group_size)) * group_size
    if padded_width != width:
        Xz = np.pad(Xz, ((0, 0), (0, padded_width - width)))
    n_groups = padded_width // group_size
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_groups,):
            raise ValueError("One group weight is required for each design group")
    l1 = np.asarray(sparse_l1, dtype=float)
    if l1.ndim:
        if l1.shape != (width,):
            raise ValueError("Per-column sparse_l1 must match the design width")
        l1 = np.pad(l1, (0, padded_width - width))

    beta = np.zeros(padded_width)
    screening_design = np.sqrt(
        np.mean(Xz.reshape(n, n_groups, group_size) ** 2, axis=2)
    )
    n_selected = int(
        max(1, min(n_groups, np.floor((len(y) / group_size - 1) * sis_ratio)))
    )
    violated = list(SIS(screening_design, yz, n_selected))
    active = np.array([], dtype=int)
    for _ in range(max_iter):
        if not violated:
            break
        active = np.array(sorted(set(active) | set(violated)), dtype=int)
        columns = extend_indices(active, group_size)
        subset_weights = weights[active] if weights is not None else None
        subset_l1 = l1[columns] if l1.ndim else l1
        _, fitted = fit_grpLasso(
            Xz[:, columns],
            yz,
            alpha,
            group_size,
            weights=subset_weights,
            sparse_l1=subset_l1,
        )
        beta[:] = 0.0
        beta[columns] = fitted
        violated = check_KKT(
            Xz, yz, beta, active, group_size, alpha, weights=weights, sparse_l1=l1
        )
    if violated:
        raise RuntimeError(
            f"Group screening did not converge within {max_iter} iterations"
        )
    coefficients = (y_std / X_std) * beta[:width]
    intercept = float(y_mean - X_mean @ coefficients)
    return (
        shrink_indices(np.flatnonzero(coefficients), group_size),
        coefficients,
        intercept,
        beta[:width].copy(),
    )
