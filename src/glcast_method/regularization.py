"""Adaptive group weights and low-level regularized fitting."""

import re

import numpy as np
import pandas as pd

from ._solver import SIS_grpLasso_KKT


def feature_group(name):
    return "AR" if name.startswith("AR_") else re.sub(r"_lag\d+$", "", name)


def adaptive_weights(
    fits, group_order, *, alpha_ref=None, alpha_picks=None, floor=0.05
):
    """Analysis-wide retrospective weights from first-pass uniform-penalty fits.

    fits columns: origin,state,alpha,feature,coef. Pick either a fixed alpha_ref
    or an alpha_picks table with columns origin,alpha (NaN picks are omitted).
    Frequency pools existing origin/state/group fits; missing groups are assigned
    frequency zero. Returns a group-name -> positive mean-rescaled weight mapping.
    """
    required = {"origin", "state", "alpha", "feature", "coef"}
    if not required.issubset(fits.columns):
        raise ValueError(f"fits requires columns {sorted(required)}")
    if (alpha_ref is None) == (alpha_picks is None):
        raise ValueError("Supply exactly one of alpha_ref or alpha_picks")
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError("floor must be finite and positive")
    if not group_order or len(set(group_order)) != len(group_order):
        raise ValueError("group_order must contain unique group names")
    f = fits.copy()
    if alpha_ref is not None:
        f = f[np.isclose(f["alpha"], alpha_ref)].copy()
    else:
        p = alpha_picks[["origin", "alpha"]].dropna().copy()
        if p["origin"].duplicated().any():
            raise ValueError("alpha_picks must have at most one row per origin")
        f["origin"] = pd.to_datetime(f["origin"])
        p["origin"] = pd.to_datetime(p["origin"])
        f = f.merge(
            p.rename(columns={"alpha": "chosen_alpha"}), on="origin", how="inner"
        )
        f = f[np.isclose(f["alpha"], f["chosen_alpha"])].copy()
    if f.empty:
        raise ValueError("No first-pass fits match the requested alpha selection")
    if not np.isfinite(f["coef"]).all():
        raise ValueError("First-pass coefficients must be finite")
    if f.duplicated(["origin", "state", "feature"]).any():
        raise ValueError("Duplicate selected first-pass fit coordinates")
    f["group"] = f["feature"].map(feature_group)
    selected = (
        f.assign(nz=f["coef"].abs() > 0)
        .groupby(["origin", "state", "group"])["nz"]
        .any()
    )
    frequency = selected.groupby("group").mean().reindex(group_order).fillna(0.0)
    w = 1.0 / (frequency.to_numpy() + floor)
    w /= w.mean()
    return dict(zip(group_order, map(float, w)))


def choose_lags(deployed_lags, expanded_lags, deployed_error, expanded_error):
    """Choose the expanded lag set only when its supplied error is smaller."""
    if not np.isfinite([deployed_error, expanded_error]).all():
        raise ValueError("Both comparison errors must be finite")
    if min(deployed_error, expanded_error) < 0:
        raise ValueError("Comparison errors must be nonnegative")
    for lags in (deployed_lags, expanded_lags):
        if (
            not lags
            or len(set(lags)) != len(lags)
            or any(not isinstance(l, int) or isinstance(l, bool) or l < 0 for l in lags)
        ):
            raise ValueError("Lag sets must contain nonnegative integers")
    return tuple(expanded_lags if expanded_error < deployed_error else deployed_lags)


def column_penalties(feature_names, alpha, y_std, multiplier=0.7):
    if (
        not np.isfinite([alpha, y_std, multiplier]).all()
        or min(alpha, multiplier) < 0
        or y_std < 0
    ):
        raise ValueError("Invalid penalty scale")
    scale = y_std if y_std > 0 else 1.0
    lag_mult = []
    for name in feature_names:
        match = re.search(r"_lag(\d+)$", name)
        if match is None:
            raise ValueError(f"Missing lag suffix in feature name: {name}")
        lag_mult.append(1.0 if name.startswith("AR_") else int(match.group(1)) + 1)
    return multiplier * (alpha / scale) * np.asarray(lag_mult)


def fit_classic(
    X,
    y,
    feature_names,
    alpha,
    group_weights=None,
    multiplier=0.7,
    *,
    alpha_y_scale=False,
):
    """Fit group or sparse-group lasso with named, contiguous feature groups.

    Auxiliary streams must have the same lag count; AR must be the last group
    and no larger than an auxiliary group. alpha is on the standardized-target
    scale by default, matching MethodConfig. Set alpha_y_scale=True to divide it
    by the training target's standard deviation. multiplier controls the
    additional column penalty; zero gives ordinary group lasso.

    Returns active group indices, original-scale coefficients, intercept, and
    standardized coefficients.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim != 2 or y.ndim != 1 or X.shape != (len(y), len(feature_names)):
        raise ValueError("X, y and feature_names dimensions disagree")
    if len(y) < 10 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Fit requires at least 10 finite complete rows")
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    if not isinstance(alpha_y_scale, bool):
        raise ValueError("alpha_y_scale must be a boolean")
    groups = [feature_group(n) for n in feature_names]
    order = list(dict.fromkeys(groups))
    if not order or order[-1] != "AR":
        raise ValueError("AR must be the last group")
    if groups != [g for g in order for _ in range(groups.count(g))]:
        raise ValueError("Feature columns must be contiguous by group")
    counts = [groups.count(g) for g in order]
    group_size = counts[0]
    if any(n != group_size for n in counts[:-1]) or counts[-1] > group_size:
        raise ValueError("Auxiliary groups must be equal-sized and AR no larger")
    weights = None
    if group_weights is not None:
        try:
            weights = np.array([group_weights[g] for g in order], dtype=float)
        except KeyError as e:
            raise ValueError(f"Missing group weight: {e}") from e
        if not np.isfinite(weights).all() or min(weights) <= 0:
            raise ValueError("Group weights must be finite and positive")
    scale = (float(y.std(ddof=0)) or 1.0) if alpha_y_scale else 1.0
    penalties = column_penalties(feature_names, alpha, scale, multiplier)
    return SIS_grpLasso_KKT(
        X, y, group_size, alpha / scale, weights=weights, sparse_l1=penalties
    )
