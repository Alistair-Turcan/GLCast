"""Select donor models and combine their predictions with the local model."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from scipy.optimize import nnls


def q_aggregate_weights(
    y_val, Yhat_val, total_step=10, selection=False, eps=1e-3, tau=1.0
):
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


@dataclass(frozen=True)
class CombinerConfig:
    library: int = 20
    k: int = 5
    ridge: float = 1000.0
    shrinkage: float = 10.0
    tau_z: float = 0.5

    def __post_init__(self):
        if any(
            not isinstance(v, int) or isinstance(v, bool)
            for v in (self.library, self.k)
        ):
            raise ValueError("library and k must be integers")
        if self.library < 0 or self.k < 0 or self.k > self.library:
            raise ValueError("Require 0 <= k <= library")
        parameters = [self.ridge, self.shrinkage, self.tau_z]
        if (
            any(not isinstance(v, Real) or isinstance(v, bool) for v in parameters)
            or not np.isfinite(parameters).all()
        ):
            raise ValueError("Combiner parameters must be finite")
        if self.ridge < 0 or self.shrinkage < 0 or self.tau_z <= 0:
            raise ValueError("Require ridge/shrinkage >= 0 and tau_z > 0")


@dataclass
class Combination:
    predictions: np.ndarray
    weights: np.ndarray
    names: tuple[str, ...]
    gamma: float
    stacking_weights: np.ndarray
    q_weights: np.ndarray
    candidate_predictions: np.ndarray


def stacking_nnls(y, F, lam):
    """Nonnegative regression; self is column zero and is not ridge-penalized."""
    n, J = F.shape
    d = np.ones(J)
    d[0] = 0.0
    A = np.vstack([F, np.sqrt(lam * n) * np.diag(d)])
    b = np.concatenate([y, np.zeros(J)])
    return nnls(A, b)[0]


def combine_candidates(
    y_val,
    validation_predictions,
    future_predictions,
    names: Sequence[str],
    config: CombinerConfig | None = None,
):
    """Combine self + discrepancy-ordered donors, using validation labels only.

    Matrices have shape (rows, candidates); column zero is self. Future labels
    are neither accepted nor needed here. Returns None if self is nonfinite.
    Weights are nonnegative but are not forced to sum to one. Predictions are raw
    (unclipped).
    """
    cfg = config or CombinerConfig()
    y = np.asarray(y_val, dtype=float)
    V = np.asarray(validation_predictions, dtype=float)
    T = np.asarray(future_predictions, dtype=float)
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("y_val must be a nonempty finite one-dimensional vector")
    if V.ndim != 2 or T.ndim != 2 or V.shape[0] != len(y):
        raise ValueError(
            "Expected 2-D candidate matrices with len(y_val) validation rows"
        )
    if V.shape[1] == 0 or T.shape[1] != V.shape[1] or len(names) != V.shape[1]:
        raise ValueError("Candidate counts must agree and include self as column zero")
    if T.shape[0] == 0:
        raise ValueError("At least one prediction row is required")
    if any(not isinstance(n, str) or not n for n in names) or len(set(names)) != len(
        names
    ):
        raise ValueError("Candidate names must be unique strings")
    V, T = V[:, : cfg.library + 1], T[:, : cfg.library + 1]
    selected_names = list(names[: cfg.library + 1])
    if not (np.isfinite(V[:, 0]).all() and np.isfinite(T[:, 0]).all()):
        return None
    ok = np.isfinite(V).all(axis=0) & np.isfinite(T).all(axis=0)
    V, T = V[:, ok], T[:, ok]
    selected_names = [n for n, keep in zip(selected_names, ok) if keep]
    if V.shape[1] - 1 > cfg.k:
        rmse = np.sqrt(((V[:, 1:] - y[:, None]) ** 2).mean(axis=0))
        top = np.sort(np.argsort(rmse, kind="stable")[: cfg.k])
        keep = np.concatenate([[0], 1 + top])
        V, T = V[:, keep], T[:, keep]
        selected_names = [selected_names[i] for i in keep]
    w_stack = stacking_nnls(y, V, cfg.ridge)
    scale = float(y.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    w_q = q_aggregate_weights(
        y / scale, V / scale, total_step=10, selection=False, eps=1e-3, tau=cfg.tau_z**2
    )
    if len(w_q) == 0:
        return None
    gamma = len(y) / (len(y) + cfg.shrinkage)
    w = gamma * w_stack + (1.0 - gamma) * w_q
    return Combination(T @ w, w, tuple(selected_names), gamma, w_stack, w_q, T)


def _combine_origin(block, config=None):
    """Combine a block whose donor arrays contain only the selected alpha."""
    cfg = config or CombinerConfig()
    donors = block["donors"][: cfg.library]
    validation = np.column_stack(
        [block["self_val"], *block["donor_val"][: len(donors)]]
    )
    predictions = np.column_stack(
        [block["self_test"], *block["donor_test"][: len(donors)]]
    )
    return combine_candidates(
        block["y_val"], validation, predictions, ["self", *donors], cfg
    )


def combine_origin(block, alpha_index, config=None):
    """Compatibility adapter for blocks with donor-by-alpha-by-row arrays.

    New callers should pass prediction matrices directly to combine_candidates.
    """
    selected = dict(block)
    selected["donor_val"] = np.asarray(block["donor_val"])[:, alpha_index, :]
    selected["donor_test"] = np.asarray(block["donor_test"])[:, alpha_index, :]
    return _combine_origin(selected, config)
