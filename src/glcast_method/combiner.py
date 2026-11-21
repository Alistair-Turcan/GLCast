"""Validation-ranked donor selection and the as-run shrunk-stacking combiner."""
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.optimize import nnls
from ._modern.harness import q_aggregate_weights


@dataclass(frozen=True)
class CombinerConfig:
    library: int = 20
    k: int = 5
    ridge: float = 1000.0
    shrinkage: float = 10.0
    tau_z: float = 0.5

    def __post_init__(self):
        if any(not isinstance(v, int) or isinstance(v, bool) for v in (self.library, self.k)):
            raise ValueError("library and k must be integers")
        if self.library < 0 or self.k < 0 or self.k > self.library:
            raise ValueError("Require 0 <= k <= library")
        if not np.isfinite([self.ridge, self.shrinkage, self.tau_z]).all():
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


def combine_candidates(y_val, validation_predictions, future_predictions,
                       names: Sequence[str], config: CombinerConfig | None = None):
    """Combine self + discrepancy-ordered donors, using validation labels only.

    Matrices have shape (rows, candidates); column zero is self. Future labels
    are neither accepted nor needed here. Returns None if self is nonfinite.
    Weights are nonnegative but are not forced to sum to one. Predictions are raw
    (unclipped), preserving the existing method's combination step.
    """
    cfg = config or CombinerConfig()
    y = np.asarray(y_val, dtype=float)
    V = np.asarray(validation_predictions, dtype=float)
    T = np.asarray(future_predictions, dtype=float)
    if y.ndim != 1 or len(y) == 0 or not np.isfinite(y).all():
        raise ValueError("y_val must be a nonempty finite one-dimensional vector")
    if V.ndim != 2 or T.ndim != 2 or V.shape[0] != len(y):
        raise ValueError("Expected 2-D candidate matrices with len(y_val) validation rows")
    if V.shape[1] == 0 or T.shape[1] != V.shape[1] or len(names) != V.shape[1]:
        raise ValueError("Candidate counts must agree and include self as column zero")
    if T.shape[0] == 0:
        raise ValueError("At least one prediction row is required")
    if len(set(names)) != len(names) or any(not isinstance(n, str) for n in names):
        raise ValueError("Candidate names must be unique strings")
    V, T = V[:, :cfg.library + 1], T[:, :cfg.library + 1]
    selected_names = list(names[:cfg.library + 1])
    if not (np.isfinite(V[:, 0]).all() and np.isfinite(T[:, 0]).all()):
        return None
    ok = np.isfinite(V).all(axis=0) & np.isfinite(T).all(axis=0)
    V, T = V[:, ok], T[:, ok]
    selected_names = [n for n, keep in zip(selected_names, ok) if keep]
    if V.shape[1] - 1 > cfg.k:
        rmse = np.sqrt(((V[:, 1:] - y[:, None]) ** 2).mean(axis=0))
        top = np.sort(np.argsort(rmse, kind="stable")[:cfg.k])
        keep = np.concatenate([[0], 1 + top])
        V, T = V[:, keep], T[:, keep]
        selected_names = [selected_names[i] for i in keep]
    w_stack = stacking_nnls(y, V, cfg.ridge)
    scale = float(y.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    w_q = q_aggregate_weights(y / scale, V / scale, total_step=10,
                              selection=False, eps=1e-3, tau=cfg.tau_z ** 2)
    if len(w_q) == 0:
        return None
    gamma = len(y) / (len(y) + cfg.shrinkage)
    w = gamma * w_stack + (1.0 - gamma) * w_q
    return Combination(T @ w, w, tuple(selected_names), gamma, w_stack, w_q, T)


def combine_origin(block: dict, alpha_index: int, config: CombinerConfig | None = None):
    """Adapter for the recovered geographic candidate-block dictionary."""
    cfg = config or CombinerConfig()
    donors = block["donors"][:cfg.library]
    V = np.column_stack([block["self_val"]] +
                        [block["donor_val"][j, alpha_index] for j in range(len(donors))])
    T = np.column_stack([block["self_test"]] +
                        [block["donor_test"][j, alpha_index] for j in range(len(donors))])
    return combine_candidates(block["y_val"], V, T, ["self"] + list(donors), cfg)
