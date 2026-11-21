import numpy as np
import pytest

from glcast_method import CombinerConfig, combine_candidates, combine_origin
from glcast_method.combiner import q_aggregate_weights, stacking_nnls


def test_donor_only_ridge():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    F = np.column_stack([y, y])
    w = stacking_nnls(y, F, 1000.0)
    np.testing.assert_allclose(w, [1.0, 0.0], atol=1e-10)


def test_self_only_gamma_and_no_normalization():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    V = (y / 2)[:, None]
    result = combine_candidates(y, V, V, ["self"])
    assert result.gamma == 4 / 14
    np.testing.assert_allclose(result.stacking_weights, [2.0])
    np.testing.assert_allclose(result.q_weights, [1.0])
    np.testing.assert_allclose(result.weights, [1.0 + 4 / 14])
    assert result.weights.sum() > 1


def test_stable_skill_selection_and_library_limit():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    V = np.column_stack([y + 1, y + 8, y + 0.1, y - 0.1, y])
    cfg = CombinerConfig(library=3, k=2)
    result = combine_candidates(
        y, V, V, ["self", "far", "plus", "minus", "outside"], cfg
    )
    assert result.names == ("self", "plus", "minus")
    assert np.all(result.weights >= 0)
    cfg = CombinerConfig(library=3, k=1)
    tied = combine_candidates(
        y,
        np.column_stack([y + 1, y + 2, y + 2, y + 3]),
        np.column_stack([y + 1, y + 2, y + 2, y + 3]),
        ["self", "first", "second", "third"],
        cfg,
    )
    assert tied.names == ("self", "first")


def test_nonfinite_donors_excluded_but_self_required():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    V = np.column_stack([y, y, y])
    T = V.copy()
    T[0, 1] = np.nan
    result = combine_candidates(y, V, T, ["self", "missing", "kept"])
    assert result.names == ("self", "kept")
    V[0, 0] = np.nan
    assert combine_candidates(y, V, T, ["self", "missing", "kept"]) is None


def test_constant_validation_sd_fallback():
    y = np.ones(4)
    V = np.column_stack([y, y * 2])
    result = combine_candidates(y, V, V, ["self", "donor"])
    np.testing.assert_allclose(result.q_weights, q_aggregate_weights(y, V, tau=0.25))
    assert np.isfinite(result.predictions).all()


def test_origin_adapter():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    block = {
        "y_val": y,
        "self_val": y,
        "self_test": y,
        "donors": ["D"],
        "donor_val": (y + 1)[None, None, :],
        "donor_test": (y + 2)[None, None, :],
    }
    adapted = combine_origin(block, 0)
    direct = combine_candidates(
        y, np.column_stack([y, y + 1]), np.column_stack([y, y + 2]), ["self", "D"]
    )
    np.testing.assert_array_equal(adapted.weights, direct.weights)


def test_q_iteration_reference_formula():
    y = np.array([1.0, 3.0, 2.0, 4.0])
    V = np.column_stack([y + 0.25, y - 0.75, np.ones(4)])
    tau = 0.25
    rss = np.sum((y[:, None] - V) ** 2, axis=0)
    logits = -rss / (2 * tau)
    theta = np.exp(logits - logits.max())
    theta /= theta.sum()
    previous = theta.copy()
    aggregate = V @ theta
    for _ in range(10):
        logits = -rss / (2 * tau) + np.sum((aggregate[:, None] - V) ** 2, axis=0) / (
            8 * tau
        )
        theta = np.exp(logits - logits.max())
        theta /= theta.sum()
        updated = 0.25 * (V @ theta) + 0.75 * aggregate
        if np.abs(theta - previous).sum() < 1e-3:
            break
        previous, aggregate = theta.copy(), updated
    np.testing.assert_array_equal(q_aggregate_weights(y, V, tau=tau), theta)


@pytest.mark.parametrize(
    "y,V,T,names",
    [
        ([np.nan], [[1]], [[1]], ["self"]),
        ([1, 2], [[1]], [[1]], ["self"]),
        ([1], [[1, 2]], [[1]], ["self"]),
        ([1], [[1, 2]], [[1, 2]], ["same", "same"]),
        ([1], [[1]], np.empty((0, 1)), ["self"]),
    ],
)
def test_invalid_inputs(y, V, T, names):
    with pytest.raises(ValueError):
        combine_candidates(y, V, T, names)


def test_bad_config():
    with pytest.raises(ValueError):
        CombinerConfig(library=1, k=2)
