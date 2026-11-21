import numpy as np
import pandas as pd
import pytest
from glcast_method import adaptive_weights, column_penalties, choose_lags, fit_classic
from glcast_method._classic import harness


def pilot_table():
    return pd.DataFrame([
        ("2020-01-01", "A", .7, "x_lag0", 1.),
        ("2020-01-01", "A", .7, "x_lag1", 0.),
        ("2020-01-01", "A", .7, "AR_lag1", 0.),
        ("2020-01-02", "A", .7, "x_lag0", 0.),
        ("2020-01-02", "A", .7, "x_lag1", 0.),
        ("2020-01-02", "A", .7, "AR_lag1", 0.),
    ], columns=["origin", "state", "alpha", "feature", "coef"])


def test_adaptive_frequency_and_named_alignment():
    weights = adaptive_weights(pilot_table(), ["missing", "AR", "x"], alpha_ref=.7)
    expected = np.array([1/.05, 1/.05, 1/.55])
    expected /= expected.mean()
    assert list(weights) == ["missing", "AR", "x"]
    np.testing.assert_allclose(list(weights.values()), expected)


def test_adaptive_selected_alpha():
    fits = pilot_table()
    picks = pd.DataFrame({"origin": ["2020-01-01", "2020-01-02"], "alpha": [.7, np.nan]})
    weights = adaptive_weights(fits, ["x", "AR"], alpha_picks=picks)
    expected = np.array([1/1.05, 1/.05])
    np.testing.assert_allclose(list(weights.values()), expected/expected.mean())
    with pytest.raises(ValueError):
        adaptive_weights(fits, ["x"], alpha_ref=.7, alpha_picks=picks)


def test_column_penalties_and_zero_sd():
    names = ["x_lag0", "x_lag1", "x_lag2", "AR_lag2"]
    np.testing.assert_allclose(column_penalties(names, .7, 2),
                               [.245, .490, .735, .245])
    np.testing.assert_allclose(column_penalties(names, .7, 0),
                               [.49, .98, 1.47, .49])


def test_lag_gate_strict_and_tie():
    assert choose_lags([0], [0, 1, 2, 3], 2., 1.) == (0, 1, 2, 3)
    assert choose_lags([0], [0, 1, 2, 3], 1., 1.) == (0,)
    assert choose_lags([0], [0, 1], 1., 2.) == (0,)
    with pytest.raises(ValueError):
        choose_lags([0], [0, 1], np.nan, 1.)


def test_classic_fit_matches_recovered_solver_and_padding():
    # Simple deterministic numerical fixture, not a research dataset.
    x = np.arange(12, dtype=float)
    X = np.column_stack([x, x[::-1], x % 3])
    y = x + 1.
    names = ["x_lag0", "x_lag1", "AR_lag1"]
    w = {"x": .5, "AR": 1.5}
    actual = fit_classic(X, y, names, .7, w)
    sd = y.std()
    expected = harness.SIS_grpLasso_KKT(
        X, y, 2, .7/sd, weights=np.array([.5, 1.5]),
        sparse_l1=column_penalties(names, .7, sd))
    for a, e in zip(actual, expected):
        np.testing.assert_array_equal(a, e)
    assert len(actual[1]) == 3  # pad never leaks into returned coefficients
    assert np.isfinite(actual[1]).all()


def test_classic_rejects_oversized_ar():
    X = np.ones((12, 3))
    with pytest.raises(ValueError):
        fit_classic(X, np.arange(12.), ["x_lag0", "AR_lag1", "AR_lag2"], .7)
