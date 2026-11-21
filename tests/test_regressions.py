"""Independent regression cases for package behavior and numerical fixes."""

import numpy as np
import pandas as pd
import pytest

from glcast_method import CombinerConfig, MethodConfig, _grid, run_walkforward
from glcast_method._alpha import pro_pooled_alphas
from glcast_method._design import compute_walkforward_schedule
from glcast_method._solver import _fit_grpLasso_fista, check_KKT


@pytest.mark.parametrize("column_penalty", [1.0, np.array([1.0, 0.5])])
def test_sparse_group_fit_matches_closed_form(column_penalty):
    # For X.T @ X / n = I, the minimizer is the sparse-group prox of X.T @ y / n.
    values = np.array([3.0, 4.0])
    X = np.sqrt(2) * np.eye(2)
    y = np.sqrt(2) * values
    soft = np.maximum(values - column_penalty, 0)
    expected = (1 - 1 / np.linalg.norm(soft)) * soft
    actual = _fit_grpLasso_fista(X, y, alpha=1, group_size=2, sparse_l1=column_penalty)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    gradient = actual - values + actual / np.linalg.norm(actual) + column_penalty
    np.testing.assert_allclose(gradient, 0, atol=1e-12)


def test_sparse_group_kkt_accounts_for_column_penalty():
    X = np.sqrt(2) * np.eye(2)
    y = np.sqrt(2) * np.array([0.8, 0.8])
    assert check_KKT(X, y, np.zeros(2), [], 2, alpha=0.5, sparse_l1=0.6) == []
    assert check_KKT(X, y, np.zeros(2), [], 2, alpha=0.5, sparse_l1=0) == [0]


def test_alpha_without_history_cannot_win():
    origins = pd.to_datetime(["2020-01-01", "2020-01-02"])
    predictions = pd.DataFrame(
        [
            dict(
                origin=origins[0],
                state="A",
                alpha=0.1,
                split="test",
                y_true=1.0,
                y_pred=2.0,
            ),
            dict(
                origin=origins[1],
                state="A",
                alpha=0.2,
                split="test",
                y_true=1.0,
                y_pred=100.0,
            ),
        ]
    )
    picks = pro_pooled_alphas(predictions, origins)
    assert np.isnan(picks.loc[0, "alpha"])
    assert picks.loc[1, "alpha"] == 0.1


def test_alpha_selection_with_no_observed_history_returns_nan():
    origins = pd.to_datetime(["2020-01-01", "2020-01-02"])
    predictions = pd.DataFrame(
        [
            dict(
                origin=origins[1],
                state="A",
                alpha=0.1,
                split="test",
                y_true=1.0,
                y_pred=1.0,
            )
        ]
    )
    assert pro_pooled_alphas(predictions, origins)["alpha"].isna().all()


@pytest.mark.parametrize(
    "changes",
    [
        {"alpha_y_scale": "false"},
        {"alpha_y_scale": 1},
        {"combiner": 1},
        {"alphas": [True]},
        {"alphas": ["0.1"]},
        {"lag_decay": "0.7"},
        {"group_weights": {"AR": "1.0"}},
        {"group_weights": [1.0]},
    ],
)
def test_config_rejects_ambiguous_types(changes):
    with pytest.raises(ValueError):
        MethodConfig(**changes)


def test_combiner_rejects_string_penalty():
    with pytest.raises(ValueError):
        CombinerConfig(ridge="1000")


def _short_series():
    dates = pd.date_range("2020-01-01", periods=14)
    x = np.arange(14, dtype=float)
    target = pd.DataFrame({"A": 2 + x + np.sin(x)}, index=dates)
    features = {"aux": pd.DataFrame({"A": x}, index=dates)}
    config = MethodConfig(
        unit="D",
        train_window=12,
        val_window=2,
        test_window=2,
        feature_lags=(0,),
        ar_lags=(1,),
        alphas=(0.1,),
        fixed_alpha=0.1,
        serve_steps=(1,),
    )
    return target, features, config


def test_fixed_alpha_serves_first_and_only_origin():
    target, features, config = _short_series()
    schedule = compute_walkforward_schedule(target.index, 12, 2, 1, "D")
    assert schedule == [target.index[11]]
    result = run_walkforward(target, features, config)
    assert result.alpha_picks["alpha"].tolist() == [0.1]
    assert result.predictions["origin"].tolist() == schedule
    assert result.predictions["time_value"].tolist() == [target.index[12]]


def test_solver_error_includes_state_origin_and_alpha(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("solver failure")

    monkeypatch.setattr(_grid, "SIS_grpLasso_KKT", fail)
    target, features, config = _short_series()
    with pytest.raises(
        RuntimeError, match="state 'A', origin 2020-01-12, alpha 0.1: solver failure"
    ):
        run_walkforward(target, features, config)
