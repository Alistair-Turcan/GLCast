from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from glcast_method import MethodConfig, load_long_csv, run_walkforward
from glcast_method.pipeline import validate_wides
from glcast_method._modern.harness import compute_walkforward_schedule, SIS_grpLasso_KKT
from glcast_method._modern import harness


def test_all_configuration_templates():
    root = Path(__file__).resolve().parents[1]
    for path in (root / "configs").glob("*.json"):
        assert MethodConfig.from_json(path).backend in ("modern", "classic")


def test_config_rejects_unsupported_modes():
    with pytest.raises(ValueError):
        MethodConfig(group_weights={"x": 1.})
    with pytest.raises(ValueError):
        MethodConfig(serve_steps=(5,))
    with pytest.raises(ValueError):
        MethodConfig(unit="W-SAT")
    with pytest.raises(ValueError):
        MethodConfig(feature_lags=(0, 1), ar_lags=(2,))


def test_saturday_schedule():
    dates = pd.date_range("2020-01-04", periods=70, freq="7D")
    origins = compute_walkforward_schedule(dates, 55, 4, 1, "W")
    assert origins[0] == dates[54]
    assert all(o.day_of_week == 5 for o in origins)
    assert (origins[1] - origins[0]).days == 7


def test_input_validator_accepts_missing_measurements_not_gaps():
    dates = pd.date_range("2020-01-04", periods=3, freq="7D")
    target = pd.DataFrame({"A": [1., np.nan, 2.]}, index=dates)
    validate_wides(target, {"x": target.copy()}, MethodConfig())
    with pytest.raises(ValueError):
        validate_wides(target.iloc[[0, 2]], {"x": target.iloc[[0, 2]]}, MethodConfig())
    with pytest.raises(ValueError):
        validate_wides(target, {"AR": target.copy()}, MethodConfig())


def test_loader_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "fixture.csv"
    pd.DataFrame({"date": ["2020-01-04"]*2, "state": ["A"]*2,
                  "target": [1., 2.], "x": [3., 4.]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_long_csv(path, target="target", auxiliaries=["x"])


def test_insufficient_history(tmp_path):
    dates = pd.date_range("2020-01-04", periods=3, freq="7D")
    target = pd.DataFrame({"A": [1., 2., 3.]}, index=dates)
    with pytest.raises(ValueError, match="No walk-forward origins"):
        run_walkforward(target, {"x": target.copy()})


def test_modern_without_celer(monkeypatch):
    monkeypatch.setattr(harness, "celer", None)
    x = np.arange(12, dtype=float)
    X = np.column_stack([x, x % 3])
    result = SIS_grpLasso_KKT(X, x + 1, 1, .1)
    assert np.isfinite(result[1]).all()
    assert len(result[1]) == 2
