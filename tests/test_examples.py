import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_quickstart(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "quickstart"
    spec = importlib.util.spec_from_file_location("quickstart", example / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    data = pd.read_csv(example / "ctis_aggregate.csv")
    assert len(data) == 270
    assert sorted(data.state.unique()) == ["CA", "NY", "TX"]
    result = module.run(
        example / "ctis_aggregate.csv", example / "config.json", tmp_path
    )

    assert not result.predictions.empty
    assert not result.predictions.duplicated(["state", "time_value"]).any()
    assert np.isfinite(result.predictions[["y_true", "y_pred"]]).all().all()
    assert np.isfinite(result.weights["weight"]).all()
    assert (result.weights["weight"] >= 0).all()
    assert result.predictions.groupby("state").size().nunique() == 1
    pd.testing.assert_frame_equal(
        result.predictions,
        pd.read_parquet(tmp_path / "predictions.parquet"),
    )
    config = json.loads((tmp_path / "configuration.json").read_text())
    assert set(config["group_weights"]) == {*module.AUXILIARIES, "AR"}
    assert np.isclose(np.mean(list(config["group_weights"].values())), 1)
    assert (tmp_path / "weights.parquet").is_file()
    assert (tmp_path / "alpha_picks.csv").is_file()
