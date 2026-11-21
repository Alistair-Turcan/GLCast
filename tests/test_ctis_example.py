"""A real public aggregate-data smoke test; not a paper replication."""
import importlib.util
import json
from pathlib import Path

import pandas as pd


def test_ctis_aggregate_example(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "ctis_smoke"
    spec = importlib.util.spec_from_file_location("ctis_example", example / "run_example.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    receipt = module.run(example / "ctis_aggregate.csv", example / "config.json", tmp_path)
    assert receipt["input_cells"] == 270
    assert receipt["states"] == ["CA", "NY", "TX"]
    assert receipt["prediction_rows"] > 0
    assert receipt["prospective_validation"] is False
    assert json.loads((tmp_path / "receipt.json").read_text())["status"] == "complete"
    panel = pd.read_parquet(tmp_path / "predictions.parquet")
    assert panel.groupby("state").size().nunique() == 1
