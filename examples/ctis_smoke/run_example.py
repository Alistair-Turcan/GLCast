"""Run a small real aggregate-data demonstration, not a paper benchmark."""
import argparse
from dataclasses import replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np

from glcast_method import (
    MethodConfig, adaptive_weights, fit_first_pass, load_long_csv, run_walkforward,
)

AUXILIARIES = [
    "raw_whh_cmnty_cli", "smoothed_wtested_14d",
    "smoothed_wtested_positive_14d",
]


def run(data, config, output):
    started = time.perf_counter()
    data, config, output = Path(data), Path(config), Path(output)
    cfg = MethodConfig.from_json(config)
    target, features = load_long_csv(
        data, target="raw_wcli", auxiliaries=AUXILIARIES, config=cfg,
    )
    first = fit_first_pass(target, features, cfg)
    weights = adaptive_weights(
        first["fits"], [*features, "AR"], alpha_picks=first["alpha_picks"],
    )
    adaptive = replace(cfg, group_weights=weights)
    result = run_walkforward(target, features, adaptive)
    panel = result.predictions
    assert not panel.empty and not panel.duplicated(["state", "time_value"]).any()
    assert np.isfinite(panel[["y_true", "y_pred"]]).all().all()
    assert np.isfinite(result.weights["weight"]).all()
    assert (result.weights["weight"] >= 0).all()
    assert np.isclose(np.mean(list(weights.values())), 1)
    result.save(output)
    (output / "configuration.json").write_text(
        json.dumps(adaptive.to_dict(), indent=2) + "\n", encoding="utf-8",
    )
    first["alpha_picks"].to_csv(output / "first_pass_alpha_picks.csv", index=False)
    versions = {}
    for package in ["glcast-method", "numpy", "pandas", "scipy", "pyarrow", "celer"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    receipt = {
        "status": "complete", "purpose": "non-paper CTIS aggregate demonstration",
        "retrospective": True, "prospective_validation": False,
        "input_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "states": list(target.columns), "input_dates": len(target),
        "input_cells": int(target.size),
        "prediction_rows": len(panel), "weight_rows": len(result.weights),
        "first_pass_fit_rows": len(first["fits"]),
        "scheduled_origins": len(result.alpha_picks),
        "selected_group_weights": weights,
        "first_prediction": str(panel.time_value.min().date()),
        "last_prediction": str(panel.time_value.max().date()),
        "elapsed_seconds": time.perf_counter() - started,
        "versions": versions,
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8",
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    parser.add_argument("--data", type=Path, default=here / "ctis_aggregate.csv")
    parser.add_argument("--config", type=Path, default=here / "config.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.data, args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
