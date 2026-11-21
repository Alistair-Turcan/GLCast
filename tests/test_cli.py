import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def test_cli_input_and_output(tmp_path):
    root = Path(__file__).resolve().parents[1]
    example = root / "examples" / "quickstart"
    config = json.loads((example / "config.json").read_text())
    config["fixed_alpha"] = 0.01
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    arguments = [
        "--input",
        str(example / "ctis_aggregate.csv"),
        "--target",
        "raw_wcli",
        "--aux",
        "raw_whh_cmnty_cli",
        "smoothed_wtested_14d",
        "smoothed_wtested_positive_14d",
        "--config",
        str(config_path),
    ]
    validated = subprocess.run(
        [sys.executable, "-m", "glcast_method", "validate", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Valid input" in validated.stdout
    output = tmp_path / "predictions"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "glcast_method",
            "run",
            *arguments,
            "--output",
            str(output),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    panel = pd.read_parquet(output / "predictions.parquet")
    assert not panel.empty
    assert np.isfinite(panel[["y_true", "y_pred"]]).all().all()
    assert (output / "weights.parquet").is_file()
    assert (output / "alpha_picks.csv").is_file()
    saved_config = json.loads((output / "configuration.json").read_text())
    assert saved_config["fixed_alpha"] == 0.01
