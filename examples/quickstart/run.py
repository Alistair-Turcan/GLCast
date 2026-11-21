"""Fit GLCast to a small daily dataset and save historical predictions."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from glcast_method import (
    MethodConfig,
    adaptive_weights,
    fit_first_pass,
    load_long_csv,
    run_walkforward,
)

AUXILIARIES = [
    "raw_whh_cmnty_cli",
    "smoothed_wtested_14d",
    "smoothed_wtested_positive_14d",
]


def run(data, config, output):
    config = MethodConfig.from_json(config)
    target, features = load_long_csv(
        data,
        target="raw_wcli",
        auxiliaries=AUXILIARIES,
        config=config,
    )
    pilot = fit_first_pass(target, features, config)
    weights = adaptive_weights(
        pilot["fits"],
        [*features, "AR"],
        alpha_picks=pilot["alpha_picks"],
    )
    config = replace(config, group_weights=weights)
    result = run_walkforward(target, features, config)
    result.save(output)
    (Path(output) / "configuration.json").write_text(
        json.dumps(config.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=here / "ctis_aggregate.csv")
    parser.add_argument("--config", type=Path, default=here / "config.json")
    parser.add_argument("--output", type=Path, default=Path("example-output"))
    args = parser.parse_args()
    result = run(args.data, args.config, args.output)
    print(f"Saved {len(result.predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
