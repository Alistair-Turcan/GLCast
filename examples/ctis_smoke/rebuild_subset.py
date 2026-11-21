"""Rebuild the example from the four checksum-identified source CSVs."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def rebuild(source_directory, output):
    here = Path(__file__).resolve().parent
    provenance = json.loads((here / "provenance.json").read_text())
    selected = []
    for source in provenance["sources"]:
        path = Path(source_directory) / source["filename"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {path.name}")
        frame = pd.read_csv(path, parse_dates=["time_value"])
        frame = frame.loc[
            frame.geo_value.isin(["ca", "ny", "tx"])
            & frame.geo_type.eq("state")
            & frame.time_value.between("2021-01-01", "2021-03-31")
        ].copy()
        if frame.duplicated(["geo_value", "time_value"]).any():
            raise ValueError("Duplicate source state/date records")
        frame["geo_value"] = frame.geo_value.str.upper()
        selected.append(
            frame[["geo_value", "time_value", "value"]]
            .rename(columns={"value": source["signal"]})
            .set_index(["geo_value", "time_value"])
        )
    result = pd.concat(selected, axis=1).reset_index().rename(
        columns={"geo_value": "state", "time_value": "date"},
    ).sort_values(["date", "state"]).reset_index(drop=True)
    if result.shape != (270, 6) or result.isna().any().any():
        raise ValueError("Unexpected source coverage; do not impute")
    result.to_csv(output, index=False, date_format="%Y-%m-%d")
    if hashlib.sha256(Path(output).read_bytes()).hexdigest() != provenance["subset_sha256"]:
        raise ValueError("Rebuilt subset checksum mismatch")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    frame = rebuild(args.source_directory, args.output)
    print(f"Verified {len(frame)} actual aggregate rows; no imputation.")
