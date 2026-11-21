"""Load input data and run historical rolling predictions."""

import multiprocessing as mp
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._alpha import pro_pooled_alphas
from ._design import compute_walkforward_schedule
from ._geo import _geo_origin_task
from ._grid import GridSpec, _run_one_origin
from .combiner import _combine_origin
from .config import MethodConfig

PRED_COLUMNS = ["state", "time_value", "y_true", "y_pred", "origin", "block"]
WEIGHT_COLUMNS = [
    "block",
    "origin",
    "state",
    "candidate",
    "weight",
    "gamma",
    "w_stack",
    "w_qagg",
]


@dataclass
class RunResult:
    predictions: pd.DataFrame
    weights: pd.DataFrame
    alpha_picks: pd.DataFrame

    def save(self, directory):
        """Write predictions.parquet, weights.parquet, and alpha_picks.csv."""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        self.predictions.to_parquet(out / "predictions.parquet", index=False)
        self.weights.to_parquet(out / "weights.parquet", index=False)
        self.alpha_picks.to_csv(out / "alpha_picks.csv", index=False)


def validate_wides(target, features, cfg):
    if not isinstance(target, pd.DataFrame) or not isinstance(
        target.index, pd.DatetimeIndex
    ):
        raise ValueError("Target must be a DataFrame with a DatetimeIndex")
    if target.empty or target.index.has_duplicates or target.columns.has_duplicates:
        raise ValueError("Target must be nonempty with unique dates and states")
    if not target.index.is_monotonic_increasing or target.index.tz is not None:
        raise ValueError("Dates must be sorted, timezone-naive timestamps")
    if not (target.index == target.index.normalize()).all():
        raise ValueError("Dates must be midnight daily/weekly labels")
    expected = pd.date_range(
        target.index.min(), target.index.max(), freq="7D" if cfg.unit == "W" else "D"
    )
    if not target.index.equals(expected):
        raise ValueError(
            "Dates must form a regular D/7D grid; insert explicit missing rows before running"
        )
    if any(not isinstance(s, str) or not s for s in target.columns):
        raise ValueError("State identifiers must be nonempty strings")
    if not features:
        raise ValueError("At least one auxiliary stream is required")
    for name, frame in features.items():
        if (
            not isinstance(name, str)
            or not name
            or name.startswith("AR_")
            or name == "AR"
        ):
            raise ValueError(
                "Auxiliary names must be nonempty and not use the reserved AR prefix"
            )
        if not frame.index.equals(target.index) or frame.columns.has_duplicates:
            raise ValueError(f"Auxiliary {name} must align to the target date grid")
        if not set(frame.columns).issubset(target.columns):
            raise ValueError(f"Auxiliary {name} has unknown states")
    for frame in [target, *features.values()]:
        try:
            arr = frame.to_numpy(dtype=float)
        except (ValueError, TypeError) as e:
            raise ValueError("Targets and auxiliary streams must be numeric") from e
        if np.isinf(arr).any():
            raise ValueError("Infinite values are invalid; missing values must be NaN")
    if cfg.group_weights is not None:
        missing = set(features).union({"AR"}) - set(cfg.group_weights)
        if missing:
            raise ValueError(f"Missing named group weights: {sorted(missing)}")


def load_long_csv(
    path, *, target, auxiliaries, config=None, date="date", state="state"
):
    """Load a long table without imputation; return aligned wide matrices."""
    cfg = config or MethodConfig()
    auxiliaries = list(auxiliaries)
    selected = [date, state, target] + auxiliaries
    if not auxiliaries or len(set(selected)) != len(selected):
        raise ValueError("Provide distinct date/state/target/auxiliary column names")
    frame = pd.read_csv(path, dtype={state: str})
    missing = set(selected) - set(frame)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")
    if frame[state].isna().any() or (frame[state].str.len() == 0).any():
        raise ValueError("Missing state identifiers")
    frame[date] = pd.to_datetime(frame[date], errors="raise")
    if frame[date].isna().any() or frame.duplicated([date, state]).any():
        raise ValueError("Missing dates or duplicate date/state records")
    for col in [target] + auxiliaries:
        frame[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    states = sorted(frame[state].unique())
    tw = (
        frame.pivot(index=date, columns=state, values=target)
        .sort_index()
        .reindex(columns=states)
    )
    fw = {
        name: frame.pivot(index=date, columns=state, values=name)
        .sort_index()
        .reindex(index=tw.index, columns=states)
        for name in auxiliaries
    }
    validate_wides(tw, fw, cfg)
    return tw, fw


def _grid_spec(cfg):
    return GridSpec(
        unit=cfg.unit,
        feature_lags=cfg.feature_lags,
        ar_lags=cfg.ar_lags,
        train_window=cfg.train_window,
        val_window=cfg.val_window,
        test_window=cfg.test_window,
        alphas=(cfg.fixed_alpha,) if cfg.fixed_alpha is not None else cfg.alphas,
        donor_screen=cfg.screen,
        donor_limit=cfg.combiner.library,
        alpha_y_scale=cfg.alpha_y_scale,
        group_weights=cfg.group_weights,
        lag_decay=cfg.lag_decay,
    )


def fit_first_pass(target_wide, feature_wides, config=None, *, origins=None):
    """Fit a uniform-weight pilot without column penalties for adaptive weights.

    Returns ``fits``, ``self_preds``, ``donor_ranks`` and ``alpha_picks`` tables.
    """
    from dataclasses import replace

    cfg = replace(
        config or MethodConfig(backend="classic"), group_weights=None, lag_decay=0.0
    )
    validate_wides(target_wide, feature_wides, cfg)
    scheduled = list(
        compute_walkforward_schedule(
            target_wide.index,
            cfg.train_window,
            cfg.test_window,
            cfg.retrain_every,
            cfg.unit,
        )
    )
    if origins is not None:
        requested = set(pd.to_datetime(origins))
        if not requested.issubset(set(scheduled)):
            raise ValueError("Requested origins are outside the configured schedule")
        scheduled = [o for o in scheduled if o in requested]
    if not scheduled:
        raise ValueError("No usable first-pass origins")
    spec = _grid_spec(cfg)
    states = sorted(target_wide.columns)
    tasks = [(o, spec, target_wide, feature_wides, states) for o in scheduled]
    if cfg.n_jobs > 1:
        with mp.Pool(cfg.n_jobs) as pool:
            results = pool.map(_run_one_origin, tasks, chunksize=1)
    else:
        results = [_run_one_origin(t) for t in tasks]
    tables = {}
    for i, key in enumerate(("fits", "self_preds", "donor_ranks")):
        parts = [r[i] for r in results if not r[i].empty]
        tables[key] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if tables["fits"].empty:
        raise ValueError(
            "No first-pass fits: inspect complete-case history and lag coverage"
        )
    if cfg.fixed_alpha is None:
        picks = pro_pooled_alphas(tables["self_preds"], scheduled, per_state=False)
        picks["origin"] = picks["block"].map(dict(enumerate(scheduled, 1)))
    else:
        picks = pd.DataFrame(
            {
                "origin": scheduled,
                "alpha": cfg.fixed_alpha,
                "block": range(1, len(scheduled) + 1),
            }
        )
    tables["alpha_picks"] = picks
    return tables


def run_walkforward(
    target_wide, feature_wides, config=None, *, origins=None, progress=None
):
    """Run rolling predictions over observed historical target and auxiliary data.

    Complete-case row selection uses observed targets and lagged inputs.
    Local fits use the configured training window, and the combiner uses its
    trailing calibration window. Automatic alpha selection uses prediction
    errors from complete test windows at earlier origins.

    Use `combine_candidates` directly to combine predictions for unlabeled rows.
    """
    cfg = config or MethodConfig()
    validate_wides(target_wide, feature_wides, cfg)
    spec = _grid_spec(cfg)
    scheduled = list(
        compute_walkforward_schedule(
            target_wide.index,
            cfg.train_window,
            cfg.test_window,
            cfg.retrain_every,
            cfg.unit,
        )
    )
    if origins is not None:
        requested = set(pd.to_datetime(origins))
        if not requested.issubset(set(scheduled)):
            raise ValueError("Requested origins are outside the configured schedule")
        scheduled = [o for o in scheduled if o in requested]
    if not scheduled:
        raise ValueError(
            "No walk-forward origins: supply more history or a valid origin subset"
        )
    states = sorted(target_wide.columns)
    tasks = ((o, spec, target_wide, feature_wides, states) for o in scheduled)
    pool = mp.Pool(cfg.n_jobs) if cfg.n_jobs > 1 else None
    fitted = (
        pool.imap(_run_one_origin, tasks, chunksize=1)
        if pool
        else map(_run_one_origin, tasks)
    )
    rows, weights, alpha_rows, history = [], [], [], []
    valid_origins = 0
    try:
        for bi, (o, result) in enumerate(zip(scheduled, fitted), start=1):
            fits, self_preds, ranks = result
            if not self_preds.empty:
                history.append(self_preds)
            sa = cfg.fixed_alpha if cfg.fixed_alpha is not None else np.nan
            if cfg.fixed_alpha is None and bi > 1:
                if history:
                    all_preds = pd.concat(history, ignore_index=True)
                    picks = pro_pooled_alphas(
                        all_preds, scheduled[:bi], per_state=False
                    )
                    sa = float(picks.loc[picks["block"] == bi, "alpha"].iloc[0])
            alpha_rows.append({"block": bi, "origin": o, "alpha": sa})
            if fits.empty or not np.isfinite(sa):
                if progress:
                    progress(
                        f"origin {bi}/{len(scheduled)} {o.date()}: no usable fit/alpha (or burn-in)"
                    )
                continue
            rank_o = ranks if not ranks.empty else None
            candidates = _geo_origin_task(
                (
                    o,
                    target_wide,
                    feature_wides,
                    states,
                    cfg.feature_lags,
                    cfg.ar_lags,
                    cfg.unit,
                    cfg.train_window,
                    cfg.val_window,
                    cfg.test_window,
                    fits,
                    rank_o,
                    cfg.combiner.library,
                    sa,
                )
            )
            served = 0
            for st, block in candidates.items():
                combined = _combine_origin(block, cfg.combiner)
                if combined is None:
                    continue
                for i, t in enumerate(pd.to_datetime(block["dates_test"])):
                    step = int((t - o) / pd.Timedelta(1, unit=cfg.unit))
                    if step in cfg.serve_steps:
                        rows.append(
                            {
                                "state": st,
                                "time_value": t,
                                "y_true": float(block["y_test"][i]),
                                "y_pred": float(combined.predictions[i]),
                                "origin": o,
                                "block": bi,
                            }
                        )
                        served += 1
                for j, name in enumerate(combined.names):
                    weights.append(
                        {
                            "block": bi,
                            "origin": o,
                            "state": st,
                            "candidate": name,
                            "weight": float(combined.weights[j]),
                            "gamma": combined.gamma,
                            "w_stack": float(combined.stacking_weights[j]),
                            "w_qagg": float(combined.q_weights[j]),
                        }
                    )
            valid_origins += int(served > 0)
            if progress:
                progress(
                    f"origin {bi}/{len(scheduled)} {o.date()}: {served} served rows"
                )
    finally:
        if pool:
            pool.terminate()
            pool.join()
    panel = pd.DataFrame(rows, columns=PRED_COLUMNS)
    if not panel.empty:
        panel = (
            panel.sort_values(["state", "time_value", "origin"])
            .groupby(["state", "time_value"], as_index=False)
            .last()
        )[PRED_COLUMNS]
    else:
        warnings.warn(
            "No predictions: inspect missing values, history, alpha coverage and lag availability",
            RuntimeWarning,
            stacklevel=2,
        )
    expected_origins = len(scheduled) - int(cfg.fixed_alpha is None)
    if valid_origins < expected_origins:
        warnings.warn(
            "Some origins produced no served rows", RuntimeWarning, stacklevel=2
        )
    return RunResult(
        panel,
        pd.DataFrame(weights, columns=WEIGHT_COLUMNS),
        pd.DataFrame(alpha_rows, columns=["block", "origin", "alpha"]),
    )
