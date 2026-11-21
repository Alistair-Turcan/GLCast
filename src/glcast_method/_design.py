"""Lagged designs, rolling calendars, and donor screening."""

import numpy as np
import pandas as pd


def build_design_for_state(
    target_wide, feature_wides, state, start_date, end_date, feature_lags, ar_lags, unit
):
    """Build complete target/predictor rows for one state and date interval."""
    max_lag = max(max(feature_lags), max(ar_lags))
    start_ext = start_date - pd.Timedelta(max_lag, unit=unit)
    idx = target_wide.loc[start_ext:end_date].index
    if state not in target_wide.columns or len(idx) == 0:
        return None

    y_series = target_wide.loc[idx, state]
    blocks, feature_names, group_names = [], [], []
    for name, wide in feature_wides.items():
        if state not in wide.columns:
            continue
        series = wide.loc[idx, state]
        for lag in feature_lags:
            blocks.append(series.shift(lag))
            feature_names.append(f"{name}_lag{lag}")
        group_names.append(name)
    for lag in ar_lags:
        blocks.append(y_series.shift(lag))
        feature_names.append(f"AR_lag{lag}")
    group_names.append("AR")

    design = pd.concat(blocks, axis=1)
    design.columns = feature_names
    # Targets participate in the mask because the runner processes historical data.
    aligned = pd.concat([y_series.rename("y"), design], axis=1)
    aligned = aligned.loc[start_date:end_date].dropna()
    if aligned.empty:
        return None
    return (
        aligned.drop(columns="y").to_numpy(dtype=float),
        aligned["y"].to_numpy(dtype=float),
        pd.to_datetime(aligned.index),
        feature_names,
        group_names,
    )


def compute_walkforward_schedule(dates, train_size, test_size, retrain_every, unit="D"):
    """Return training end dates with a complete following test window.

    The first window starts at the first input date. Weekly data retain their
    existing weekday anchor.
    """
    dates = pd.DatetimeIndex(dates).sort_values()
    if dates.empty:
        return []
    start = dates.min() + pd.Timedelta(train_size - 1, unit=unit)
    end = dates.max() - pd.Timedelta(test_size, unit=unit)
    if start > end:
        return []
    return list(pd.date_range(start, end, freq=pd.Timedelta(retrain_every, unit=unit)))


def standardize_xy(X, y):
    X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    X_mean, X_std = X.mean(axis=0), X.std(axis=0, ddof=0)
    X_std[X_std == 0] = 1.0
    y_mean, y_std = y.mean(), y.std(ddof=0)
    if y_std == 0:
        y_std = 1.0
    return (X - X_mean) / X_std, (y - y_mean) / y_std, X_mean, X_std, y_mean, y_std


def estimate_R_hat_from_series(d0, dl, s):
    """Sum squared donor discrepancies over the largest s absolute differences."""
    common = d0.index.intersection(dl.index)
    if len(common) == 0:
        return np.inf
    delta = (dl.loc[common] - d0.loc[common]).to_numpy()
    count = min(s, len(delta))
    indices = np.argpartition(np.abs(delta), -count)[-count:]
    return float(np.sum(delta[indices] ** 2))


def rank_candidate_states(d_by_state, target_state, all_states, s):
    scores = [
        (
            state,
            estimate_R_hat_from_series(d_by_state[target_state], d_by_state[state], s),
        )
        for state in all_states
        if state != target_state and state in d_by_state
    ]
    scores.sort(key=lambda item: item[1])
    return [state for state, _ in scores]
