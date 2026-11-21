"""Transport fitted donor coefficients onto a recipient state's design."""

import numpy as np
import pandas as pd

from ._design import build_design_for_state


def _geo_origin_task(args):
    (
        origin,
        target_wide,
        feature_wides,
        states,
        feature_lags,
        ar_lags,
        unit,
        train_window,
        val_window,
        test_window,
        fits,
        ranks,
        donor_limit,
        alpha,
    ) = args
    origin = pd.to_datetime(origin)
    train_start = origin - pd.Timedelta(train_window - 1, unit=unit)
    val_start = origin - pd.Timedelta(val_window - 1, unit=unit)
    test_end = origin + pd.Timedelta(test_window, unit=unit)
    selected_fits = fits[fits["alpha"] == alpha]
    lookup = {
        state: (
            pd.Series(group["coef"].to_numpy(), index=group["feature"]),
            float(group["intercept"].iloc[0]),
        )
        for state, group in selected_fits.groupby("state")
    }
    candidates = {}

    for state in states:
        if state not in lookup:
            continue
        built = build_design_for_state(
            target_wide,
            feature_wides,
            state,
            train_start,
            test_end,
            feature_lags,
            ar_lags,
            unit,
        )
        if built is None:
            continue
        X, y, dates, names, _ = built
        validation = (dates >= val_start) & (dates <= origin)
        test = (dates > origin) & (dates <= test_end)
        if validation.sum() < 2 or test.sum() < 1:
            continue
        X_val, X_test = X[validation], X[test]
        coefficients, intercept = lookup[state]
        coefficients = coefficients.reindex(names).fillna(0.0).to_numpy()
        donors = []
        if ranks is not None and not ranks.empty:
            donors = (
                ranks[ranks["target_state"] == state]
                .sort_values("rank")["donor_state"]
                .tolist()[:donor_limit]
            )
        donor_val = np.full((len(donors), validation.sum()), np.nan)
        donor_test = np.full((len(donors), test.sum()), np.nan)
        for j, donor in enumerate(donors):
            if donor in lookup:
                donor_coefficients, donor_intercept = lookup[donor]
                donor_coefficients = (
                    donor_coefficients.reindex(names).fillna(0.0).to_numpy()
                )
                donor_val[j] = X_val @ donor_coefficients + donor_intercept
                donor_test[j] = X_test @ donor_coefficients + donor_intercept
        candidates[state] = {
            "y_val": y[validation],
            "y_test": y[test],
            "dates_val": dates[validation].to_numpy(),
            "dates_test": dates[test].to_numpy(),
            "self_val": X_val @ coefficients + intercept,
            "self_test": X_test @ coefficients + intercept,
            "donors": donors,
            "donor_val": donor_val,
            "donor_test": donor_test,
        }
    return candidates
