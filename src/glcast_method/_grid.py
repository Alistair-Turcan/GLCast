"""Fit state models and rank donors for one rolling origin."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._design import build_design_for_state, rank_candidate_states, standardize_xy
from ._solver import SIS_grpLasso_KKT


@dataclass
class GridSpec:
    unit: str
    feature_lags: tuple[int, ...]
    ar_lags: tuple[int, ...]
    train_window: int
    val_window: int
    test_window: int
    alphas: tuple[float, ...]
    donor_screen: int
    donor_limit: int
    alpha_y_scale: bool
    group_weights: dict[str, float] | None
    lag_decay: float


def _run_one_origin(args):
    origin, spec, target_wide, feature_wides, states = args
    origin = pd.to_datetime(origin)
    train_start = origin - pd.Timedelta(spec.train_window - 1, unit=spec.unit)
    val_start = origin - pd.Timedelta(spec.val_window - 1, unit=spec.unit)
    test_end = origin + pd.Timedelta(spec.test_window, unit=spec.unit)
    fit_rows, prediction_rows, rank_rows = [], [], []
    designs, discrepancies = {}, {}

    for state in states:
        built = build_design_for_state(
            target_wide,
            feature_wides,
            state,
            train_start,
            test_end,
            spec.feature_lags,
            spec.ar_lags,
            spec.unit,
        )
        if built is None:
            continue
        X, y, dates, feature_names, groups = built
        train = (dates >= train_start) & (dates <= origin)
        validation = (dates >= val_start) & (dates <= origin)
        test = (dates > origin) & (dates <= test_end)
        if train.sum() < 10 or validation.sum() < 2 or test.sum() < 2:
            continue
        designs[state] = (X, y, dates, feature_names, groups, train, validation, test)
        if spec.donor_limit:
            Xz, yz, *_ = standardize_xy(X[train], y[train])
            discrepancies[state] = pd.Series((Xz.T @ yz) / len(yz), index=feature_names)

    for state, (
        X,
        y,
        dates,
        feature_names,
        groups,
        train,
        validation,
        test,
    ) in designs.items():
        X_train, y_train = X[train], y[train]
        scale = (float(y_train.std(ddof=0)) or 1.0) if spec.alpha_y_scale else 1.0
        weights = (
            None
            if spec.group_weights is None
            else [spec.group_weights[g] for g in groups]
        )
        lag_multipliers = np.array(
            [
                1.0 if name.startswith("AR_") else int(name.rsplit("_lag", 1)[1]) + 1
                for name in feature_names
            ]
        )
        for alpha in spec.alphas:
            solver_alpha = float(alpha) / scale
            column_penalty = spec.lag_decay * solver_alpha * lag_multipliers
            try:
                _, coefficients, intercept, _ = SIS_grpLasso_KKT(
                    X_train,
                    y_train,
                    len(spec.feature_lags),
                    solver_alpha,
                    weights=weights,
                    sparse_l1=column_penalty,
                )
            except (
                ValueError,
                RuntimeError,
                ArithmeticError,
                np.linalg.LinAlgError,
            ) as exc:
                raise RuntimeError(
                    f"Fit failed for state {state!r}, origin {origin.date()}, "
                    f"alpha {alpha}: {exc}"
                ) from exc
            for name, coefficient in zip(feature_names, coefficients):
                fit_rows.append(
                    {
                        "origin": origin,
                        "state": state,
                        "alpha": float(alpha),
                        "intercept": float(intercept),
                        "feature": name,
                        "coef": float(coefficient),
                    }
                )
            for split, mask in (("val", validation), ("test", test)):
                predictions = X[mask] @ coefficients + intercept
                for date, truth, prediction in zip(dates[mask], y[mask], predictions):
                    prediction_rows.append(
                        {
                            "origin": origin,
                            "state": state,
                            "alpha": float(alpha),
                            "split": split,
                            "time_value": date,
                            "y_true": float(truth),
                            "y_pred": float(prediction),
                        }
                    )

    for state in discrepancies:
        ranking = rank_candidate_states(
            discrepancies, state, sorted(discrepancies), spec.donor_screen
        )
        for rank, donor in enumerate(ranking[: spec.donor_limit], start=1):
            rank_rows.append(
                {
                    "origin": origin,
                    "target_state": state,
                    "s_value": spec.donor_screen,
                    "rank": rank,
                    "donor_state": donor,
                }
            )

    return (
        pd.DataFrame(
            fit_rows,
            columns=["origin", "state", "alpha", "intercept", "feature", "coef"],
        ),
        pd.DataFrame(
            prediction_rows,
            columns=[
                "origin",
                "state",
                "alpha",
                "split",
                "time_value",
                "y_true",
                "y_pred",
            ],
        ),
        pd.DataFrame(
            rank_rows,
            columns=["origin", "target_state", "s_value", "rank", "donor_state"],
        ),
    )
