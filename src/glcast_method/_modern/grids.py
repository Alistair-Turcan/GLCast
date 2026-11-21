"""Recovered numerical method implementation."""

import os



import traceback



from dataclasses import dataclass, field



from typing import Dict, List, Optional, Sequence



import numpy as np



import pandas as pd



from .harness import (build_design_for_state, compute_walkforward_schedule,
                      SIS_grpLasso_KKT, standardize_xy, rank_candidate_states)



@dataclass
class GridSpec:
    name: str
    dataset: str            # 'covid' or 'ili'
    unit: str               # 'D' or 'W'
    feature_lags: Sequence[int]
    ar_lags: Sequence[int]
    train_window: int
    val_window: int
    test_window: int
    retrain_every: int
    alphas: Sequence[float]
    donor_dump: bool = False
    donor_s_values: Sequence[int] = (25, 50)
    donor_k: int = 20       # store top-k per s_value
    sched_start_minus_one: bool = True  # spec footing: first te = series_min + (train-1)
    alpha_y_scale: bool = False  # if True, fit with alpha/y_std(train) per state:
                                 # equivalent to centering but NOT scaling y (spec weekly footing)
    group_size: int = 0          # 0 -> len(feature_lags); 1 -> plain Lasso reference



def _run_one_origin(args):
    (te, spec, target_wide, feature_wides, states) = args
    te = pd.to_datetime(te)
    train_start = te - pd.Timedelta(spec.train_window - 1, unit=spec.unit)
    val_start = te - pd.Timedelta(spec.val_window - 1, unit=spec.unit)
    test_start = te + pd.Timedelta(1, unit=spec.unit)
    test_end = te + pd.Timedelta(spec.test_window, unit=spec.unit)

    fit_rows: List[dict] = []
    pred_rows: List[dict] = []
    rank_rows: List[dict] = []

    data_cache = {}
    d_by_state = {}
    for st in states:
        built = build_design_for_state(target_wide, feature_wides, st,
                                       train_start, test_end,
                                       spec.feature_lags, spec.ar_lags, spec.unit)
        if built is None:
            continue
        X_all, y_all, dates_all, feat_names, group_names = built
        dates_all = pd.to_datetime(dates_all)
        idx_train = (dates_all >= train_start) & (dates_all <= te)
        idx_val = (dates_all >= val_start) & (dates_all <= te)
        idx_test = (dates_all >= test_start) & (dates_all <= test_end)
        if idx_train.sum() < 10 or idx_val.sum() < 2 or idx_test.sum() < 2:
            continue
        data_cache[st] = (X_all, y_all, dates_all, feat_names, idx_train, idx_val, idx_test)
        if spec.donor_dump:
            X_tr, y_tr = X_all[idx_train], y_all[idx_train]
            Xz_d, yz_d, *_ = standardize_xy(X_tr, y_tr)
            d_by_state[st] = pd.Series((Xz_d.T @ yz_d) / max(len(yz_d), 1), index=feat_names)

    for st, (X_all, y_all, dates_all, feat_names, idx_train, idx_val, idx_test) in data_cache.items():
        X_tr, y_tr = X_all[idx_train], y_all[idx_train]
        y_std_raw = float(np.std(y_tr, ddof=0)) if spec.alpha_y_scale else 1.0
        if y_std_raw == 0.0:
            y_std_raw = 1.0
        for alpha in spec.alphas:
            try:
                active_g, coef, intercept, _ = SIS_grpLasso_KKT(
                    X_expanded=X_tr, y=y_tr,
                    group_size=spec.group_size or len(spec.feature_lags),
                    alpha=float(alpha) / y_std_raw)
            except Exception:
                traceback.print_exc()
                continue
            for fn, c in zip(feat_names, coef):
                fit_rows.append({"origin": te, "state": st, "alpha": float(alpha),
                                 "intercept": float(intercept), "feature": fn, "coef": float(c)})
            coef_s = pd.Series(coef, index=feat_names)
            for split, idx in (("val", idx_val), ("test", idx_test)):
                Xs, ys, ds = X_all[idx], y_all[idx], dates_all[idx]
                yhat = Xs @ coef_s.to_numpy() + intercept
                for i, dt in enumerate(ds):
                    pred_rows.append({"origin": te, "state": st, "alpha": float(alpha),
                                      "split": split, "time_value": pd.to_datetime(dt),
                                      "y_true": float(ys[i]), "y_pred": float(yhat[i])})

    if spec.donor_dump and len(d_by_state) >= 2:
        for st in d_by_state:
            for s_val in spec.donor_s_values:
                ranking = rank_candidate_states(d_by_state, st, sorted(d_by_state.keys()), s=s_val)
                for r, donor in enumerate(ranking[: spec.donor_k], start=1):
                    rank_rows.append({"origin": te, "target_state": st, "s_value": int(s_val),
                                      "rank": int(r), "donor_state": donor})

    return (pd.DataFrame(fit_rows), pd.DataFrame(pred_rows), pd.DataFrame(rank_rows))



def run_grid(spec: GridSpec, target_wide, feature_wides, states, out_dir, n_jobs=1,
             origin_subset=None):
    """Run the grid. Returns dict of combined DataFrames; writes parquet parts."""
    os.makedirs(out_dir, exist_ok=True)
    origins = compute_walkforward_schedule(target_wide.index, spec.train_window,
                                           spec.test_window, spec.retrain_every, spec.unit,
                                           start_minus_one=spec.sched_start_minus_one)
    if origin_subset is not None:
        origins = [o for o in origins if o in set(origin_subset)]
    print(f"[{spec.name}] {len(origins)} origins, {len(states)} states, "
          f"{len(spec.alphas)} alphas", flush=True)

    tasks = [(te, spec, target_wide, feature_wides, states) for te in origins]
    parts_dir = os.path.join(out_dir, "parts")
    os.makedirs(parts_dir, exist_ok=True)

    if n_jobs > 1:
        import multiprocessing as mp
        with mp.Pool(n_jobs) as pool:
            results = pool.map(_run_one_origin, tasks, chunksize=1)
    else:
        results = [_run_one_origin(t) for t in tasks]

    fits = pd.concat([r[0] for r in results if not r[0].empty], ignore_index=True)
    preds = pd.concat([r[1] for r in results if not r[1].empty], ignore_index=True)
    out = {"fits": fits, "self_preds": preds}
    fits.to_parquet(os.path.join(out_dir, "fits.parquet"), index=False)
    preds.to_parquet(os.path.join(out_dir, "self_preds.parquet"), index=False)
    if spec.donor_dump:
        ranks = pd.concat([r[2] for r in results if not r[2].empty], ignore_index=True)
        ranks.to_parquet(os.path.join(out_dir, "donor_ranks.parquet"), index=False)
        out["donor_ranks"] = ranks
    print(f"[{spec.name}] done: fits={len(fits)}, preds={len(preds)}", flush=True)
    return out


