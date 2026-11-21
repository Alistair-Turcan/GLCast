"""Recovered numerical method implementation."""

from typing import Dict, List, Optional, Sequence



import numpy as np



import pandas as pd



from .harness import build_design_for_state, q_aggregate_weights



def _fit_lookup(fits_o):
    """fits_o: rows for one origin (state, alpha, intercept, feature, coef).
    Return dict (state, alpha) -> (coef_series, intercept)."""
    lut = {}
    for (st, a), g in fits_o.groupby(["state", "alpha"]):
        lut[(st, float(a))] = (pd.Series(g["coef"].to_numpy(), index=g["feature"].to_numpy()),
                               float(g["intercept"].iloc[0]))
    return lut



def _geo_origin_task(args):
    (O, target_wide, feature_wides, states, feature_lags, ar_lags, unit,
     train_window, val_window, test_window, fits_o, ranks_o, k, alphas,
     self_alpha) = args
    O = pd.to_datetime(O)
    train_start = O - pd.Timedelta(train_window - 1, unit=unit)
    test_end = O + pd.Timedelta(test_window, unit=unit)
    val_start = O - pd.Timedelta(val_window - 1, unit=unit)
    lut = _fit_lookup(fits_o)
    n_alpha = len(alphas)
    out = {}
    for s in states:
        built = build_design_for_state(target_wide, feature_wides, s, train_start,
                                       test_end, feature_lags, ar_lags, unit)
        if built is None:
            continue
        X_all, y_all, dates_all, feat_names, _ = built
        dates_all = pd.to_datetime(dates_all)
        idx_val = (dates_all >= val_start) & (dates_all <= O)
        idx_test = (dates_all > O) & (dates_all <= test_end)
        if idx_val.sum() < 2 or idx_test.sum() < 1:
            continue
        Xv, yv = X_all[idx_val], y_all[idx_val]
        Xt, yt = X_all[idx_test], y_all[idx_test]
        dv, dt = dates_all[idx_val], dates_all[idx_test]

        # self column at self_alpha
        sa = self_alpha.get(s, np.nan) if isinstance(self_alpha, dict) else self_alpha
        if not np.isfinite(sa) or (s, float(sa)) not in lut:
            continue
        scoef, sint = lut[(s, float(sa))]
        w = scoef.reindex(feat_names).fillna(0.0).to_numpy()
        self_val = Xv @ w + sint
        self_test = Xt @ w + sint

        # donors
        donors = []
        if ranks_o is not None and len(ranks_o):
            donors = (ranks_o[ranks_o["target_state"] == s].sort_values("rank")
                      ["donor_state"].tolist()[:k])
        donor_val = np.full((len(donors), n_alpha, len(yv)), np.nan)
        donor_test = np.full((len(donors), n_alpha, len(yt)), np.nan)
        for j, d in enumerate(donors):
            for ai, a in enumerate(alphas):
                if (d, float(a)) not in lut:
                    continue
                dcoef, dint = lut[(d, float(a))]
                wd = dcoef.reindex(feat_names).fillna(0.0).to_numpy()
                donor_val[j, ai] = Xv @ wd + dint
                donor_test[j, ai] = Xt @ wd + dint
        out[s] = {"y_val": yv, "y_test": yt,
                  "dates_val": dv.to_numpy(), "dates_test": dt.to_numpy(),
                  "self_val": self_val, "self_test": self_test,
                  "donors": donors, "donor_val": donor_val, "donor_test": donor_test}
    return out


