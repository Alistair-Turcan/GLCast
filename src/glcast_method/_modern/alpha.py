"""Completed-block alpha selection as used in the modern method."""

from typing import Dict, List, Optional, Sequence



import numpy as np



import pandas as pd



def block_index_map(origins: Sequence) -> Dict[pd.Timestamp, int]:
    """origin -> 1-based block index."""
    return {pd.to_datetime(o): i + 1 for i, o in enumerate(origins)}



def pro_pooled_alphas(self_preds: pd.DataFrame, origins: Sequence, per_state: bool = False):
    """Causal alpha picks per scored block.

    self_preds: grid self predictions (test split rows used).
    Returns DataFrame: block -> alpha (and state -> alpha if per_state).
    Picks at block bi use cumulative SSE over blocks < bi (all blocks incl. block 1).
    Block 1 gets NaN (no history).
    """
    test = self_preds[self_preds["split"] == "test"].copy()
    bmap = block_index_map(origins)
    test["block"] = test["origin"].map(bmap)
    test["se"] = (np.clip(test["y_pred"], 0, None) - test["y_true"]) ** 2

    alphas = sorted(test["alpha"].unique())
    blocks = sorted(bmap.values())
    picks = []
    if not per_state:
        sse = test.groupby(["block", "alpha"])["se"].agg(["sum", "count"]).reset_index()
        for bi in blocks:
            if bi == 1:
                picks.append({"block": bi, "alpha": np.nan})
                continue
            hist = sse[sse["block"] < bi]
            cs = hist.groupby("alpha").agg(cum_se=("sum", "sum"), cum_n=("count", "sum"))
            cs = cs.reindex(alphas).fillna(0.0)
            rmse = np.sqrt(cs["cum_se"] / cs["cum_n"].clip(lower=1))
            picks.append({"block": bi, "alpha": float(rmse.idxmin())})
    else:
        sse = test.groupby(["state", "block", "alpha"])["se"].agg(["sum", "count"]).reset_index()
        states = sorted(test["state"].unique())
        for bi in blocks:
            if bi == 1:
                picks.append({"block": bi, "alpha": np.nan})
                continue
            hist = sse[sse["block"] < bi]
            cs = hist.groupby(["state", "alpha"]).agg(cum_se=("sum", "sum"), cum_n=("count", "sum"))
            # mean over states of cumulative RMSE per alpha
            rmse = np.sqrt(cs["cum_se"] / cs["cum_n"].clip(lower=1)).reset_index()
            rmse = rmse[rmse["state"].isin(states)]
            per_alpha = rmse.groupby("alpha")[0].mean()
            per_alpha = per_alpha.reindex(alphas)
            if per_alpha.isna().all():
                # no valid history yet (e.g. target not yet observed): fall back to
                # the mid-grid default self alpha used downstream (0.7)
                picks.append({"block": bi, "alpha": np.nan})
                continue
            picks.append({"block": bi, "alpha": float(per_alpha.idxmin())})
    return pd.DataFrame(picks)


