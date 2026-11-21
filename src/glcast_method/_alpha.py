"""Regularization selection from preceding historical prediction windows."""

import numpy as np
import pandas as pd


def pro_pooled_alphas(self_preds, origins, per_state=False):
    """Select an alpha using earlier origins with observed prediction errors.

    Test windows are used in full, including dates beyond a subsequent origin.
    Predictions are clipped at zero for this selection criterion only.
    per_state=True averages state RMSEs; either mode returns one alpha per block.
    """
    block_map = {pd.to_datetime(origin): i + 1 for i, origin in enumerate(origins)}
    test = self_preds[self_preds["split"] == "test"].copy()
    test["block"] = test["origin"].map(block_map)
    finite = np.isfinite(test["y_true"]) & np.isfinite(test["y_pred"])
    test = test.loc[finite].copy()
    test["se"] = (np.clip(test["y_pred"], 0, None) - test["y_true"]) ** 2
    picks = []
    for block in block_map.values():
        history = test[test["block"] < block]
        if history.empty:
            picks.append({"block": block, "alpha": np.nan})
            continue
        if per_state:
            by_state = history.groupby(["state", "alpha"])["se"].mean().pow(0.5)
            errors = by_state.groupby("alpha").mean()
        else:
            errors = history.groupby("alpha")["se"].mean().pow(0.5)
        # An alpha without observations has no score and cannot win selection.
        errors = errors.dropna().sort_index()
        alpha = float(errors.idxmin()) if not errors.empty else np.nan
        picks.append({"block": block, "alpha": alpha})
    return pd.DataFrame(picks, columns=["block", "alpha"])
