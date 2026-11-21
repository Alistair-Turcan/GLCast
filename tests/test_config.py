import json
from dataclasses import replace

import pytest

from glcast_method import MethodConfig


def test_config_roundtrip(tmp_path):
    cfg = MethodConfig(
        backend="classic",
        alpha_y_scale=False,
        lag_decay=0.7,
        fixed_alpha=None,
        group_weights={"stream": 1.1, "AR": 0.9},
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert MethodConfig.from_json(path).to_dict() == cfg.to_dict()
    assert replace(cfg, lag_decay=0.0).alpha_y_scale is False


@pytest.mark.parametrize(
    "changes",
    [
        {"lag_decay": -1.0},
        {"lag_decay": float("nan")},
        {"backend": "modern", "lag_decay": 0.7},
        {"group_weights": {"AR": 0.0}},
        {"group_weights": {"AR": float("inf")}},
        {"ar_lags": (1, 2), "feature_lags": (0,)},
        {"fixed_alpha": 0.999},
        {"alphas": (0.1, 0.1)},
    ],
)
def test_invalid_configuration(changes):
    kwargs = {"backend": "classic", "alpha_y_scale": False, "lag_decay": 0.7}
    kwargs.update(changes)
    with pytest.raises(ValueError):
        MethodConfig(**kwargs)
