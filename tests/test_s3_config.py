from dataclasses import replace
import json
import pytest
from glcast_method import MethodConfig


def test_s3_roundtrip(tmp_path):
    cfg = MethodConfig(backend="classic", alpha_y_scale=False, lag_decay=.7,
                       fixed_alpha=None, group_weights={"stream":1.1,"AR":.9})
    path = tmp_path/"s3.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert MethodConfig.from_json(path).to_dict() == cfg.to_dict()
    assert replace(cfg,lag_decay=0.).alpha_y_scale is False


@pytest.mark.parametrize("changes",[
    {"lag_decay":-1.}, {"lag_decay":float("nan")},
    {"backend":"modern","lag_decay":.7},
    {"group_weights":{"AR":0.}}, {"group_weights":{"AR":float("inf")}},
    {"ar_lags":(1,2),"feature_lags":(0,)},
    {"fixed_alpha":.999}, {"alphas":(.1,.1)},
])
def test_other_guards_preserved(changes):
    kwargs = dict(backend="classic",alpha_y_scale=False,lag_decay=.7)
    kwargs.update(changes)
    with pytest.raises(ValueError):
        MethodConfig(**kwargs)
