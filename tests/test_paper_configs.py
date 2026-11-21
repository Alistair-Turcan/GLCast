"""Regression guard against inheriting an earlier example's smaller grid."""
from pathlib import Path
from glcast_method import MethodConfig


def test_paper_configs_share_full_penalty_grid():
    expected = (0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007,
                0.01, 0.05, 0.1, 0.12, 0.15, 0.18, 0.2, 0.3, 0.4,
                0.5, 0.6, 0.7)
    root = Path(__file__).resolve().parents[1]/"configs"
    for name in ["classic_ili_weekly.json", "classic_covid_daily.json", "modern_weekly.json"]:
        cfg = MethodConfig.from_json(root/name)
        assert cfg.alphas == expected, name
        assert cfg.backend == "classic" and cfg.alpha_y_scale is False
        assert cfg.lag_decay == 0.7 and cfg.fixed_alpha is None
        assert cfg.combiner.library == 20 and cfg.combiner.k == 5
