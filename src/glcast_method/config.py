"""Explicit analysis configuration; no automatic dataset-specific tuning."""
from dataclasses import dataclass, field, asdict
import json
import numpy as np
from .combiner import CombinerConfig

ALPHAS19 = (0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01, 0.05,
            0.1, 0.12, 0.15, 0.18, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


@dataclass
class MethodConfig:
    backend: str = "modern"
    unit: str = "W"
    train_window: int = 55
    val_window: int = 4
    test_window: int = 4
    retrain_every: int = 1
    feature_lags: tuple[int, ...] = (0,)
    ar_lags: tuple[int, ...] = (2,)
    alphas: tuple[float, ...] = ALPHAS19
    alpha_y_scale: bool = False
    fixed_alpha: float | None = None
    serve_steps: tuple[int, ...] = (2,)
    screen: int = 50
    n_jobs: int = 1
    group_weights: dict[str, float] | None = None
    lag_decay: float = 0.0
    combiner: CombinerConfig = field(default_factory=CombinerConfig)

    def __post_init__(self):
        for name in ("feature_lags", "ar_lags", "alphas", "serve_steps"):
            setattr(self, name, tuple(getattr(self, name)))
        if isinstance(self.combiner, dict):
            self.combiner = CombinerConfig(**self.combiner)
        if self.backend not in ("modern", "classic") or self.unit not in ("D", "W"):
            raise ValueError("backend must be modern/classic and unit D/W")
        for name in ("train_window", "val_window", "test_window", "retrain_every", "screen", "n_jobs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.train_window < 10 or not 2 <= self.val_window <= self.train_window or self.test_window < 2:
            raise ValueError("Require train>=10, 2<=val<=train and test>=2")
        for name in ("feature_lags", "ar_lags", "serve_steps"):
            values = getattr(self, name)
            lower = 1 if name != "feature_lags" else 0
            if not values or len(set(values)) != len(values) or any(
                not isinstance(v, int) or isinstance(v, bool) or v < lower for v in values
            ):
                raise ValueError(f"Invalid {name}")
        if max(self.serve_steps) > self.test_window:
            raise ValueError("serve_steps must fit inside test_window")
        if not self.alphas or not np.isfinite(self.alphas).all() or min(self.alphas) <= 0:
            raise ValueError("alphas must be finite positive numbers")
        if len(set(self.alphas)) != len(self.alphas):
            raise ValueError("alphas must be unique")
        if self.fixed_alpha is not None and self.fixed_alpha not in self.alphas:
            raise ValueError("fixed_alpha must be present in alphas")
        if not np.isfinite(self.lag_decay) or self.lag_decay < 0:
            raise ValueError("lag_decay must be finite and nonnegative")
        if self.backend == "modern":
            if self.group_weights is not None or self.lag_decay:
                raise ValueError("Adaptive weights/lag-decay require the classic backend")
            if len(self.ar_lags) != len(self.feature_lags):
                raise ValueError("Recovered modern backend requires equal-sized AR and auxiliary groups")
        elif len(self.ar_lags) > len(self.feature_lags):
            raise ValueError("Classic backend requires AR group no larger than auxiliary groups")
        # Both direct-standardized and target-SD alpha scales are supported
        # by the unchanged classic grid, including positive column penalties.
        if self.group_weights is not None:
            if not self.group_weights or not np.isfinite(list(self.group_weights.values())).all():
                raise ValueError("group_weights must be a nonempty finite mapping")
            if min(self.group_weights.values()) <= 0:
                raise ValueError("group weights must be positive")

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))

    def to_dict(self):
        return asdict(self)
