"""Configuration for historical GLCast predictions."""

import json
from dataclasses import asdict, dataclass, field
from numbers import Real

import numpy as np

from .combiner import CombinerConfig

ALPHAS19 = (
    0.0002,
    0.0005,
    0.001,
    0.002,
    0.003,
    0.005,
    0.007,
    0.01,
    0.05,
    0.1,
    0.12,
    0.15,
    0.18,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
)


@dataclass
class MethodConfig:
    backend: str = "classic"
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
            try:
                setattr(self, name, tuple(getattr(self, name)))
            except TypeError as exc:
                raise ValueError(f"{name} must be a sequence") from exc
        if isinstance(self.combiner, dict):
            try:
                self.combiner = CombinerConfig(**self.combiner)
            except TypeError as exc:
                raise ValueError(f"Invalid combiner configuration: {exc}") from exc
        if not isinstance(self.combiner, CombinerConfig):
            raise ValueError(
                "combiner must be a CombinerConfig or configuration mapping"
            )
        if not isinstance(self.alpha_y_scale, bool):
            raise ValueError("alpha_y_scale must be a boolean")
        if self.backend not in ("modern", "classic") or self.unit not in ("D", "W"):
            raise ValueError("backend must be modern/classic and unit D/W")
        for name in (
            "train_window",
            "val_window",
            "test_window",
            "retrain_every",
            "screen",
            "n_jobs",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            self.train_window < 10
            or not 2 <= self.val_window <= self.train_window
            or self.test_window < 2
        ):
            raise ValueError("Require train>=10, 2<=val<=train and test>=2")
        for name in ("feature_lags", "ar_lags", "serve_steps"):
            values = getattr(self, name)
            lower = 1 if name != "feature_lags" else 0
            if (
                not values
                or len(set(values)) != len(values)
                or any(
                    not isinstance(v, int) or isinstance(v, bool) or v < lower
                    for v in values
                )
            ):
                raise ValueError(f"Invalid {name}")
        if max(self.serve_steps) > self.test_window:
            raise ValueError("serve_steps must fit inside test_window")
        if (
            not self.alphas
            or any(not isinstance(a, Real) or isinstance(a, bool) for a in self.alphas)
            or not np.isfinite(self.alphas).all()
            or min(self.alphas) <= 0
        ):
            raise ValueError("alphas must be finite positive numbers")
        if len(set(self.alphas)) != len(self.alphas):
            raise ValueError("alphas must be unique")
        if self.fixed_alpha is not None:
            if (
                not isinstance(self.fixed_alpha, Real)
                or isinstance(self.fixed_alpha, bool)
                or self.fixed_alpha not in self.alphas
            ):
                raise ValueError("fixed_alpha must be present in alphas")
        if (
            not isinstance(self.lag_decay, Real)
            or isinstance(self.lag_decay, bool)
            or not np.isfinite(self.lag_decay)
            or self.lag_decay < 0
        ):
            raise ValueError("lag_decay must be finite and nonnegative")
        if self.backend == "modern":
            if self.group_weights is not None or self.lag_decay:
                raise ValueError(
                    "Adaptive weights/lag-decay require the classic backend"
                )
            if len(self.ar_lags) != len(self.feature_lags):
                raise ValueError(
                    "The modern configuration requires equal-sized AR and auxiliary groups"
                )
        elif len(self.ar_lags) > len(self.feature_lags):
            raise ValueError(
                "Classic backend requires AR group no larger than auxiliary groups"
            )
        if self.group_weights is not None:
            if not isinstance(self.group_weights, dict) or not self.group_weights:
                raise ValueError("group_weights must be a nonempty finite mapping")
            if any(not isinstance(k, str) or not k for k in self.group_weights):
                raise ValueError("Group names must be nonempty strings")
            values = list(self.group_weights.values())
            if (
                any(not isinstance(v, Real) or isinstance(v, bool) for v in values)
                or not np.isfinite(values).all()
                or min(values) <= 0
            ):
                raise ValueError("group weights must be positive")

    @classmethod
    def from_json(cls, path):
        with open(path, encoding="utf-8") as f:
            values = json.load(f)
        if not isinstance(values, dict):
            raise ValueError("Configuration must be a JSON object")
        try:
            return cls(**values)
        except TypeError as exc:
            raise ValueError(f"Invalid method configuration: {exc}") from exc

    def to_dict(self):
        return asdict(self)
