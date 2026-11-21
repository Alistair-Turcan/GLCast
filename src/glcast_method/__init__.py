"""GLCast models and geographic prediction sharing."""

from .combiner import Combination, CombinerConfig, combine_candidates, combine_origin
from .config import MethodConfig
from .pipeline import RunResult, fit_first_pass, load_long_csv, run_walkforward
from .regularization import adaptive_weights, choose_lags, column_penalties, fit_classic

__version__ = "0.2.0"
__all__ = [
    "Combination",
    "CombinerConfig",
    "MethodConfig",
    "RunResult",
    "adaptive_weights",
    "choose_lags",
    "column_penalties",
    "combine_candidates",
    "combine_origin",
    "fit_classic",
    "fit_first_pass",
    "load_long_csv",
    "run_walkforward",
]
