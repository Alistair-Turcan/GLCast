"""GLCast method-only package."""
from .combiner import CombinerConfig, Combination, combine_candidates, combine_origin
from .config import MethodConfig
from .regularization import adaptive_weights, choose_lags, column_penalties, fit_classic
from .pipeline import RunResult, load_long_csv, fit_first_pass, run_walkforward

__version__ = "0.1.2"
__all__ = [
    "CombinerConfig", "Combination", "combine_candidates", "combine_origin",
    "MethodConfig", "adaptive_weights", "choose_lags", "column_penalties",
    "fit_classic", "RunResult", "load_long_csv", "fit_first_pass", "run_walkforward",
]
