# GLCast

This folder contains the revised GLCast method, not its evaluation results:

- local Group Lasso fitting with screening and iterative KKT checks;
- classic-analysis adaptive group weights and within-group lag penalties;
- a strict-improvement auxiliary-lag gate;
- training-discrepancy donor screening and geographic coefficient transport;
- validation-ranked donor selection and the unified shrunk-stacking combiner.

**All three S3 templates use the classic numerical backend, direct alpha,
positive column multiplier 0.7, and automatic pooled alpha choices.** The
retained modern backend is still available as a library option, but is not the
backend used for any of this bundle's nine S3 main analyses.

This independent repository contains the method, not the paper's evaluation
pipeline. The companion `glcast_paper_repro` repository contains the manuscript
audit, data-preparation and scoring pipeline, and executed result notebooks.
Neither repository requires hidden session storage.

## Install and test

Python 3.10 or later is required. From this folder:

```bash
uv venv .venv
uv pip install --python .venv/bin/python ".[fast,test]"
.venv/bin/python -m pytest
.venv/bin/glcast-method --help
```

The `fast` extra installs celer for the recovered ordinary Group Lasso path.
Without it, the recovered FISTA fallback is used. Sparse-group fits use the
recovered FISTA implementation even with celer installed. To keep numerical
reproduction consistent, retain the same solver/dependency versions; optional
backends need not produce identical floating-point coefficients.

Before starting multiprocess runs, limit BLAS threads in the shell:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

The default is one process. Change `n_jobs` in a copied configuration only when
appropriate for your input size and available memory.

## Input

Supply your own long CSV with one row per date and state:

| Column | Meaning |
|---|---|
| date | Midnight date label, same daily or 7-day cadence across the table |
| state | Nonempty string jurisdiction identifier |
| target | Numeric continuous target |
| auxiliary columns | Named numeric auxiliary streams, in the order specified |

Column names are configurable. Missing measurements may be blank/NaN, but no
imputation is performed: the recovered design retains complete target/predictor
rows. Duplicate state/date keys, infinite values, irregular global date grids
and incompatible configurations are rejected. A weekly series can be Saturday-
anchored; it is **not** converted to Sunday weeks. Explicitly insert missing
calendar rows before running if the global date grid has gaps.

Use substantive stream names (not the reserved `AR`/`AR_` prefix). National streams
can be repeated across states, as in the original input preparation; this package
does not download, aggregate or transform external surveillance data.

## Weekly S3 template: historical walk-forward predictions

These commands refer to a user-supplied `your_data.csv`, not a bundled example:

```bash
.venv/bin/glcast-method validate \
  --input your_data.csv --date epiweek_end --state state_abbr \
  --target target_nssp_ed_pct --aux noaa_temp_f noaa_prcp_in \
  --config configs/modern_weekly.json

.venv/bin/glcast-method run \
  --input your_data.csv --date epiweek_end --state state_abbr \
  --target target_nssp_ed_pct --aux noaa_temp_f noaa_prcp_in \
  --config configs/modern_weekly.json --output your_predictions
```

Specify **all** auxiliary columns intended for your model. The two shown in the
command are usage placeholders, not the complete modern-analysis feature set.
The template has `group_weights=null`, so these CLI commands use **uniform**
weights, not the paper's adaptive S3 weights. Complete the explicit two-pass
preparation below and serialize its resulting configuration for adaptive S3.

When you run it, the method writes `predictions.parquet`, `weights.parquet`,
`alpha_picks.csv` and `configuration.json`. These contain method outputs, not
scores or comparisons. The first origin is burn-in; predictions are un-clipped.
Rows with insufficient complete-case history/candidates may be absent.

**Scope:** this runner is historical replay, not a turnkey live forecast service.
It preserves the recovered training/validation overlap, complete-case target
mask and earlier-origin alpha selection. In particular, earlier-origin whole
test windows can extend beyond the next origin. It does not implement a
release-vintage availability cutoff. Read `METHOD.md` before treating it as a
prospective forecasting pipeline.

## Use the combiner without future labels

```python
from glcast_method import combine_candidates

# Supply real arrays from your own fitted models:
# y_val: (n_validation,)
# validation_predictions: (n_validation, 1 + n_donors)
# future_predictions: (n_future, 1 + n_donors)
# Column zero is self; donors are already in training-discrepancy order.
result = combine_candidates(
    y_val, validation_predictions, future_predictions,
    names=["self", *donor_names],
)
if result is not None:
    predictions = result.predictions
    weights = dict(zip(result.names, result.weights))
```

The combiner accepts no future labels. It returns `None` if self predictions
are nonfinite. Donors with nonfinite validation or prediction values are
excluded. It retains at most five validation-ranked donors from the first
twenty supplied donors, always keeping self. Weights are nonnegative but
generally **do not sum to one**.

## S3 configurations and explicit adaptive preparation

All files in `configs/` are templates for the recovered classic numerical
backend and unified combiner, with the appropriate daily/weekly calendars.
`group_weights: null` means **uniform weights**, not an implicit adaptive fit.
For the full adaptive method, first derive weights, then explicitly supply them:

```python
from dataclasses import replace
from glcast_method import (
    MethodConfig, load_long_csv, fit_first_pass, adaptive_weights, run_walkforward,
)

cfg = MethodConfig.from_json("configs/classic_covid_daily.json")
target, features = load_long_csv(
    "your_daily_data.csv", target="target",
    auxiliaries=your_auxiliary_column_names, config=cfg,
)
pilot = fit_first_pass(target, features, cfg)  # uniform weights, no column penalty
order = [*features, "AR"]
weights = adaptive_weights(pilot["fits"], order,
                           alpha_picks=pilot["alpha_picks"])
cfg = replace(cfg, group_weights=weights)
predictions = run_walkforward(target, features, cfg)
```

Use the first pass's pooled alpha picks for **all S3 targets**, including COVID.
`fit_first_pass` explicitly turns off column penalties and uses uniform group
weights while retaining direct alpha and the caller's lag design. The adaptive
helper computes inverse selection frequency plus 0.05, normalized to mean one,
including AR. You can serialize `cfg.to_dict()` to JSON and pass it to the CLI.
These are **analysis-wide retrospective** weights, not causal online updates.
The final fitted named weights in the parent bundle are not implicitly supplied
by a template. Retain the first pass's weights for monthly fallback fitting but
derive the fallback's own pooled alpha history.

`choose_lags(deployed_lags, expanded_lags, deployed_error, expanded_error)` applies
the strict-improvement lag gate to caller-supplied matched errors; ties keep the
deployed set. It does not launch a tuning sweep. The headline classic gate was
retrospective. Run the gate explicitly and pass the chosen lags to the config.
The templates specify the selected lag shapes, not a new evaluation of the gate.

The portable runner does **not** recreate the paper's multi-cadence fallback
stitching, outbreak scoring or evaluation masks. The fitting components,
adaptive-weight preparation, lag gate and combiner are supplied as reusable
method code. This boundary is deliberate: this is not the benchmark repository.

## Run the included real-data example

`examples/ctis_smoke/` contains 270 public **aggregate** CTIS records:
CA, NY and TX, daily from 2021-01-01 through 2021-03-31. The target is aggregate
weighted COVID-like illness percentage; the three auxiliary streams are
aggregate community illness/testing measurements. No respondent-level records,
Google search streams or retail data are included.

```bash
.venv/bin/python examples/ctis_smoke/run_example.py --output ctis_outputs
```

This executes the public first-pass/adaptive-weight/final-prediction APIs, then
writes predictions, raw combination weights, alpha histories, the resulting
configuration and an execution receipt. The example deliberately uses a small
three-alpha, 28-day-training configuration. **It is not a paper target,
performance benchmark or prospective validation.** Missing data were not
imputed; the selected source cells are all observed. See its README and
provenance JSON for CC BY 4.0 attribution, exact transformations and checksums.
The test suite executes this actual example rather than generating mock data.

## Contents

- `src/glcast_method/`: public APIs plus separately preserved modern/classic engines
- `configs/`: portable configuration templates
- `tests/`: numerical, validation and interface tests, plus the public aggregate example
- `examples/ctis_smoke/`: real CTIS aggregate subset, provenance and runnable demo
- `METHOD.md`: mathematical rules, implementation caveats and timing conventions
- `PROVENANCE.md`, `SOURCE_HASHES.json`, `LICENSE`: source attribution and traceability
