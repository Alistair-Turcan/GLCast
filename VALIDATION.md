# Independent method-distribution validation

This records software fidelity and execution, **not prospective forecasting
validation** or completion of the separate full-paper audit.

## Executed checks

The current method suite has **37 passing tests**, including configuration and
input validation, combiner/regularization checks and a real CTIS aggregate-data
example. It passed in the source checkout and in two isolated environments:
one installed from the wheel, the other from the source distribution. The
isolated test directory contained tests, configurations and example assets,
but no `src` directory. Installed module paths were checked to reside in each
environment's `site-packages`.

The CTIS demonstration performed a uniform first pass, retrospective adaptive
weight calculation and final historical walk-forward prediction. It produced
147 prediction rows and 63 combination-weight rows from 270 observed aggregate
records, with 8 scheduled origins. Wheel, source-distribution and source-checkout
predictions agreed **bitwise** in the tested numerical environment. The example
subset was also rebuilt checksum-identically from the four original aggregate
CSVs; no imputation or synthetic observations were used.

The CLI input-validation command passed in both isolated environments. The
adaptive two-pass demonstration was executed through the public Python API;
the template CLI alone does not implicitly derive adaptive weights.

## Environment

Tested with CPython 3.11.13, NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1,
PyArrow 25.0.1, celer 0.7.4 and scikit-learn 1.9.1. The exact isolated
environment is pinned in `requirements-validated.txt`. Broader dependency
ranges in `pyproject.toml` are installation compatibility ranges, not evidence
that every combination or Python version was tested.

The celer-enabled classic backend was used. The optional no-celer FISTA path
is not established as bitwise-equivalent. Limit BLAS threads to one when
repeating numerical-fidelity checks.

## Boundaries and historical evidence

The numerical engines preserve their recovered source hashes; the public API
wrappers and example are tracked separately. Current distribution checks do not
turn the original manuscript into an independent prospective benchmark.
The companion reproduction repository records fresh fit receipts and compares
saved panels through a separate scoring/coverage path.

Earlier v2/v3 validation records are historical reference evidence only. Their
test counts and archived prediction comparisons are not counted as fresh
full-paper refits in this audit. Prediction/intercept atol 1e-8, coefficient
atol 1e-10, raw-weight atol 1e-12 and rtol 1e-12 remain the inherited strict
parity targets; the CTIS installation comparison was stronger (bitwise).

The numerical and temporal limitations in `METHOD.md` remain unchanged.
