# Method versus paper reproduction

The `glcast_method` Python package exposes configuration, local fitting,
two-pass adaptive-weight preparation, historical walk-forward replay, lag-choice
helpers, and geographic combination. It does not download surveillance data,
obtain retail-data licenses, label epidemic upswings, or define the paper's score
masks. Those responsibilities belong to the separate paper reproduction repository.

## Exact numerical reference

The audit environment starts from Python 3.11, NumPy 2.4.6, pandas 3.0.6,
SciPy 1.17.1, PyArrow 25.0.1, scikit-learn 1.9.1 and celer 0.7.4.
These are reference versions, not a claim that every future dependency version
will produce bit-identical coefficients. See the locked environment in the
paper repository for full-result reproduction.

`CombinerConfig(tau_z=0.5)` supplies `tau=0.25` to the Q-inspired update.
Prediction weights are nonnegative but not generally normalized; display-share
normalization must not change predictions.

## Interpretation limits

- Positive-coordinate-penalty fits preserve group-then-coordinate finite
  accelerated updates, not an exact sparse-group proximal optimizer.
- Adaptive weights pool the full historical analysis period.
- Earlier-origin alpha selection uses those origins' complete test blocks,
  which can overlap later origin dates.
- Calibration overlaps the local fitting window.
- Complete-case historical replay uses target observations to construct rows;
  it is not a release-vintage-aware live forecasting service.
- Identical reanalysis of the same data is not independent validation.

## Licensing

The MIT notice supplied with the existing method source is preserved verbatim.
It does not license third-party datasets or proprietary retail inputs.
Paper authors should confirm repository ownership and the final publication
citation before public release; no repository URL or article DOI is invented.
