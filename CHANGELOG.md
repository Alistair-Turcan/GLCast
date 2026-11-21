# Changelog

## 0.2.0

- Simplify installation and usage documentation, configuration examples, and the
  bundled quickstart. Remove research audit files and generated artifacts.
- Consolidate shared numerical code and make the classic configuration the
  default. The `modern` setting remains available for compatibility.
- Correct the sparse-group solver to apply elementwise thresholding before
  group shrinkage, and restart accelerated iterations when needed for convergence.
  These fixes can change fitted coefficients and predictions.
- Exclude alpha candidates without usable historical errors. Permit the first
  origin to produce predictions when `fixed_alpha` is supplied, and fit only
  the selected alpha in that case.
- Make `fit_classic` use direct alpha by default; `alpha_y_scale=True` retains
  target-standard-deviation scaling.
- Reject invalid configuration types and include fitting context in solver errors.
- Test the base and optional-solver installations from built wheels.

Private modules under `_classic` and `_modern` have been replaced by shared
implementation modules. Code importing these private modules must be updated.
