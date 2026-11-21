# Source and scope

The recovered GLCast source carries the MIT license in `LICENSE`, copyright
2026 Alistair Turcan, Ruiqi Lyu and Bryan Wilder.

This method-only distribution separates two recovered numerical engines:

- `_modern`: the fitting/grid and donor task used with the revised modern
  combiner, plus the pooled earlier-origin alpha selector.
- `_classic`: the recovered padded-group, adaptive-weight-capable sparse-group
  fitter and grid; the donor task shares the recovered transport implementation.

The public combiner was extracted from the revised modern driver; the reference
selection, stacking objective, validation scaling, Q iteration and blending
constants are retained. The public input/CLI/configuration wrappers and helper
interfaces were added for portability. Analysis code, data loaders, baseline
methods and result-generation utilities were omitted.

`SOURCE_HASHES.json` records the SHA-256 digest of each original numerical source
module and of its extracted packaged version. Function bodies were copied
without changing their numerical statements; module headers, unused imports
and unrelated functions can differ. Numerical caveats are listed in `METHOD.md`.

The three S3 templates all select the classic numerical backend with direct
alpha, column multiplier 0.7, automatic pooled alpha choices and initially
uniform weights. Adaptive weights require explicit first-pass preparation.
The public configuration validator's one S3 change permits direct alpha with
positive column multiplier; no optimizer, engine, transport or combiner
numerical statement was changed. The parent bundle's
`provenance/method_revision_audit.json` records the allowlisted difference.

The templates are components/configuration templates, not a claim that
this folder recreates every paper-specific evaluation detail. In particular,
multi-cadence fallback overlays and scored-cell masks are outside this package.
The retained modern engine is not used by the S3 analyses.

Dependency versions used to validate this distribution are recorded in
`VALIDATION.md`. The independent audit distribution now includes one explicitly
attributed public aggregate-data example in `examples/ctis_smoke/`; it contains
no retail, Google search or respondent-level inputs. The example is not a paper
target or a new validation cohort. Its dataset is CC BY 4.0, separately from the
method code's MIT license. Provider attribution and exact source/subset checksums
are recorded with the example.
