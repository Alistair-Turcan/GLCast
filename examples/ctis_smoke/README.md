# CTIS aggregate-data method smoke test

**This is a non-paper demonstration, not a reproduction of a manuscript target
or prospective validation.** It uses only public, state-level CTIS aggregate
measurements; there are no individual survey records, retail sales, Google
search measurements, or synthetic observations.

## Data and attribution

Source: **U.S. COVID-19 Trends and Impact Survey (CTIS), Delphi Group at
Carnegie Mellon University, in partnership with Facebook**. Public aggregates
are provided under **Creative Commons Attribution 4.0 International**:
https://creativecommons.org/licenses/by/4.0/ .
See the provider's licensing and CTIS pages for the distinction between public
aggregates and individual survey access. No provider endorsement is implied.
[1, 2]

The four original aggregate CSVs were supplied with the audit's source data.
Their names and SHA-256 checksums are recorded in `provenance.json`. Selection:
CA, NY and TX, daily 2021-01-01 through 2021-03-31. Transformations: select
state aggregates and dates, uppercase state identifiers, align four signals,
rename columns and sort. **No imputation, standardization, smoothing or new
observations are performed by this subset builder.** Some provider signals are
already labeled smoothed in the original files.

| Column | Role | Unit |
|---|---|---|
| `state` | CA, NY, TX | jurisdiction |
| `date` | Daily date label | ISO date |
| `raw_wcli` | Target: weighted COVID-like illness aggregate | percent |
| `raw_whh_cmnty_cli` | Auxiliary: community illness aggregate | percent |
| `smoothed_wtested_14d` | Auxiliary: tested during preceding 14 days | percent |
| `smoothed_wtested_positive_14d` | Auxiliary: test positivity aggregate | percent |

The subset contains **270 rows, 90 days per jurisdiction**, with all four
measurements observed in each row. These records are aggregate estimates, not
case counts or a distinct independent validation cohort. Original provider
issue dates are not used as historical availability cutoffs.

## Execute

From the repository root after installing the method with its `fast` extra:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python examples/ctis_smoke/run_example.py --output ctis_outputs
```

This runs the public API in two passes: uniform/no-column-penalty first pass,
retrospective adaptive group weights, then final historical predictions with
column multiplier 0.7 and the recovered geographic combiner. The demonstration
uses daily train/validation/test windows of 28/7/7 days, retrains every seven
days, auxiliary lags 0–1, AR lags 1–2, three declared alphas
`0.001, 0.01, 0.1`, and serves steps 1–7. The first origin is burn-in.
These settings are for a small executable example, not paper settings or a
new tuning exercise.

Expected structural outputs from the validated source-checkout execution:
8 scheduled origins, **147 served prediction rows**, 63 combination-weight
rows and a finite four-group mean-one adaptive-weight mapping.
Floating-point outputs depend on the numerical backend/dependency versions;
the receipt records actual versions and elapsed time. Raw combination weights
need not sum to one. Predictions remain un-clipped.

```bash
glcast-method validate \
  --input examples/ctis_smoke/ctis_aggregate.csv \
  --target raw_wcli \
  --aux raw_whh_cmnty_cli smoothed_wtested_14d smoothed_wtested_positive_14d \
  --config examples/ctis_smoke/config.json
```

The CLI validation command checks the input/configuration; it does not perform
the adaptive first pass. Running the CLI `run` with the unmodified template
would use uniform group weights. Use `run_example.py` for the complete explicit
two-pass demonstration, or pass its resulting `configuration.json` to the CLI.

## Rebuild the identical subset

If you hold the four original files listed in `provenance.json`:

```bash
python examples/ctis_smoke/rebuild_subset.py \
  --source-directory your_original_ctis_csvs \
  --output rebuilt_ctis_aggregate.csv
```

The builder refuses mismatched source hashes, duplicate source keys, unexpected
coverage or missing selected values, and verifies the rebuilt CSV checksum.
It does not silently fill gaps or download a different data vintage.

## References

1. Data Licensing | Delphi Epidata API. [cited 2026 Nov 19]. Available from: https://cmu-delphi.github.io/delphi-epidata/api/covidcast_licensing.html
2. Delphi's COVID-19 Trends and Impact Surveys (CTIS) | DELPHI. [cited 2026 Nov 19]. Available from: https://delphi.cmu.edu/epidemic-signals/ctis/
