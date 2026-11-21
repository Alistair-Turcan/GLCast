# Example data

`ctis_aggregate.csv` contains public state-level aggregates from the **U.S.
COVID-19 Trends and Impact Survey (CTIS), Delphi Group at Carnegie Mellon
University, in partnership with Facebook**.

These data are licensed under [Creative Commons Attribution 4.0
International](https://creativecommons.org/licenses/by/4.0/). See the provider's
[data licensing page](https://cmu-delphi.github.io/delphi-epidata/api/covidcast_licensing.html)
and [CTIS description](https://delphi.cmu.edu/epidemic-signals/ctis/).

The example selects CA, NY and TX from January 1 through March 31, 2021: 270
rows covering 90 days per state. Four signals were aligned by date and state,
state identifiers were uppercased, and columns were renamed and sorted. No
values were imputed or rescaled. Signals named `smoothed_...` were already
smoothed by the provider. No provider endorsement is implied.

| Column | Measurement |
| --- | --- |
| `raw_wcli` | Weighted COVID-like illness percentage; the example's target |
| `raw_whh_cmnty_cli` | Community illness percentage |
| `smoothed_wtested_14d` | Percentage tested during the preceding 14 days |
| `smoothed_wtested_positive_14d` | Test positivity percentage |

The dataset contains aggregate estimates, not individual survey records. The
MIT license at the repository root applies to the code; the data retain the
license above.
