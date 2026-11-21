# GLCast

GLCast is a Python package for modeling related geographic time series. It fits
group-lasso models for individual locations and combines their predictions with
models fitted for other locations. It supports daily and weekly data,
configurable lags, and adaptive group weights.

The package provides a historical walk-forward runner and a standalone combiner
for predictions from models you have already fitted. The runner currently
requires observed targets on the dates being predicted; see
[Prediction behavior](#prediction-behavior).

## Installation

Requires Python 3.10 or later. Install from a checkout:

```bash
git clone https://github.com/Alistair-Turcan/GLCast.git
cd GLCast
python -m pip install .
```

To use the optional `celer` solver for ordinary group-lasso fits:

```bash
python -m pip install ".[fast]"
```

Fits with column penalties use the built-in solver with either installation.

## Try the included example

Run a small example using the bundled CTIS aggregate data:

```bash
python examples/quickstart/run.py --output example-output
```

This loads the data, estimates adaptive group weights, and saves historical
predictions. Dataset attribution and licensing are recorded in
[the example data license](examples/quickstart/DATA_LICENSE.md).

## Input data

Provide a CSV with one row per date and location, a numeric target, and one or
more numeric auxiliary signals. For example:

```csv
date,state,target,temperature
2024-01-06,CA,2.1,12.0
2024-01-06,NY,3.4,1.0
2024-01-13,CA,2.3,11.5
2024-01-13,NY,3.2,0.5
```

These rows illustrate the format. A run needs enough history for the configured
training window and lags, followed by the dates to predict.

- `date` contains timezone-naive midnight dates at daily or seven-day intervals. Weekly
  data retain their weekday anchor.
- `state` contains a nonempty location identifier. Identifiers can represent
  locations other than U.S. states.
- `target` and the selected auxiliary columns contain numeric values. Missing
  measurements may be blank; infinite values are rejected.
- Each date/location pair must be unique. Insert explicit missing rows if an
  entire date is absent from the shared calendar.
- Missing measurements are handled by complete-case filtering. GLCast does not
  impute them, so some locations or dates may have no prediction.
- Auxiliary names must not be `AR` or start with `AR_`.

Column names are configurable through the Python API and CLI.

## Configuration

Save the following as `weekly.json`:

```json
{
  "backend": "classic",
  "unit": "W",
  "train_window": 55,
  "val_window": 4,
  "test_window": 4,
  "retrain_every": 1,
  "feature_lags": [0],
  "ar_lags": [2],
  "serve_steps": [2],
  "alpha_y_scale": false,
  "lag_decay": 0.7,
  "n_jobs": 1
}
```

This configuration fits a model each week using a 55-week training window and
returns predictions two weeks after each fitting date, called an *origin*.
`val_window` specifies the trailing calibration window within the training
window. `test_window` specifies the prediction window; `serve_steps` selects
which offsets within it appear in the output.

Lags are measured backward from each row's date. Here, `feature_lags: [0]` uses
that date's auxiliary measurements, and `ar_lags: [2]` uses the target from two
weeks earlier. For daily data, use `unit: "D"` and choose windows and lags in days.

The classic backend supports adaptive group weights and lag penalties.
Omitted options use `MethodConfig` defaults, including the alpha search grid and
uniform group weights. `lag_decay` controls the column penalty multiplier;
set it to `0` to disable that penalty. `MethodConfig()` and the CLI default to
the classic backend with this penalty disabled. The configuration above enables
it explicitly. `fixed_alpha` can select one value from the configured `alphas`
instead of selecting an alpha from earlier predictions.

The `configs/` directory also provides `daily.json`, `weekly_step1.json`, and
`weekly_step2.json`. The weekly files differ only in the prediction offset they
return. The `modern` backend setting remains available for compatibility; the
classic setting supports the full configuration shown here.

## Command line

Check your input and configuration:

```bash
glcast-method validate \
  --input data.csv --target target --aux temperature \
  --config weekly.json
```

Run historical predictions:

```bash
glcast-method run \
  --input data.csv --target target --aux temperature \
  --config weekly.json --output predictions
```

List additional auxiliary columns after `--aux`. Use `--date` and `--state` to
change the date and location column names. `python -m glcast_method` provides
the same commands.

The run command writes:

| File | Contents |
| --- | --- |
| `predictions.parquet` | Location, prediction date, observed target, prediction, origin, and block number |
| `weights.parquet` | Candidate models and their combination weights for each location and origin |
| `alpha_picks.csv` | Selected regularization strength for each origin |
| `configuration.json` | Configuration used for the run |

The prediction columns are `state`, `time_value`, `y_true`, `y_pred`, `origin`,
and `block`.

## Python API

```python
from glcast_method import MethodConfig, load_long_csv, run_walkforward

config = MethodConfig.from_json("weekly.json")
target, features = load_long_csv(
    "data.csv",
    target="target",
    auxiliaries=["temperature"],
    config=config,
)

result = run_walkforward(target, features, config)
print(result.predictions.head())
result.save("predictions")
```

`result` contains three pandas DataFrames: `predictions`, `weights`, and
`alpha_picks`. `result.save()` writes those tables. The CLI additionally saves
`configuration.json`.

### Adaptive group weights

With the classic backend, estimate weights from a first pass and supply them
to the final run. This continues from the data and configuration above:

```python
from dataclasses import replace
from glcast_method import adaptive_weights, fit_first_pass

pilot = fit_first_pass(target, features, config)
weights = adaptive_weights(
    pilot["fits"],
    [*features, "AR"],
    alpha_picks=pilot["alpha_picks"],
)
adaptive_config = replace(config, group_weights=weights)
result = run_walkforward(target, features, adaptive_config)
result.save("adaptive-predictions")
```

The first pass uses uniform group weights and disables column penalties.
Adaptive weights are calculated across that historical first pass. Leaving
`group_weights` unset uses uniform weights; the CLI does not estimate adaptive
weights automatically.

### Combine existing predictions

`combine_candidates` combines a location's own model with candidate models.
Supply observed validation targets, candidate validation predictions, and
candidate predictions for the dates of interest:

```python
import numpy as np
from glcast_method import combine_candidates

combined = combine_candidates(
    y_val=np.array([1.0, 2.0, 3.0, 4.0]),
    validation_predictions=np.array([[1.0, 1.2], [2.0, 1.8],
                                     [3.0, 3.1], [4.0, 3.9]]),
    future_predictions=np.array([[5.0, 4.8], [6.0, 6.1]]),
    names=["self", "other"],
)
if combined is not None:
    print(combined.predictions)
    print(dict(zip(combined.names, combined.weights)))
```

Columns must have the same order in both matrices and in `names`. Column zero
is the location's own model. Order the remaining candidate models by your donor
ranking; by default, only the first 20 are considered, and up to five are retained
using validation error. The combiner excludes donors with nonfinite predictions and
returns `None` if the self model has nonfinite predictions. Future target
values are not needed by this function.

## Prediction behavior

- At each origin, the runner fits local models on complete rows in the configured
  training window. It uses the trailing calibration window to combine candidate
  predictions. Prediction rows require observed targets and the corresponding
  lagged inputs.
- The runner uses `fixed_alpha` when set. Otherwise, it selects alpha from
  prediction errors on complete test windows from earlier origins. Candidates
  without usable errors are excluded. The first origin can produce predictions
  with `fixed_alpha` if it has enough complete rows.
- Group weights default to uniform. To use adaptive weights, estimate them from
  the frequency of nonzero groups in the selected first-pass fits and pass them
  as `group_weights`.
- When multiple origins predict the same location and date, the runner retains
  the latest origin.
- Alpha selection floors candidate predictions at zero when calculating its
  internal errors. Returned combined predictions are unmodified and can be
  negative. Combination weights are nonnegative and need not sum to one.
- `n_jobs` defaults to one. For multiple processes, limit BLAS threads in the
  calling environment to avoid excessive parallelism.

## License

GLCast code is available under the [MIT License](LICENSE). The bundled example
data have their own attribution and license, linked above.
