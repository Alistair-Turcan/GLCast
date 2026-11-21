# Method and numerical conventions

## Local fits

The numerical backends retain their recovered implementations, rather than
substituting a newly written optimizer. Designs contain each auxiliary stream's
lagged columns, followed by target autoregression columns. Training-window
means/standard deviations use `ddof=0`; constant columns/targets use SD one.
Intercepts are recovered from training centering.

All S3 templates use the classic numerical backend and `alpha_y_scale=false`:
alpha is passed directly to the standardized solver. The older alpha/response-SD
convention remains a supported library option but is not used in S3. These
conventions are not interchangeable. Numeric alpha grids and the raw stacking
ridge remain sensitive to target units.

S3 first-pass selection frequencies pool origin/state fits after choosing the
uniform/no-column pass's own pooled earlier-origin alpha. A group is selected if any coefficient is
nonzero. For requested groups, compute `1/(frequency + 0.05)` and divide by their
mean. Missing group frequencies are set to zero; unavailable groups do not
create fabricated observations. Stream names, not column positions, align the
resulting weights to each state's design.

S3 column penalties have base `0.7 * alpha` (no response-SD division).
Auxiliary columns multiply this base by `(lag + 1)`; AR columns use the base.
The trailing AR group is zero-padded by the recovered solver when shorter than
the common auxiliary group size.

### Optimizer limitations retained for numerical fidelity

The ordinary group fit uses celer when available, otherwise the recovered FISTA
fallback. The classic sparse-group path uses the recovered sequence of a group
threshold followed by a column threshold. That ordering is preserved here for
fidelity; it should **not be presented as a newly verified exact proximal solver
for the displayed sparse-group objective**. The screening loop scans plain
group residual norms, at positive tolerance, and has an iteration cap. It does
not provide an exact global-optimality certificate.

Changing these numerical algorithms would define a different implementation
and requires separate validation. This package does not silently repair them.

## Donor library and transport

On training-standardized data, form each state's vector of predictor/response
inner products. Rank donors by the sum of squared differences over the largest
absolute discrepancies (screen default 50, limited by available common
coordinates). Keep the first twenty donors.

The recovered geographic task aligns donor **raw-scale coefficients by feature
name**, fills missing coefficients with zero, retains the donor intercept and
evaluates that model on the target state's raw design. This is the as-run
transport implementation; do not reinterpret it as a different standardized-
coefficient transport rule. Self uses the target-state fit at the selected alpha.

## Unified combiner

1. Require finite self validation and prediction columns. Discard nonfinite
   donors. Candidate filtering can use predictor availability, but not future
   outcome values.
2. From the twenty-member training-ranked donor library, select up to five
   donors with smallest validation RMSE. Stable ties retain donor order.
3. Fit nonnegative stacking weights minimizing
   `||y - Fw||²/(2n) + 1000*sum(w_donor²)/2`. Self is unpenalized, there is no
   extra intercept, and the weights are not constrained in sum. The augmented
   NNLS design adds `sqrt(1000*n)*diag(0,1,...,1)`.
4. Divide validation outcomes and candidate predictions by the target validation
   SD (`ddof=0`, fallback one). The Q-parent runs at `tau = 0.5²`.
5. Return `w = gamma*w_stack + (1-gamma)*w_q`,
   `gamma = n/(n+10)`, and predictions `F_future @ w`.

The Q-parent is the **implemented ten-step damped iteration**, with uniform
prior, `eps=1e-3`, initial log weights `-RSS/(2*tau)`, and update log weights
`-RSS/(2*tau) + distance_to_current_aggregate²/(8*tau)`.
The aggregate used within the iteration updates with 0.25 new and 0.75 old.
This is not silently replaced by the different fixed point of the manuscript's
displayed half/half Q-objective. The package does not assert that objective has
been exactly minimized.

Stacking is on the raw scale, Q on a validation-SD-rescaled scale. The blend's
nonnegative weights need not sum to one; do not renormalize them. Raw predictions
may be negative when some candidates are negative. Zero-flooring, if desired,
is a recipient's downstream display/evaluation choice, not applied internally.

## Historical timing and lag gate

The runner reproduces the existing construction:

- Training includes origin O and the preceding `train_window-1` steps.
- Validation is the last `val_window` rows **inside** that training interval,
  not a separate held-out fitting set.
- Test is O+1 through O+test_window; complete-case construction includes observed
  targets, so unknown future targets are excluded by the high-level runner.
- All earlier-origin test-window prediction errors enter pooled alpha selection.
  With overlapping windows these can include labels after the current origin;
  "earlier origin" is not equivalent to a release-cutoff guarantee.
- Alpha selection clips predictions below zero for its internal loss, matching
  the recovered picker. The returned combined predictions remain raw.
- The first origin is burn-in; the default modern configuration serves only O+2
  weeks. Other configurations explicitly specify `serve_steps`.
- Cadence is fixed elapsed days or seven-day weeks, preserving weekday anchoring.

S3 adaptive weights pool the supplied uniform/no-column refits once per analysis,
for both classic targets and all seven modern targets. Weights are normalized
to mean one including AR and held fixed for final and monthly fallback fitting.
The optional library lag gate keeps auxiliary lags 0–3 only on a strict caller-supplied improvement
over the deployed set. If its comparison errors come from the scored period,
the selection is retrospective. V3 does not apply that gate or select a new main
lag design: the four expanded designs are sensitivity analyses only. The helper neither computes evaluation
statistics nor makes that selection prospective.

Global constants were inherited, not re-tuned by this package. A release-aware
prospective variant would require a separately specified data-vintage cutoff,
availability-aware loss history, a validation-only lag gate and causal weight
preparation. Such a variant is outside this packaging task.

## APIs and schemas

`combine_candidates` accepts `y_val (n,)`, `F_val (n,J)`, `F_future (m,J)`,
and J distinct candidate names; column zero is self and remaining columns must
already be in discrepancy order. It returns predictions, final and parent
weights, names, gamma and the selected future design, or `None` for invalid self.

`combine_origin` adapts recovered blocks with keys `y_val`, `self_val`,
`self_test`, `donors`, `donor_val`, `donor_test` at a specified alpha index.
Donor tensors are donor × alpha × row.

`fit_first_pass` returns tables. `adaptive_weights` requires first-pass columns
`origin,state,alpha,feature,coef` with features named `stream_lagN`/`AR_lagN`.
Supply exactly one of `alpha_ref` or a table `origin,alpha` of selected penalties.
Its output is a JSON-compatible group-name/weight mapping.

`fit_classic` accepts a finite design, outcomes and ordered feature names; it
returns the recovered solver tuple (active group indices, raw coefficients,
intercept, standardized coefficients). Group size restrictions are validated.

`choose_lags` accepts deployed/expanded lag lists and two finite nonnegative
errors. Errors must be comparable (same metric, cells and aggregation), a
responsibility of the caller. It returns the expanded list only for strict
improvement.

`run_walkforward` returns `RunResult(predictions, weights, alpha_picks)`.
Prediction columns are `state,time_value,y_true,y_pred,origin,block`.
Weights include `candidate,weight,w_stack,w_qagg,gamma` plus state/origin/block.
No scores, p-values, comparisons or plots are computed.
