# API reference

Use `import kernelmix as km` and supply inputs as NumPy arrays; pandas is not required. The [tutorial](tutorial.md) shows how the calls form an analysis. The [model guide](methods.md) defines the equations and maps their symbols to API names.

An exposure profile is one row of exposure values. Each posterior draw is a set of parameter values sampled from the distribution given the data and priors. Prediction and contrast objects retain draws jointly across profiles, preserving their dependence.

## Fitting

```python
km.fit(
    y, Z, X=None, *, family="gaussian", draws=1000, warmup=1000,
    chains=4, thin=1, seed=None, standardize=True, intercept=True,
    selection="none", groups=None, prior=None, r_fixed=None, missing="raise",
    exposure_names=None, covariate_names=None, initial=None, proposal_scales=None,
    cluster=None, offset=None, event=None, baseline_breaks=None,
    kernel="gaussian", kernel_alpha=1, approximation="dense", n_components=32,
    knots=None, random_slopes=None, random_covariance=None, random_scale=2,
    correlation=None, observation_time=None, correlation_parameter=None,
    variance_design=None, entry=None, upper=None, subject=None, n_causes=None,
)
```

`fit` returns a `BKMRFit` object. Selecting a structured option returns its `StructuredFit` subclass; the [structured-model guide](structured.md) defines the additional inputs, priors, array shapes and predictions.

| Family | Required outcome inputs | Mean or probability |
| --- | --- | --- |
| Gaussian | Numeric `y` | Continuous outcome mean |
| Binomial | `y` equal to 0 or 1 | Probit probability |
| Poisson / negative binomial | Nonnegative integer `y`; `offset` when needed | Count mean, including the log offset |
| Survival | Default: positive observed times and binary `event`; structured: cause codes, entry and censoring bounds | Relative hazard from `predict`; survival/hazard from `predict_survival` |

`cluster` supplies one string or integer label per observation. Missing labels
participate in `missing="drop"`. Integer `1` and string `"1"` are distinct.
`offset` is a finite vector of log observation exposures for count families.
`event` is required only for survival; `baseline_breaks` supplies internal positive,
strictly increasing cuts. Omit cuts to use event-time deciles, or pass `[]` for a
constant baseline hazard. Family-specific arguments are rejected elsewhere.

| Input | Meaning |
| --- | --- |
| `y`, `Z`, `X` | Outcome vector, exposure matrix and optional adjustment matrix, with matching rows. Supply raw adjustment columns without an intercept when `intercept=True`. |
| `family` | `"gaussian"`, `"binomial"` (probit), `"poisson"`, `"negative_binomial"` (NB2) or `"survival"` (piecewise-exponential proportional hazards). |
| `draws`, `warmup`, `chains`, `thin` | Saved draws per chain, discarded iterations, sequential chains and spacing between saved draws. Total iterations per chain are `warmup + draws * thin`. At least two saved draws are required. |
| `seed` | Nonnegative integer or `None`; uses local random generators without changing NumPy's global random state. |
| `standardize` | Center exposures and divide by retained-training sample standard deviations. Constant exposures are invalid when this is enabled. |
| `selection` | `"none"`, `"component"` or `"group"`. Group selection requires one `groups` label per exposure and permits at most one active exposure in each included group. |
| `prior` | `Prior`, a dictionary of its fields, or `None` for defaults. |
| `r_fixed` | Finite nonnegative vector fixing exposure relevance values; disallows variable selection. |
| `missing` | `"raise"` or joint complete-case `"drop"`. Infinity remains invalid except for right-censoring upper bounds. |
| Names | Optional unique names for exposure and raw adjustment columns. Otherwise use input DataFrame columns, or generated names. |
| `initial` | Shared starting values; omitted entries use defaults. See [initialization](#initialization). |
| `proposal_scales` | Fixed positive log-scale proposal standard deviations (SDs), default `0.3`. All routes accept `r` and `lambda`; default clustered fits add `cluster_lambda`, NB2 adds `dispersion`, and structured survival adds `baseline_hazard`. |

Adjustment columns must be linearly independent, with fewer columns than retained rows. Default probit fits reject complete or quasi-complete separation, where covariates separate outcome classes, possibly with observations on the boundary. With the default flat coefficient prior, the unnormalized posterior density can then have an infinite integral. Structured fits use a proper normal prior.

For computation, dense fitting uses $O(n^2)$ memory and $O(n^3)$ matrix operations, where $n$ is the retained observation count. Finite GP bases reduce this cost, but correlated residual blocks may remain dense.

### Initialization

| Fitting route | `initial` keys |
| --- | --- |
| Default Gaussian/probit | `beta`, `r`, `lambda`, `delta`, `pi`, `sigma2`; probit adds `ystar` |
| Default count/survival | `beta`, `r`, `lambda`, `delta`, `pi`, `latent_white`; NB2 adds `dispersion`, survival adds `baseline_hazard` |
| Default clustered fits | Add `lambda_cluster`; count/survival also add `u_white` |
| Structured | Cause-indexed coefficients and `gp_weight`; see [shapes and restrictions](structured.md#set-starting-values) |

Inapplicable keys raise errors.

### Prior configuration

```python
km.Prior(
    r_distribution="invunif", r_mean=5.0, r_sd=5.0,
    r_lower=0.0, r_upper=100.0, lambda_mean=10.0, lambda_sd=10.0,
    sigma2_shape=0.001, sigma2_rate=0.001,
    inclusion_alpha=1.0, inclusion_beta=1.0,
    cluster_lambda_mean=10.0, cluster_lambda_sd=10.0, beta_sd=5.0,
    dispersion_shape=2.0, dispersion_rate=0.1,
    baseline_shape=0.1, baseline_rate=0.1,
)
```

Check each prior's parameterization when setting its fields. `r_distribution` accepts `"gamma"`, `"unif"` or `"invunif"`; inverse-uniform bounds apply to `1/r` for active exposures, while excluded exposures have `r=0`. Relevance and amplitude gamma priors use mean/SD fields; dispersion and baseline gamma priors use shape/rate. Residual variance uses inverse-gamma shape/rate, and inclusion probabilities use a beta prior. See the [prior definitions](methods.md#priors-and-exposure-selection).

For the default count/survival sampler, `prior=None` uses `lambda_mean=1`, `lambda_sd=.5`,
`cluster_lambda_mean=1`, `cluster_lambda_sd=.5`; the other fields retain their
listed defaults. A dictionary changes the supplied fields and keeps the family's other defaults. A `Prior` object supplies
all its values explicitly, including its listed amplitude defaults. `beta_sd`
controls the proper coefficient prior for count/survival. Dispersion and baseline
parameters use gamma shape/rate. [Structured priors](structured.md#structured-priors)
use a direct GP amplitude and separate Gaussian residual-variance defaults.

## Structured model results

| Task | Method or helper |
| --- | --- |
| Exact alternative kernels | `kernel_matrix(Z1, Z2=None, r=None, *, kernel="gaussian", alpha=1)` |
| Intercept/slope coefficient draws | `fit.random_effects(cluster_new=None, ...)` |
| Gaussian residual variances | `fit.predict_residual_variance(Znew=None, variance_design_new=None, ...)` |
| Cause-specific h and effect summaries | `fit.for_cause(cause)` |
| Competing-risk probabilities | `fit.predict_cumulative_incidence(..., times=..., cause=None)` |
| Time-varying predictor paths | `fit.predict_survival_path(..., start=..., stop=..., subject_new=..., times=...)` |

`StructuredFit` retains the shared summary, sample, diagnostic, contrast and
save/load methods. `RandomCoefficientPrediction` stores coefficient draws;
`CompetingRiskPrediction.for_cause(k)` returns a curve accepted by `plot_survival`.
Additional inputs, initialization shapes and runnable examples are in the [structured guide](structured.md).

## Fit results and diagnostics

`BKMRFit` stores `data`, `samples`, `settings` and `sampling_info`, with `family`,
`exposure_names` and `covariate_names` properties. Stored arrays are read-only;
extracted samples are independent copies.

| Method | Return and use |
| --- | --- |
| `fit.summary(probability=0.95)` | Text summary of parameter means, SDs, equal-tailed intervals, available posterior inclusion probabilities (PIPs) and diagnostic messages. |
| `fit.summary_table(probability=0.95)` | List of dictionaries with `parameter`, `mean`, `sd`, `lower`, `upper`. |
| `fit.posterior_samples(parameter=None, combine_chains=False)` | Dictionary of arrays, or one named parameter array. Leading axes are chain and retained draw; combining chains merges those axes. |
| `fit.inclusion_probabilities()` | Exposure names and PIPs for selection fits. Group fits also return group PIP and within-group conditional PIP; the conditional value is undefined (`NaN`) if a group was never selected. Raises without selection. |
| `fit.diagnostics()` | Parameter diagnostics and messages. Uses rank-normalized/folded split R-hat and bulk/tail effective sample sizes. |
| `fit.save(path)` | Save training data, settings and states in a compressed non-pickle archive. Replaces the destination atomically. |
| `km.BKMRFit.load(path, *, max_bytes=512 * 1024 * 1024)` | Load and validate an archive, limiting total uncompressed member size. |
| `km.diagnose(values)` | Diagnostics for one finite scalar-parameter array shaped `(chains, draws)`. |

The reported intervals are equal-tailed: the posterior probability below the lower limit equals that above the upper limit. A PIP is the fraction of draws including an exposure or group. R-hat compares chains; bulk/tail effective sample sizes account for draw dependence in the distribution's center/tails.

Default-route saved parameters include `beta`, `r`, `lambda`, `sigma2`, `delta` and `pi`; its binary fits also retain `ystar`. Structured arrays are described in the [structured guide](structured.md). `delta` records exposure inclusion. `beta` includes the added intercept when fitted. No-selection fits store `pi=1`. Diagnostics cannot establish model validity; R-hat requires multiple chains and short or constant sequences can yield undefined diagnostics.

## Prediction

```python
fit.predict_h(Znew=None, *, probability=0.95, max_draws=None, seed=None)
fit.predict(
    Znew=None, Xnew=None, *, kind="mean", scale="response",
    probability=0.95, max_draws=None, seed=None, offset=None, cluster_new=None,
)
fit.posterior_moments_h(Znew=None, max_draws=None)
```

Supply exposures in original units and training column order; named columns must
match. Prediction reuses training standardization. New profiles require `Xnew`
when adjustments were fitted, without the added intercept. Omitting new data
uses retained training rows.

| Call | Quantity |
| --- | --- |
| `predict_h` | Joint draws of `h`, excluding linear adjustment and all random effects. |
| `predict(..., kind="mean", scale="response")` | Gaussian/count means, binary probabilities, or survival relative hazards. |
| `predict(..., kind="observation", scale="response")` | Gaussian, binary or count future observations; survival is unsupported. |
| `predict(..., kind="mean", scale="latent")` | `h` plus linear adjustment and requested random effects, before link transformation; count offsets are included. Correlated Gaussian means also condition on training residuals. |
| `predict(..., kind="observation", scale="latent")` | Gaussian outcomes or binary latent outcomes including residual noise; binary residual variance is one. Count/survival latent observation draws are unsupported. |
| `posterior_moments_h` | Dictionary with deterministic integrated `mean` and `covariance` for `h`, averaging conditional moments over saved draws. |

`max_draws` selects evenly spaced states **per chain** (at least two), without
refitting. `seed` controls prediction draws. A `Prediction` stores `draws` shaped
`(chains, selected_draws, rows)`, `mean`, `sd`, `lower`, `upper`, joint `covariance`,
`probability`, `scale` and `kind`. `to_records()` gives one summary per row.
Intervals are equal-tailed and pointwise: each describes uncertainty for one row, rather than for all rows at once. Observation intervals also include new outcome variation.

## Cluster and survival results

```python
fit.random_intercepts(cluster_new=None, *, probability=.95, max_draws=None, seed=None)
fit.predict_survival(
    Znew=None, Xnew=None, *, times, kind="survival", cluster_new=None,
    probability=.95, max_draws=None, seed=None,
)
km.plot_survival(prediction, row=0, ax=None)
```

`random_intercepts` returns a `Prediction` in training-cluster order, or in the supplied label order. The prediction rules for known and new clusters are described below.

| Survival result | Meaning |
| --- | --- |
| `times` | Finite, nonnegative, strictly increasing prediction times |
| `draws` | Array with axes `(chain, draw, row, time)` |
| `mean`, `sd`, `lower`, `upper` | Arrays shaped `(row, time)` |
| `probability`, `kind` | Interval probability and survival/hazard scale |
| `to_records()` | One summary per profile/time pair |

`kind="hazard"` evaluates hazards on the grid. Plots use survival curves or
hazard points with intervals; they do not invent hazard steps between times.

`predict` additionally accepts `offset=None` and `cluster_new=None`.
For training count prediction, omitted offsets reuse retained offsets. For new
rows, offsets are required if they were supplied during fitting. Response-scale
count observations are Poisson or NB2 draws. Latent count observations and survival
observation draws are unsupported and raise an error.

Omitted `cluster_new` sets random effects to zero, including for training
predictions. Known labels use their fitted cluster coefficients; unseen labels receive new Gaussian
coefficients. Equal labels share a draw. `predict_h` excludes random effects.
For slopes and residual conditioning, see [structured prediction](structured.md#separate-gaussian-variance-and-correlation). Default Gaussian/probit prediction samples `h + u` jointly when labels are
supplied, preserving posterior covariance.

For the default count/survival route, samples include `h` and auxiliary `latent_white`; negative-binomial
fits also include `dispersion`; survival fits include `baseline_hazard`.
Default-route clustered fits include `lambda_cluster`, and count/survival fits also retain `u`.
Default Gaussian/probit fits integrate out random intercepts during parameter sampling; `random_intercepts` draws them conditionally afterward. Count/survival fits store `sigma2=1` for a consistent result format. It is not an estimated residual variance and is omitted from their summaries. Saved white coordinates are internal standardized sampling variables tied to a matrix decomposition. Use saved `h` for prediction that can be transferred across platforms.

## Exposure contrasts and curves

Exposure contrasts compare predictions at specified profiles. A quantile profile sets each exposure to a percentile of its own distribution; the all-median profile uses every exposure's median. Such combinations need not occur in observed rows.

```python
km.overall_effect(
    fit, quantiles=(0.25, 0.5, 0.75), reference=0.5, scale="latent",
    X=None, probability=0.95, max_draws=None, seed=None,
)
```

```python
km.univariate_response(
    fit, variable=None, grid=None, points=50, fixed_quantile=0.5,
    center=True, scale="latent", X=None, probability=0.95,
    max_draws=None, seed=None,
)
```

```python
km.bivariate_response(
    fit, variables=(0, 1), grids=None, points=20, fixed_quantile=0.5,
    center=True, scale="latent", X=None, probability=0.95,
    max_draws=None, seed=None,
)
```

```python
km.single_exposure_effects(
    fit, quantiles=(0.25, 0.75), fixed_quantiles=(0.25, 0.5, 0.75),
    scale="latent", X=None, probability=0.95, max_draws=None, seed=None,
)
```

```python
km.interaction_effects(
    fit, variables=None, quantiles=(0.25, 0.75), fixed_quantile=0.5,
    scale="latent", X=None, probability=0.95, max_draws=None, seed=None,
)
```

| Function | Quantity and result |
| --- | --- |
| `overall_effect` | Each all-exposure quantile profile minus the all-exposure `reference` profile. Returns `EffectResult`. |
| `univariate_response` | One named or zero-based indexed exposure, or all exposures with `variable=None`. Other exposures stay at `fixed_quantile`. Returns one joint `CurveResult`. |
| `bivariate_response` | Every combination of the supplied grid values for two distinct exposures. Other exposures stay at `fixed_quantile`. Returns `SurfaceResult`. |
| `single_exposure_effects` | High-minus-low quantile contrast for each exposure at every background `fixed_quantiles` value. Returns `EffectResult`. |
| `interaction_effects` | All unordered pairs from the selected names/indices, or all exposures by default. Returns `EffectResult`. For each pair the contrast is high/high - low/high - high/low + low/low, with other exposures fixed. |

Quantiles use retained original training exposures and lie in `[0, 1]`. Default
grids span the observed range; explicit grids must be finite, nonempty and
strictly increasing. An all-variable curve shares one supplied `grid`; a surface
uses one sequence per axis in `grids`. Dense joint calculations limit calls to
4096 output locations.

With `center=True`, curves and surfaces subtract the **all-median** reference profile, even if `fixed_quantile` is changed. This differs from R `bkmr` curve centering by the average fitted grid value. The pairwise interaction above also differs from the R summary that changes all background exposures together. These definitions should be specified when comparing results between packages.

| Summary scale | Target |
| --- | --- |
| Latent | Contrasts of `h`; `X` is rejected |
| Response | Gaussian/count means, binary probabilities or survival relative hazards; random effects are zero |
| Count response | Zero log offset, representing unit observation exposure |
| Survival response | Differences in relative hazards, not hazard ratios or survival-probability differences |

For response summaries, `X` fixes one raw adjustment vector, defaulting to training covariate means. Each joint draw is transformed to the requested scale before subtraction. This retains covariance and makes identical-profile contrasts exactly zero.

Results store `draws` shaped `(chains, selected_draws, locations)`, `mean`, `sd`,
`lower`, `upper`, `scale`, `probability`, `kind` and `metadata`. `locations` labels
outputs; `profiles` holds distinct original-unit rows; `coefficients` maps joint
profile draws to summaries. `to_records()` combines locations and summaries.
`SurfaceResult` adds `variables`, `grids` and `shape`; reshape summaries to `shape`
with the second grid changing fastest.

These summaries describe associations at the specified values of the other variables. Check that nearby exposure combinations occur in the data. The intervals do not account for unmeasured confounding or a wrongly specified model.

## Plotting

Matplotlib is an optional dependency and is imported only when a plot is requested.

```python
km.plot_trace(fit, parameter="lambda", indices=None, ax=None)
km.plot_pips(fit, ax=None)
km.plot_response(curve, ax=None)
km.plot_overall(effect, ax=None)
km.plot_surface(surface, ax=None)
km.plot_survival(prediction, row=0, ax=None)
```

Each function returns axes, creating them when `ax=None`. The caller controls
display, export and styles. Trace `indices` selects a vector component, such as
`indices=0` for the first `beta`. Traces exclude warmup. PIP plots require selection.

`plot_response` accepts a univariate curve; `plot_overall` accepts an overall effect. Both show pointwise posterior intervals. `plot_surface` needs at least two values per grid and shows the posterior mean, with zero-centered colors when centered; its result retains interval limits. Use separate curves for different exposure units.

## Kernel helper

```python
km.gaussian_kernel(Z1, Z2=None, r=None)
```

Returns the Gaussian kernel between supplied rows. `Z2=None` reuses `Z1` and
`r=None` uses unit relevance. Relevance must be nonnegative. Inputs are used
directly, without a fitted training transform.

## Method sources

- [Bobb et al. (2015), Bayesian kernel machine regression for estimating the health effects of multi-pollutant mixtures](https://doi.org/10.1093/biostatistics/kxu058).
- [Bobb et al. (2018), Statistical software for analyzing the health effects of multiple concurrent exposures via Bayesian kernel machine regression](https://doi.org/10.1186/s12940-018-0413-y).

- [Murray et al. (2010), Elliptical slice sampling](https://proceedings.mlr.press/v9/murray10a.html).
- [Stan count-distribution definitions](https://mc-stan.org/docs/functions-reference/unbounded_discrete_distributions.html).
- [INLA manual, piecewise-constant proportional hazards (§3.15)](https://inla.r-inla-download.org/r-inla.org/doc/inla-manual/inla-manual.pdf).
