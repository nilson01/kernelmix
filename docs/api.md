# API reference

Import the public API with `import kernelmix as km`. Arrays are NumPy arrays; pandas is not required. Examples that connect these functions are in the [tutorial](tutorial.md).

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

Returns `BKMRFit`, or its `StructuredFit` subclass when a structured option is selected.
[Structured models](structured.md) defines the additional inputs, priors, shapes
and predictions. Original default fitting paths remain unchanged.

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
| `draws`, `warmup`, `chains`, `thin` | Retained states per chain, discarded iterations, sequential chains and spacing between retained states. Total iterations per chain are `warmup + draws * thin`. At least two retained states are required. |
| `seed` | Nonnegative integer or `None`; uses local random generators without changing NumPy's global random state. |
| `standardize` | Center exposures and divide by retained-training sample standard deviations. Constant exposures are invalid when this is enabled. |
| `selection` | `"none"`, `"component"` or `"group"`. Group selection requires one `groups` label per exposure and permits at most one active exposure in each included group. |
| `prior` | `Prior`, a dictionary of its fields, or `None` for defaults. |
| `r_fixed` | Finite nonnegative vector fixing exposure relevance values; disallows variable selection. |
| `missing` | `"raise"` or joint complete-case `"drop"`. Infinity remains invalid except for right-censoring upper bounds. |
| Names | Optional unique names for exposure and raw adjustment columns. Otherwise use input DataFrame columns, or generated names. |
| `initial` | Shared chain initialization dictionary. Basic-sampler keys: `beta`, `r`, `lambda`, `delta`, `pi`. Gaussian/probit use `sigma2` and binary `ystar`; new families use `latent_white`, with NB2 `dispersion` or survival `baseline_hazard`. Clusters add `lambda_cluster` and new-family `u_white`. Omitted values receive defaults; inapplicable keys raise errors. Structured models use cause-indexed coefficients and `gp_weight`; see their guide. |
| `proposal_scales` | Positive log-scale standard deviations for `r`, `lambda`, and applicable `cluster_lambda` or NB2 `dispersion`. Defaults are `0.3`; scales stay fixed. |

The adjustment design must have full column rank and fewer columns than retained observations. The default binary sampler rejects complete or quasi-complete separation under its flat coefficient prior; structured fits use a proper normal prior. Default fitting uses dense covariance matrices, with quadratic memory and cubic linear-algebra cost. Predictive-process and random-Fourier options replace the GP with the selected finite basis; dense residual correlation blocks can remain costly. Successful fitting does not establish convergence.

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

`r_distribution` accepts `"gamma"`, `"unif"` and `"invunif"`. The inverse-uniform option places the uniform distribution on `1/r`, so its bounds apply to the reciprocal of an active relevance value. Excluded exposures have relevance zero. Gamma priors use mean and standard deviation; the residual variance uses inverse-gamma shape and rate. Inclusion probabilities use a beta prior. The residual variance prior is used for Gaussian outcomes.

Count/survival `prior=None` uses `lambda_mean=1`, `lambda_sd=.5`,
`cluster_lambda_mean=1`, `cluster_lambda_sd=.5`; the other fields retain their
listed defaults. A dictionary overlays family defaults. A `Prior` object supplies
all its values explicitly, including its listed amplitude defaults. `beta_sd`
controls the proper coefficient prior for count/survival. Dispersion and baseline
parameters use gamma shape/rate; other-family fields are unused as documented.


## Structured model results

| Task | Method or helper |
| --- | --- |
| Exact alternative kernels | `kernel_matrix(Z1, Z2=None, r=None, kernel="gaussian", alpha=1)` |
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

`BKMRFit` retains `data`, `samples`, `settings` and `sampling_info`. Properties `family`, `exposure_names` and `covariate_names` describe the fitted inputs. Stored arrays are copied and read-only; extracted samples are independent copies.

| Method | Return and use |
| --- | --- |
| `fit.summary(probability=0.95)` | Text summary of parameter means, SDs, equal-tailed intervals, available PIPs and diagnostic messages. |
| `fit.summary_table(probability=0.95)` | List of dictionaries with `parameter`, `mean`, `sd`, `lower`, `upper`. |
| `fit.posterior_samples(parameter=None, combine_chains=False)` | Dictionary of arrays, or one named parameter array. Leading axes are chain and retained draw; combining chains merges those axes. |
| `fit.inclusion_probabilities()` | Exposure names and PIPs for selection fits. Group fits also return group PIP and within-group conditional PIP; the conditional value is undefined (`NaN`) if a group was never selected. Raises without selection. |
| `fit.diagnostics()` | Parameter diagnostics and messages. Uses rank-normalized/folded split R-hat and bulk/tail effective sample sizes. |
| `fit.save(path)` | Save training data, settings and states in a compressed non-pickle archive. Replaces the destination atomically. |
| `km.BKMRFit.load(path, *, max_bytes=512 * 1024 * 1024)` | Load and validate an archive, limiting total uncompressed member size. |
| `km.diagnose(values)` | Diagnostics for one finite scalar-parameter array shaped `(chains, draws)`. |

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

All new exposure values use original units and the training column order. Named input columns must match the stored names and order. Training standardization is reused. `Xnew` contains raw adjustment columns without the intercept; it is required for new profiles when adjustments were fitted. With no new data, retained training rows and their adjustments are used.

| Call | Quantity |
| --- | --- |
| `predict_h` | Joint draws of `h`, excluding linear adjustment and random intercepts. |
| `predict(..., kind="mean", scale="response")` | Gaussian/count means, binary probabilities, or survival relative hazards. |
| `predict(..., kind="observation", scale="response")` | Gaussian, binary or count future observations; survival is unsupported. |
| `predict(..., kind="mean", scale="latent")` | `h` plus linear adjustment and requested random intercept; count log offsets are included. This is before the probit/log-link transformation. |
| `predict(..., kind="observation", scale="latent")` | Gaussian outcomes or binary latent outcomes including residual noise; binary residual variance is one. Count/survival latent observation draws are unsupported. |
| `posterior_moments_h` | Dictionary with deterministic integrated `mean` and `covariance` for `h`, averaging conditional moments over saved states. |

`max_draws` selects at most that many evenly spaced saved states **per chain**, with a minimum of two. It does not refit. `seed` controls conditional prediction draws. A `Prediction` contains `draws` shaped `(chains, selected_draws, rows)`, `mean`, `sd`, `lower`, `upper`, joint `covariance`, `probability`, `scale` and `kind`. Its `to_records()` method returns one summary dictionary per row. Intervals are equal-tailed pointwise posterior intervals; observation intervals also include observation noise.

## Cluster and survival results

```python
fit.random_intercepts(cluster_new=None, *, probability=.95, max_draws=None, seed=None)
fit.predict_survival(
    Znew=None, Xnew=None, *, times, kind="survival", cluster_new=None,
    probability=.95, max_draws=None, seed=None,
)
km.plot_survival(prediction, row=0, ax=None)
```

`random_intercepts` returns a `Prediction` in training-cluster order by default,
or supplied label order. Known and unseen labels follow the policy below.
`predict_survival` requires finite, nonnegative, strictly increasing `times`.
Its `SurvivalPrediction.draws` has axes `(chain, draw, row, time)`; `mean`, `sd`,
`lower` and `upper` have shape `(row, time)`. Fields `times`, `probability` and
`kind` identify the grid and summary. `to_records()` returns one row per
profile/time pair. `kind="hazard"` evaluates hazards on that grid. The plot
shows a survival curve, or hazard points and intervals without inventing step
locations between a sparse set of times.

`predict` additionally accepts `offset=None` and `cluster_new=None`.
For training count prediction, omitted offsets reuse retained offsets. For new
rows, offsets are required if they were supplied during fitting. Response-scale
count observations are Poisson or NB2 draws. Latent count observations and survival
observation draws are unsupported and raise an error.

Omitted `cluster_new` fixes the intercept to zero, including for training
predictions. Known labels use their posterior; unseen labels use a new Gaussian
intercept. Equal labels share a draw. `predict_h` always excludes random
intercepts. Default Gaussian/probit prediction samples `h + u` jointly when labels are
supplied, preserving posterior covariance.

For the default count/survival route, samples include `h` and auxiliary `latent_white`; negative-binomial
fits also include `dispersion`; survival fits include `baseline_hazard`.
Default-route clustered fits include `lambda_cluster`, and count/survival fits also retain `u`.
Default Gaussian/probit random intercepts are integrated out while fitting and drawn
conditionally by `random_intercepts`. `sigma2=1` is only a shared latent convention
for count/survival and is omitted from their summaries. Saved white coordinates
refer to the sampler's eigensystem; use saved `h` for portable prediction.


## Exposure contrasts and curves

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
| `bivariate_response` | A Cartesian grid for two distinct exposures. Other exposures stay at `fixed_quantile`. Returns `SurfaceResult`. |
| `single_exposure_effects` | High-minus-low quantile contrast for each exposure at every background `fixed_quantiles` value. Returns `EffectResult`. |
| `interaction_effects` | All unordered pairs from the selected names/indices, or all exposures by default. Returns `EffectResult`. For each pair the contrast is high/high - low/high - high/low + low/low, with other exposures fixed. |

Quantiles come from retained original training exposures and must lie in `[0, 1]`. Default grids span the observed range. Explicit grids must be finite, nonempty and strictly increasing. For an all-variable univariate call, a supplied `grid` is shared across exposures. `grids` contains one sequence for each surface axis. Calls are limited to 4096 output locations because joint prediction and contrast matrices are dense.

With `center=True`, curves and surfaces subtract the **all-median** reference profile, even if `fixed_quantile` is changed. This differs from R `bkmr` curve centering by the average fitted grid value. The pairwise interaction above also differs from the R summary that changes all background exposures together. These definitions should be specified when comparing results between packages.

Latent summaries use `h`. Response summaries use Gaussian/count means, binary probabilities or survival relative hazards, holding `X` constant and random intercepts at zero. Count summaries use zero log offsets (unit observation exposure). Survival summaries subtract relative hazards; they do not return survival-probability contrasts or hazard ratios. `X` is one raw adjustment vector; when omitted it defaults to retained training covariate means. `X` is rejected for latent summaries. Binary probabilities are transformed before contrasts are calculated. All contrasts use joint draws to preserve covariance and exactly cancel identical profiles.

Result objects contain `draws` with axes `(chains, selected_draws, locations)`, `mean`, `sd`, `lower`, `upper`, `locations`, `profiles`, `coefficients`, `scale`, `probability`, `kind` and `metadata`. `locations` identifies each variable, pair, quantile or grid value. `profiles` stores distinct original-unit prediction rows; `coefficients` maps their joint draws to reported summaries. `to_records()` returns location fields and numeric summaries. A `SurfaceResult` also contains `variables`, `grids` and `shape`; reshape its flat summaries to `shape`, with the second grid changing fastest.

These summaries describe conditional associations. Marginal quantile profiles can combine values rarely observed together. Credible intervals do not address poor exposure support, unmeasured confounding or model misspecification.

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

Each function returns a Matplotlib axes object, creates one when `ax=None`, and leaves display, saving and global styles to the caller. `indices` selects a scalar component for a vector-valued trace parameter, for example `indices=0` for the first `beta` coefficient. Traces contain retained draws only. PIP plots require a selection fit.

`plot_response` accepts a univariate curve; `plot_overall` accepts an overall effect; `plot_surface` accepts a bivariate surface with at least two values on each grid. Curve and overall plots show pointwise posterior intervals. A surface shows the posterior mean only, with zero-centered colors when centered. Result objects retain its uncertainty. Request separate curves for exposures with different units.

## Kernel helper

```python
km.gaussian_kernel(Z1, Z2=None, r=None)
```

Returns a Gaussian kernel matrix between the supplied exposure rows. `Z2=None` uses `Z1` for both sides; `r=None` uses unit relevance for each column. Relevance values are nonnegative. This helper uses the values supplied directly and does not apply a fit's training transform.

## Method sources

- [Bobb et al. (2015), Bayesian kernel machine regression for estimating the health effects of multi-pollutant mixtures](https://doi.org/10.1093/biostatistics/kxu058).
- [Bobb et al. (2018), Statistical software for analyzing the health effects of multiple concurrent exposures via Bayesian kernel machine regression](https://doi.org/10.1186/s12940-018-0413-y).


- [Murray et al. (2010), Elliptical slice sampling](https://proceedings.mlr.press/v9/murray10a.html).
- [Stan count-distribution definitions](https://mc-stan.org/docs/functions-reference/unbounded_discrete_distributions.html).
- [INLA manual, piecewise-constant proportional hazards (§3.15)](https://inla.r-inla-download.org/r-inla.org/doc/inla-manual/inla-manual.pdf).
