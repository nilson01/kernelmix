# API reference

Import the public API with `import kernelmix as km`. Arrays are NumPy arrays; pandas is not required. Examples that connect these functions are in the [tutorial](tutorial.md).

## Fitting

```python
km.fit(
    y, Z, X=None, *, family="gaussian", draws=1000, warmup=1000,
    chains=4, thin=1, seed=None, standardize=True, intercept=True,
    selection="none", groups=None, prior=None, r_fixed=None, missing="raise",
    exposure_names=None, covariate_names=None, initial=None, proposal_scales=None,
)
```

Returns `BKMRFit`.

| Input | Meaning |
| --- | --- |
| `y`, `Z`, `X` | Outcome vector, exposure matrix and optional adjustment matrix, with matching rows. Supply raw adjustment columns without an intercept when `intercept=True`. |
| `family` | `"gaussian"` for continuous outcomes or `"binomial"` for binary probit regression. Binary residual variance is fixed at one. |
| `draws`, `warmup`, `chains`, `thin` | Retained states per chain, discarded iterations, sequential chains and spacing between retained states. Total iterations per chain are `warmup + draws * thin`. At least two retained states are required. |
| `seed` | Nonnegative integer or `None`; uses local random generators without changing NumPy's global random state. |
| `standardize` | Center exposures and divide by retained-training sample standard deviations. Constant exposures are invalid when this is enabled. |
| `selection` | `"none"`, `"component"` or `"group"`. Group selection requires one `groups` label per exposure and permits at most one active exposure in each included group. |
| `prior` | `Prior`, a dictionary of its fields, or `None` for defaults. |
| `r_fixed` | Finite nonnegative vector fixing exposure relevance values; disallows variable selection. |
| `missing` | `"raise"` or joint complete-case `"drop"`. Infinity remains invalid. |
| Names | Optional unique names for exposure and raw adjustment columns. Otherwise use input DataFrame columns, or generated names. |
| `initial` | Shared dictionary of initial `beta`, `r`, `lambda`, `sigma2`, `delta`, `pi` and, for binary outcomes, `ystar`. Omitted entries receive defaults. |
| `proposal_scales` | Dictionary containing positive log-scale proposal standard deviations for `"r"` and/or `"lambda"`; both default to `0.3` and stay fixed throughout sampling. |

The adjustment design must have full column rank and fewer columns than retained observations. Binary adjustment designs with complete or quasi-complete separation are rejected under the flat coefficient prior. Fitting uses exact dense covariance matrices, with quadratic memory and cubic linear-algebra cost in the number of observations. Successful fitting does not establish convergence.

### Prior configuration

```python
km.Prior(
    r_distribution="invunif", r_mean=5.0, r_sd=5.0,
    r_lower=0.0, r_upper=100.0, lambda_mean=10.0, lambda_sd=10.0,
    sigma2_shape=0.001, sigma2_rate=0.001,
    inclusion_alpha=1.0, inclusion_beta=1.0,
)
```

`r_distribution` accepts `"gamma"`, `"unif"` and `"invunif"`. The inverse-uniform option places the uniform distribution on `1/r`, so its bounds apply to the reciprocal of an active relevance value. Excluded exposures have relevance zero. Gamma priors use mean and standard deviation; the residual variance uses inverse-gamma shape and rate. Inclusion probabilities use a beta prior. The residual variance prior is used for Gaussian outcomes.

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

Saved parameters include `beta`, `r`, `lambda`, `sigma2`, `delta` and `pi`; binary fits also retain `ystar`. `delta` records exposure inclusion. `beta` includes the added intercept when fitted. No-selection fits store `pi=1`. Diagnostics cannot establish model validity; R-hat requires multiple chains and short or constant sequences can yield undefined diagnostics.

## Prediction

```python
fit.predict_h(Znew=None, *, probability=0.95, max_draws=None, seed=None)
fit.predict(
    Znew=None, Xnew=None, *, kind="mean", scale="response",
    probability=0.95, max_draws=None, seed=None,
)
fit.posterior_moments_h(Znew=None, max_draws=None)
```

All new exposure values use original units and the training column order. Named input columns must match the stored names and order. Training standardization is reused. `Xnew` contains raw adjustment columns without the intercept; it is required for new profiles when adjustments were fitted. With no new data, retained training rows and their adjustments are used.

| Call | Quantity |
| --- | --- |
| `predict_h` | Joint draws of `h`, excluding intercept and adjustment terms. |
| `predict(..., kind="mean", scale="response")` | Gaussian outcome means or binary probabilities. |
| `predict(..., kind="observation", scale="response")` | Gaussian future outcomes including residual noise, or binary zero-or-one outcomes. |
| `predict(..., kind="mean", scale="latent")` | `h` plus the linear adjustment contribution, before the binary probit transformation. |
| `predict(..., kind="observation", scale="latent")` | Latent outcomes including Gaussian residual noise; binary residual variance is one. |
| `posterior_moments_h` | Dictionary with deterministic integrated `mean` and `covariance` for `h`, averaging conditional moments over saved states. |

`max_draws` selects at most that many evenly spaced saved states **per chain**, with a minimum of two. It does not refit. `seed` controls conditional prediction draws. A `Prediction` contains `draws` shaped `(chains, selected_draws, rows)`, `mean`, `sd`, `lower`, `upper`, joint `covariance`, `probability`, `scale` and `kind`. Its `to_records()` method returns one summary dictionary per row. Intervals are equal-tailed pointwise posterior intervals; observation intervals also include observation noise.

## Exposure contrasts and curves

```python
km.overall_effect(
    fit, quantiles=(0.25, 0.5, 0.75), reference=0.5, scale="latent",
    X=None, probability=0.95, max_draws=None, seed=None,
)
km.univariate_response(
    fit, variable=None, grid=None, points=50, fixed_quantile=0.5,
    center=True, scale="latent", X=None, probability=0.95,
    max_draws=None, seed=None,
)
km.bivariate_response(
    fit, variables=(0, 1), grids=None, points=20, fixed_quantile=0.5,
    center=True, scale="latent", X=None, probability=0.95,
    max_draws=None, seed=None,
)
km.single_exposure_effects(
    fit, quantiles=(0.25, 0.75), fixed_quantiles=(0.25, 0.5, 0.75),
    scale="latent", X=None, probability=0.95, max_draws=None, seed=None,
)
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

Latent summaries use `h`. Response summaries use Gaussian means or binary probabilities and hold `X` constant. `X` is one raw adjustment vector; when omitted it defaults to retained training covariate means. `X` is rejected for latent summaries. Binary probabilities are transformed before contrasts are calculated. All contrasts use joint draws to preserve covariance and exactly cancel identical profiles.

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
