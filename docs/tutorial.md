# An exposure-mixture analysis

KernelMix fits Bayesian kernel machine regression (BKMR) for continuous, binary, count and right-censored survival outcomes. The exposure-response function, called `h`, can include nonlinear associations and interactions. Adjustment covariates enter linearly.

This tutorial uses small, deterministic example data. Run its Python blocks in order after installing the source package as described in the [README](../README.md). The short chains make the examples quick to run. They are too short to support inference. For an analysis, choose longer runs, inspect mixing and effective sample sizes, and check sensitivity to prior choices.

## Define the data

Each row is an observation. Columns of `Z` are exposures; columns of `X` are adjustment covariates. Keep `Z` in its original units. The fit stores the training means and sample standard deviations and reuses them for prediction.

```python
import numpy as np
import kernelmix as km

t = np.linspace(-1.0, 1.0, 12)
Z = np.column_stack((1.5 + t, 2.0 + np.sin(3 * t), 10.0 + np.cos(5 * t)))
X = t[:, None]
y = 0.3 + np.sin(2 * t) + 0.2 * t**2
names = ["Exposure A", "Exposure B", "Exposure C"]
```

The default adds an intercept, so `X` should not include a constant column. Missing values raise an error. Use `missing="drop"` only when joint complete-case analysis is appropriate; that option drops a row if its outcome, exposure or adjustment value is missing.

## Fit and inspect a continuous outcome

Component selection allows each exposure to enter or leave the model. Its posterior inclusion probability (PIP) is the proportion of saved states in which that exposure is included.

```python
gaussian = km.fit(
    y, Z, X,
    family="gaussian", selection="component",
    exposure_names=names, covariate_names=["Adjustment"],
    draws=12, warmup=12, chains=2, seed=19,
)
print(gaussian.summary())
parameter_rows = gaussian.summary_table()
pip_rows = gaussian.inclusion_probabilities()
checks = gaussian.diagnostics()
```

The summary gives posterior means, standard deviations and pointwise credible intervals. Diagnostics include split R-hat, which compares chains, and effective sample sizes, which account for dependence between saved draws. These checks help detect sampling problems; passing them does not establish that the model fits the scientific question. PIPs depend on the selection model and priors and are not causal importance scores.

## Predict at exposure profiles

Supply new exposures in the training column order and their original units. New adjustment rows contain the raw `X` columns, without an intercept. They are required when adjustments were fitted.

```python
Znew = Z[:3].copy()
Xnew = X[:3].copy()
mean_prediction = gaussian.predict(Znew, Xnew, seed=20)
observation_prediction = gaussian.predict(
    Znew, Xnew, kind="observation", seed=20,
)
h_prediction = gaussian.predict_h(Znew, seed=20)
prediction_rows = mean_prediction.to_records()
```

The default predicts outcome means. Observation predictions add residual noise and describe new outcomes. `predict_h` returns only the exposure function, without the intercept or adjustment contribution. Prediction objects retain joint draws, pointwise intervals and covariance between prediction rows. Calling `predict()` without new data uses the retained training rows and their adjustments.

## Summarize exposure associations

Overall contrasts compare profiles at which every exposure is at a specified marginal quantile. Single-exposure contrasts change one exposure while fixing the others. These are conditional associations; a causal interpretation needs additional scientific assumptions.

```python
overall = km.overall_effect(gaussian, reference=0.5, seed=21)
single = km.single_exposure_effects(gaussian, seed=21)
interaction = km.interaction_effects(
    gaussian, variables=["Exposure A", "Exposure B"], seed=21,
)
curve = km.univariate_response(
    gaussian, variable="Exposure A", points=7, seed=21,
)
surface = km.bivariate_response(
    gaussian, variables=("Exposure A", "Exposure B"), points=4, seed=21,
)
overall_rows = overall.to_records()
```

By default, these summaries describe changes in `h`. Curves and surfaces hold other exposures at their medians and subtract a common all-median reference profile. A centered value of zero means no change from that reference. `center=False` returns the uncentered curve or surface. Interval limits are pointwise posterior quantiles, not a simultaneous band across the whole curve.

Each contrast uses joint prediction draws for the profiles being compared. This retains their covariance. A profile compared with itself has exactly zero contrast. Marginal quantile combinations can be poorly represented among observed profiles, especially with correlated exposures; examine data support when interpreting them.

## Fit a binary outcome

For a binary outcome, use `family="binomial"`. The probit link maps the latent mean to a probability with the standard normal cumulative distribution function. Outcomes must be zero or one.

```python
y_binary = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
binary = km.fit(
    y_binary, Z, X,
    family="binomial", exposure_names=names,
    covariate_names=["Adjustment"],
    draws=12, warmup=12, chains=2, seed=22,
)
probability_prediction = binary.predict(Znew, Xnew, seed=23)
probability_contrast = km.overall_effect(
    binary, scale="response", X=[0.0], seed=23,
)
```

Binary mean predictions are probabilities. With `kind="observation"`, predictions are zero-or-one outcomes. A response-scale contrast subtracts predicted probabilities after applying the probit transformation to each joint draw. It is not an odds ratio. In exposure summaries, `X` is one raw adjustment vector held constant at every profile. If it is omitted, response-scale summaries use the retained training covariate means. Latent-scale summaries use `h` alone and do not accept `X`.

## Fit counts and include observation time

Use `"poisson"` for conditional Poisson counts or `"negative_binomial"` for
additional conditional variation. An offset is the log observation time or
another known exposure amount. It enters with coefficient one.

```python
counts = np.array([0, 1, 0, 2, 1, 3, 1, 2, 4, 3, 5, 2])
observation_time = np.array([1, 1, 2, 1, 2, 1, 1, 3, 1, 2, 1, 1])
count_fit = km.fit(
    counts, Z, family="negative_binomial", offset=np.log(observation_time),
    draws=12, warmup=12, chains=2, seed=24,
)
count_mean = count_fit.predict(Z[:3], offset=np.log(observation_time[:3]), seed=25)
count_observation = count_fit.predict(
    Z[:3], offset=np.log(observation_time[:3]), kind="observation", seed=25,
)
```

`count_mean` describes expected counts for the supplied observation times.
Observation predictions are nonnegative integer counts. Set `family="poisson"`
to use the Poisson model. The negative-binomial `dispersion` parameter controls
extra variation; it is separate from variation due to exposures or clusters.
Response-scale exposure summaries use an offset of zero, so they compare counts
per unit observation exposure. Their latent scale describes changes in `h`,
which enter the log mean. These models use proper coefficient priors; see the
model guide for their defaults.

## Fit right-censored survival times

Each observation supplies a positive observed time and an event indicator:
`1` means the event occurred; `0` means follow-up ended first. The model uses
proportional hazards with a piecewise-constant baseline hazard.

```python
observed_time = np.array([1.0, 1.4, 2.0, 2.6, 3.0, 3.8, 4.0, 4.7, 5.2, 6.0, 6.4, 7.0])
event = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])
survival_fit = km.fit(
    observed_time, Z, family="survival", event=event,
    baseline_breaks=[2.0, 4.0], intercept=False,
    draws=12, warmup=12, chains=2, seed=26,
)
survival = survival_fit.predict_survival(
    Z[:2], times=np.linspace(0, 7, 30), seed=27,
)
survival_ax = km.plot_survival(survival)
log_hr = km.overall_effect(survival_fit, quantiles=[0.75], reference=0.25, seed=27)
hr_draws = np.exp(log_hr.draws)
hr_interval = np.quantile(hr_draws, [0.025, 0.975], axis=(0, 1))
```

The baseline intervals are $(0,2]$, $(2,4]$ and $(4,\infty)$ in this example.
An event exactly at a cut belongs to the interval on its left. The model assumes
independent right censoring and proportional hazards. It does not support delayed
entry, competing risks or time-varying covariates. Beyond observed follow-up,
predictions extend the final constant baseline hazard. Such extrapolation needs
scientific justification.

`predict_survival` returns survival probabilities, or hazards with
`kind="hazard"`. Ordinary `predict` returns relative hazards for this family.
Response-scale exposure contrasts therefore subtract relative hazards; they do
not subtract survival probabilities. For hazard ratios, exponentiate each
latent `h` contrast before summarizing it. The baseline and a fitted intercept
share a scale, so this example omits the intercept.

![Count response and survival probabilities. (a) Negative-binomial mean counts as $z_1$ varies, with offset zero. (b) Survival probabilities at the first exposure's 25th and 75th percentiles. Other exposures are fixed at their medians and $u=0$. Shading shows pointwise 95% credible intervals.](figures/count-survival.png)

The figure uses the data above, 200 warmup and 200 saved draws per chain,
two chains, and the same seeds.
Predictions use 100 draws per chain. These short fits have mixing warnings and illustrate the outputs;
they do not support inference or establish interval calibration.
[Figure values](figures/count-survival.csv).

## Account for clustered observations

`cluster` labels observations that share a Gaussian random intercept. It is
separate from `groups`, which groups exposures for selection. All five families
accept `cluster`.

```python
cluster = ["A"] * 3 + ["B"] * 3 + ["C"] * 3 + ["D"] * 3
clustered_fit = km.fit(
    counts, Z, family="poisson", cluster=cluster,
    draws=12, warmup=12, chains=2, seed=28,
)
intercepts = clustered_fit.random_intercepts(seed=29)
cluster_prediction = clustered_fit.predict(
    Z[:3], cluster_new=["A", "new", "new"], seed=29,
)
```

Known prediction labels use their posterior random intercept. Repeated unseen
labels share one new intercept per posterior draw. Omitting `cluster_new` fixes
the prediction random intercept to zero; this is not an average over clusters.
The exposure summaries also use a zero random intercept. This model captures
shared intercept variation, not arbitrary within-cluster time dependence.

## Plot the results

Install the optional Matplotlib dependency to use these functions. Each returns an axes object. The caller controls display and export.

```python
trace_ax = km.plot_trace(gaussian)
pip_ax = km.plot_pips(gaussian)
curve_ax = km.plot_response(curve)
overall_ax = km.plot_overall(overall)
surface_ax = km.plot_surface(surface)

# Optional: display the figures in an interactive session.
# import matplotlib.pyplot as plt
# plt.show()
```

Curve and overall plots include pointwise posterior intervals. The surface plot displays the posterior mean only; its result object also retains lower and upper limits. Plot exposures separately when their units differ. The trace excludes warmup draws.

## Save and load a fit

A fit includes its retained training data as well as saved states. Store and share it with the same care as the input data. The archive uses a non-pickle format.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

with TemporaryDirectory() as directory:
    path = Path(directory) / "mixture-fit.npz"
    gaussian.save(path)
    restored = km.BKMRFit.load(path)
    restored_prediction = restored.predict(Znew, Xnew, seed=20)
```

See the [API reference](api.md) for argument defaults and result fields. The method is described by [Bobb et al. (2015)](https://doi.org/10.1093/biostatistics/kxu058); [Bobb et al. (2018)](https://doi.org/10.1186/s12940-018-0413-y) describes exposure summaries and the probit extension. KernelMix's pairwise interaction summary and median-reference curve centering are specified in the API reference.
