# An exposure-mixture analysis

KernelMix fits Bayesian kernel machine regression (BKMR) for a continuous outcome or a binary outcome with a probit link. The exposure-response function, called `h`, can include nonlinear associations and interactions. Adjustment covariates enter linearly.

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
