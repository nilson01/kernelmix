# KernelMix

KernelMix fits Bayesian kernel machine regression (BKMR) for exposure mixtures in Python.
It models how an outcome varies with several exposures together, allowing curved associations and interactions between exposures. Other measured variables enter as linear adjustment terms.

To handle different outcome types, the package combines this exposure function with continuous, binary, count or survival models. NumPy and SciPy handle fitting and prediction; R is not required.

| Analysis | Functions |
| --- | --- |
| Fit outcomes | `fit`, `Prior`: Gaussian, probit, Poisson, negative binomial and piecewise-exponential survival |
| Cluster observations | Random intercepts/slopes; Gaussian AR1, exponential or supplied residual correlations |
| Model unequal residual variance | Gaussian variance regression and residual-variance prediction |
| Choose how to represent the exposure function | Gaussian, Matérn 3/2 or rational-quadratic kernels; dense, predictive-process or random-Fourier bases |
| Model survival records | Delayed entry, competing causes, known predictor histories and left/interval censoring |
| Select exposures or exposure groups | `selection="component"` or `"group"`, posterior inclusion probabilities |
| Inspect results | `summary`, `diagnostics`, `posterior_samples` |
| Predict | `predict_h`, `predict`, `predict_survival`, `posterior_moments_h` |
| Describe exposure associations | Overall, single-exposure and interaction contrasts; response curves and surfaces |
| Plot | Chains, inclusion probabilities, exposure curves, survival curves, contrasts and surfaces |

## Install

With access to this repository:

```sh
python -m pip install "kernelmix[plots] @ git+https://github.com/nilson01/kernelmix.git@main"
```

Or, from a downloaded source folder:

```sh
python -m pip install ".[plots]"
```

Python 3.10 or later is required. Omit `[plots]` if plots are not needed.

## Example

```python
import numpy as np
from kernelmix import fit, univariate_response, plot_response

rng = np.random.default_rng(17)
Z = rng.normal(size=(24, 2))
y = np.sin(Z[:, 0]) + 0.3 * Z[:, 1] + rng.normal(0, 0.5, 24)
result = fit(y, Z, draws=200, warmup=200, chains=2, seed=17)
print(result.summary())
curve = univariate_response(result, variable=0, points=25, seed=17)
ax = plot_response(curve)
```

Example output (first rows):

```text
BKMRFit(family='gaussian', n=24, exposures=2, chains=2, draws=200)
95% pointwise credible intervals
Parameter             Mean         SD      Lower      Upper
lambda               16.93      9.047      4.145      38.61
sigma2              0.2159    0.08346     0.1142     0.4188
```

These short chains illustrate the output and trigger mixing warnings. For an analysis, use longer runs and inspect agreement between chains and effective sample sizes, which account for dependence between draws.

![Exposure response curve with pointwise 95% credible intervals](docs/figures/response.png)

The curve varies the first exposure with the second at its median, relative to the all-median exposure profile. It describes a fitted association; a causal interpretation requires additional assumptions.

## Documentation

| Need | Guide |
| --- | --- |
| Run an analysis | [Tutorial](docs/tutorial.md), [observed ozone example](docs/airquality.md) |
| Choose a model | [Equations and assumptions](docs/methods.md), [structured models](docs/structured.md) |
| Find a function | [API reference](docs/api.md) |
| Check the evidence | [Verified results and open gaps](docs/reproduction.md), [development checks](docs/development.md) |
| Read offline | [Reference manual](docs/manual.pdf) |

For model choice, the guides explain how kernels, cluster effects and outcome distributions fit together. Dense Gaussian process calculations limit sample size; finite bases reduce computation but approximate the model and can change uncertainty. Group selection permits at most one active exposure per group. Survival uses a piecewise-exponential likelihood. Priors differ between default and structured fits, and favorable diagnostics do not establish convergence or valid assumptions.

## References and license

The method is due to [Bobb et al. (2015)](https://doi.org/10.1093/biostatistics/kxu058).
The software and binary-outcome extension are described by
[Bobb et al. (2018)](https://doi.org/10.1186/s12940-018-0413-y).
KernelMix is a separate Python implementation. Selected Gaussian/probit calculations, including random intercepts, are checked
at fixed parameter values against R [`bkmr`](https://cran.r-project.org/package=bkmr) 0.2.2.
See [NOTICE](NOTICE) for attribution and [LICENSE](LICENSE) for GPL-2.0-only terms.

Count and survival models use additional standard likelihoods documented in the
[model guide](docs/methods.md).
