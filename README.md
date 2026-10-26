# KernelMix

Bayesian kernel machine regression (BKMR) for exposure mixtures in Python.
KernelMix estimates how an outcome varies with several exposures together.
The exposure–response function can describe curved relationships and interactions: the association with one exposure can depend on another. Other measured variables, called adjustment covariates, enter as linear terms. It supports continuous, binary, count and survival outcomes.
NumPy and SciPy handle fitting and prediction; R is not required.

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

These short sampling runs, called chains, trigger mixing warnings: they may not explore the fitted distribution well enough. For an analysis, use longer runs and inspect agreement between chains and effective sample size, which accounts for dependence between saved draws.

![Exposure response curve with pointwise 95% credible intervals](docs/figures/response.png)

The curve changes the first exposure while holding the second at its sample median.
It shows the difference from the fitted exposure function when both exposures are at their medians. It shows an association, not an identified causal effect.

## Documentation

| Need | Guide |
| --- | --- |
| Run an analysis | [Tutorial](docs/tutorial.md), [observed ozone example](docs/airquality.md) |
| Choose a model | [Equations and assumptions](docs/methods.md), [structured models](docs/structured.md) |
| Find a function | [API reference](docs/api.md) |
| Check the evidence | [Verified results and open gaps](docs/reproduction.md), [development checks](docs/development.md) |
| Read offline | [Reference manual](docs/manual.pdf) |

The full Gaussian process (GP) model describes how exposure-function values vary together. Its dense matrix calculations limit practical sample size. Finite GP bases represent the function with fewer terms; they approximate the full model and can change uncertainty estimates. Group selection permits at most one active
exposure per group. Survival uses a piecewise-exponential likelihood.
Priors differ between default and structured fits; consult the model guides.
Diagnostics cannot prove convergence or validate assumptions.

## References and license

The method is due to [Bobb et al. (2015)](https://doi.org/10.1093/biostatistics/kxu058).
The software and binary-outcome extension are described by
[Bobb et al. (2018)](https://doi.org/10.1186/s12940-018-0413-y).
KernelMix is a separate Python implementation. Selected Gaussian/probit calculations, including random intercepts, are checked
at fixed parameter values against R [`bkmr`](https://cran.r-project.org/package=bkmr) 0.2.2.
See [NOTICE](NOTICE) for attribution and [LICENSE](LICENSE) for GPL-2.0-only terms.

Count and survival models use additional standard likelihoods documented in the
[model guide](docs/methods.md).
