# KernelMix

KernelMix fits Bayesian kernel machine regression (BKMR) for exposure mixtures in Python.
It models how an outcome varies with several exposures together, allowing curved associations and interactions between exposures. Other measured variables enter as linear adjustment terms.

To handle different outcome types, the package combines this exposure function with continuous, binary, count or survival models. NumPy and SciPy handle fitting and prediction; R is not required.

| Task | Start here |
| --- | --- |
| Choose an outcome model and priors | [Models and assumptions](https://nilson01.github.io/kernelmix/methods/) |
| Fit, inspect, predict and compare exposures | [Analysis tutorial](https://nilson01.github.io/kernelmix/tutorial/) |
| Model clusters, kernels or detailed survival records | [Structured models](https://nilson01.github.io/kernelmix/structured/) |
| Find arguments and result fields | [API reference](https://nilson01.github.io/kernelmix/reference/) |
| Check numerical evidence and limitations | [Evidence](https://nilson01.github.io/kernelmix/reproduction/) |

## Install

```sh
python -m pip install kernelmix
```

For plots, use `python -m pip install "kernelmix[plots]"`.

Or install the released source directly from GitHub:

```sh
python -m pip install "kernelmix[plots] @ git+https://github.com/nilson01/kernelmix.git@v0.3.1"
```

Python 3.10 or later is required.

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

![Exposure response curve with pointwise 95% credible intervals](https://nilson01.github.io/kernelmix/figures/response.png)

The curve varies the first exposure with the second at its median, relative to the all-median exposure profile. It describes a fitted association; a causal interpretation requires additional assumptions.

## Documentation

Browse the [searchable documentation](https://nilson01.github.io/kernelmix/).

The [common questions](https://nilson01.github.io/kernelmix/faq/) explain prediction scales, missing values, selection and diagnostics. The [observed ozone example](https://nilson01.github.io/kernelmix/airquality/) includes source data and saved results. Download the [reference manual](https://nilson01.github.io/kernelmix/manual.pdf) for offline reading. Contributors can find [development checks](https://nilson01.github.io/kernelmix/development/) separately.

## References and license

[Bobb et al. (2015)](https://doi.org/10.1093/biostatistics/kxu058) introduced BKMR for exposure-mixture analysis. [Bobb et al. (2018)](https://doi.org/10.1186/s12940-018-0413-y) described the R software and its binary-outcome extension. KernelMix implements these methods separately in Python and adds count and survival likelihoods, defined in the [model guide](https://nilson01.github.io/kernelmix/methods/).

We compare selected Gaussian and probit posterior summaries and conditional calculations, including random intercepts, with installed R [`bkmr`](https://cran.r-project.org/package=bkmr) 0.2.2. The [evidence summary](https://nilson01.github.io/kernelmix/reproduction/) records results and limits. See [NOTICE](https://github.com/nilson01/kernelmix/blob/main/NOTICE) for attribution and [LICENSE](https://github.com/nilson01/kernelmix/blob/main/LICENSE) for GPL-2.0-only terms.
