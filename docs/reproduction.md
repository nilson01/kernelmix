# Numerical evidence and limits

KernelMix follows the Gaussian BKMR model of [Bobb et al. (2015)](https://doi.org/10.1093/biostatistics/kxu058) and the probit extension described by [Bobb et al. (2018)](https://doi.org/10.1186/s12940-018-0413-y). Numerical agreement, repeated-dataset performance and original-study reproduction are separate checks.

## Fitted R comparisons

We fit the same 12 observations and two exposures in Python and installed R `bkmr` 0.2.2. Inputs, sample-standardization, intercepts and priors agree. Exposure selection is disabled. Relevance and variance-ratio priors are Gamma(shape 2, rate 2); Gaussian residual variance has an inverse-gamma(shape 3, scale 1) prior. Adjustment coefficients have flat priors, and probit residual variance is one. Both implementations use four chains and 3,000 warmup transitions per chain, followed by 120,000 saved draws in R and 60,000 in Python. Different seeds and proposals are allowed because the target is the posterior distribution, not identical chains.

We compare means and the 10th, 50th and 90th percentiles of coefficients, relevance parameters, variance parameters, three values of $h$ and one joint exposure contrast. Monte Carlo standard error (MCSE) measures uncertainty from finite posterior sampling. The pooled posterior standard deviation is the square root of the average of the two sample variances. The combined MCSE is the square root of the sum of the two squared MCSEs.

| Model | Mean/percentile comparisons | Result |
| --- | ---: | --- |
| Gaussian | 40 | Pass |
| Gaussian with random intercepts | 44 | Pass |
| Probit | 36 | Pass |
| Probit with random intercepts | 40 | Pass |

Each quantity must have $\widehat R\le1.01$ and bulk/tail effective sample sizes at least 400. Each summary's MCSE must be at most 2% of the pooled posterior standard deviation. Differences must be within six combined MCSEs; the absolute difference plus three combined MCSEs must also be within 15% of that standard deviation. These are numerical screening rules, not a calibrated equivalence test or a proof of convergence. The first, shorter attempt was inconclusive because some R summaries lacked precision; its tolerances were not widened.

Constructed-state comparisons separately check conditional means, covariances, inclusion probabilities and prior calculations. [Reproduction commands](development.md#r-reference-checks) document the required R installation and conventions. R is never required for ordinary fitting or prediction.

## Independent extension checks

R `bkmr` has no matching count or survival models. Here, NB2 denotes the negative-binomial model and GP denotes the Gaussian process. We instead compare three fixed fits with independent numerical integration. Two chains each retain 100,000 draws after 1,000 warmup transitions. All 12 checked posterior moments and four selected covariances pass their precision and error limits.

| Target | Unknown quantities sampled together | Fixed structure |
| --- | --- | --- |
| NB2 counts | Constant $h$, GP amplitude and dispersion | Four observations, fixed relevance zero |
| Right-censored survival | Constant $h$, GP amplitude and baseline rate | Four subjects, one baseline interval, no delayed entry |
| Structured Gaussian | Constant $h$, GP amplitude, random intercept/slope and residual variance | Six observations; fixed random-coefficient covariance and AR1 correlation; one nonzero Fourier feature |

The covariance checks detect dependence between selected sampled quantities. A deliberately broken pairing of NB2 draws is rejected. These small constant-function cases do not establish general posterior agreement, calibration of intervals, or validation of every structured option. Separate likelihood and conditional-distribution tests cover additional inputs, censoring terms, kernels and prediction rules. The [numerical results](results/posterior-checks.json) record the checked summaries; the [developer runner](development.md#independent-posterior-targets) reproduces these targets.

## Original studies

Exact reproduction remains blocked by missing original data and sufficient fitting settings.

| Study in the [2015 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5963470/?report=xml) | Design | Missing inputs |
| --- | --- | --- |
| Simulations | 300 datasets per scenario; $n=100$; three Bangladesh or 13 Boston exposure components; 10,000 iterations, last 5,000 retained | Empirical exposure rows; complete response/noise, prior, grouping and fitting settings |
| Bangladesh | 375 children; log cord-blood As/Mn/Pb; motor score with covariate adjustment | Analysis rows, preprocessing and full fitting settings |
| Dog crossover | 142 exposure records from 13 dogs; 13 elements; random intercepts | Outcomes, exposures, dog IDs, exclusions, groups and settings |

The applications use 25,000 warmup iterations and 25,000 further iterations, retaining every fifth draw. Original seeds remain unknown. The inspected [author repository](https://github.com/jenfb/bkmr/tree/45413e338a316362d629f53bd2a917c4bf485c1e) lacks these data; full supplementary settings were not recovered. The simulations resampled observed exposure rows, so generated substitutes would change the target. The [observed ozone example](airquality.md) demonstrates a separate exploratory analysis. Repeated-dataset bias, interval coverage and selection performance remain open.

## Platform evidence

| Environment | Installed-wheel result |
| --- | --- |
| macOS, Python 3.14, current dependencies | 512 tests passed |
| macOS, Python 3.10, NumPy 1.24.0 / SciPy 1.10.0 / Matplotlib 3.7.0 | 512 tests passed |
| Linux, Python 3.10, minimum dependencies | 512 tests passed |
| Linux, Python 3.14, current dependencies | 512 tests passed |
| Windows, Python 3.13, current dependencies | 512 tests passed |

The [hosted checks](https://github.com/nilson01/kernelmix/actions/runs/40263682700) test installed wheels outside the source checkout.
