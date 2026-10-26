# Verified results and open gaps

KernelMix implements the BKMR formulation introduced by [Bobb et al. (2015)](https://doi.org/10.1093/biostatistics/kxu058), including the probit model described in the [2018 software paper](https://doi.org/10.1186/s12940-018-0413-y). We check numerical calculations separately from interval coverage across repeated datasets and reproduction of the original studies. Agreement in one of these checks does not establish the others.

## Evidence status

| Check | Result and limit |
| --- | --- |
| Installed R bkmr 0.2.2 fixed-state comparisons | Selected Gaussian/probit means and covariances at fixed parameter values, kernel/prior calculations and inclusion summaries agree under documented conventions |
| Independent likelihood and conditional-posterior checks | Small examples with independently calculated answers agree within their test tolerances |
| [Observed ozone example](airquality.md) | A reproducible exploratory workflow on public environmental measurements |
| Repeated-dataset calibration | Open: systematic estimation error, interval coverage and exposure-selection performance need a separate study |
| Original-paper numerical reproduction | Blocked by missing original study assets and incomplete settings |

For the R comparisons, we hold constructed parameter values fixed and compare calculations at those values. We do not compare fitted chains. Markov chain Monte Carlo (MCMC) approximates the posterior by sampling, and different samplers can produce different chains even with equal seeds. [Numerical evidence](development.md#numerical-evidence) gives versions and comparison conventions.

## Original-paper targets

The [2015 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5963470/?report=xml) describes the following designs:

| Target | Recorded design | Unresolved input |
| --- | --- | --- |
| Main simulations | 300 datasets per scenario; $n=100$; three Bangladesh or 13 Boston exposure components; 10,000 iterations, last 5,000 retained | Original empirical exposure rows; complete response/noise, prior, grouping and fitting settings |
| Bangladesh application | 375 children; log cord-blood As/Mn/Pb; motor score and specified covariate adjustment | Original analysis rows, preprocessing and full fitting settings |
| Dog crossover application | 142 exposure records from 13 dogs; 13 elements; dog random intercepts | Original outcomes/exposures, dog IDs, exclusions, complete groups and settings |

The application chains use 25,000 warmup iterations followed by 25,000 iterations, retaining every fifth draw. Original seeds remain unknown. The inspected [author repository](https://github.com/jenfb/bkmr/tree/45413e338a316362d629f53bd2a917c4bf485c1e)
does not contain these data assets. Full supplementary settings were not recovered.

A simulated substitute would change the reproduction target. R `SimData` generates Gaussian exposures in its normal and “realistic” modes (rounded covariance for the latter), or uniform exposures in another mode. The original study resampled observed exposure rows. The NIEHS and binary examples linked from the 2018 paper are simulated, not observed-data applications.

Exact reproduction needs the missing inputs and settings. The ozone example
cannot replace the original studies.

## Platform evidence

| Environment | Evidence for the 0.3.0 runtime |
| --- | --- |
| Local Python 3.14, current dependencies | 484 tests passed from an installed wheel |
| Local Python 3.11, minimum dependencies | 484 tests passed from an installed wheel |
| Hosted Linux/Python 3.10 and 3.14; Windows/Python 3.13 | Not yet verified |

Hosted jobs have not executed. Python 3.10 and Windows runtime compatibility
remain unverified. Check the [Actions results](https://github.com/nilson01/kernelmix/actions)
for platform results; configuration alone does not establish compatibility.
