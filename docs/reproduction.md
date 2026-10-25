# Verified results and open gaps

KernelMix implements the BKMR exposure model described by [Bobb et al. (2015)](https://doi.org/10.1093/biostatistics/kxu058), with the probit and software extension described by [Bobb et al. (2018)](https://doi.org/10.1186/s12940-018-0413-y). Software agreement, interval calibration and original-study reproduction are
separate checks.

## Evidence status

| Check | Result and limit |
| --- | --- |
| Installed R bkmr 0.2.2 fixed-state comparisons | Selected Gaussian/probit conditional moments, kernel/prior calculations and inclusion summaries agree under documented conventions |
| Independent likelihood and conditional-posterior checks | Specified small mathematical targets are computed correctly within their test tolerances |
| [Observed ozone example](airquality.md) | A reproducible exploratory workflow on public environmental measurements |
| Repeated-dataset calibration | Open: bias, coverage and selection performance need a separate study |
| Original-paper numerical reproduction | Blocked by missing original study assets and incomplete settings |

The installed-R fixtures contain constructed states, not matched fitted MCMC chains. Equal seeds do not produce matching chains across different samplers. See [numerical evidence](development.md#numerical-evidence) for exact software versions and comparison limits.

## Original-paper targets

The [2015 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5963470/?report=xml) describes the following designs:

| Target | Recorded design | Unresolved input |
| --- | --- | --- |
| Main simulations | 300 datasets per scenario; $n=100$; three Bangladesh or 13 Boston exposure components; 10,000 iterations, last 5,000 retained | Original empirical exposure rows; complete response/noise, prior, grouping and fitting settings |
| Bangladesh application | 375 children; log cord-blood As/Mn/Pb; motor score and specified covariate adjustment | Original analysis rows, preprocessing and full fitting settings |
| Dog crossover application | 142 exposure records from 13 dogs; 13 elements; dog random intercepts | Original outcomes/exposures, dog IDs, exclusions, complete groups and settings |

The application chains use 25,000 warmup iterations followed by 25,000 iterations, retaining every fifth draw. Original seeds remain unknown. The inspected [author repository](https://github.com/jenfb/bkmr/tree/45413e338a316362d629f53bd2a917c4bf485c1e)
does not contain these data assets. Full supplementary settings were not recovered.

The default normal and “realistic” modes of R `SimData` generate Gaussian exposures; the latter uses a rounded covariance. The helper also offers uniform exposures. The original study instead resampled empirical exposure rows. Substituting that generator would change the experiment and must not be labeled an exact reproduction. The NIEHS and binary examples linked from the 2018 software paper are simulated; they are not observed-data applications.

Exact reproduction needs the missing inputs and settings. The ozone example
cannot replace the original studies.

## Platform evidence

| Environment | Evidence for the 0.3.0 runtime |
| --- | --- |
| Local Python 3.14, current dependencies | 484 tests passed from an installed wheel |
| Local Python 3.11, minimum dependencies | 484 tests passed from an installed wheel |
| Hosted Linux/Python 3.10 and 3.14; Windows/Python 3.13 | Blocked before runner startup by account billing/spending limits |

The [main-branch check](https://github.com/nilson01/kernelmix/actions/runs/40243787433)
records the startup blocker. No hosted platform passed. Python 3.10 and Windows
runtime compatibility remain unverified; configuration is not test evidence.
