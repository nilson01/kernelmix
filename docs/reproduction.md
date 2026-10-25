# Reference studies and reproducibility

KernelMix implements the BKMR exposure model described by [Bobb et al. (2015)](https://doi.org/10.1093/biostatistics/kxu058), with the probit and software extension described by [Bobb et al. (2018)](https://doi.org/10.1186/s12940-018-0413-y). Numerical agreement, statistical calibration and reproduction of a published study require different evidence.

## Evidence available

| Evidence | What it establishes |
| --- | --- |
| Installed R bkmr 0.2.2 fixed-state comparisons | Selected Gaussian/probit conditional moments, kernel/prior calculations and inclusion summaries agree under documented conventions |
| Independent likelihood and conditional-posterior checks | Specified small mathematical targets are computed correctly within their test tolerances |
| [Observed ozone example](airquality.md) | A reproducible exploratory workflow on public environmental measurements |
| Repeated-dataset calibration | Not established by the checks above |
| Original-paper numerical reproduction | Blocked by missing original study assets and incomplete settings |

The installed-R fixtures contain constructed states, not matched fitted MCMC chains. Different samplers and seeds do not imply identical sampling paths. See [numerical evidence](development.md#numerical-evidence) for exact software versions and comparison limits.

## Original-paper targets

The [2015 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5963470/?report=xml) describes the following designs:

| Target | Recorded design | Unresolved input |
| --- | --- | --- |
| Main simulations | 300 datasets per scenario; $n=100$; three Bangladesh or 13 Boston exposure components; 10,000 iterations, last 5,000 retained | Original empirical exposure rows; complete response/noise, prior, grouping and fitting settings |
| Bangladesh application | 375 children; log cord-blood As/Mn/Pb; motor score and specified covariate adjustment | Original analysis rows, preprocessing and full fitting settings |
| Dog crossover application | 142 exposure records from 13 dogs; 13 elements; dog random intercepts | Original outcomes/exposures, dog IDs, exclusions, complete groups and settings |

The application chains use 25,000 warmup iterations followed by 25,000 iterations, retaining every fifth draw. Original seeds remain unknown. The accessible author repository at revision `45413e338a316362d629f53bd2a917c4bf485c1e` does not contain these three data assets. Full supplementary settings were not recovered. Published figures and article prose are not redistributed.

The default normal and “realistic” modes of R `SimData` generate Gaussian exposures; the latter uses a rounded covariance. The helper also offers uniform exposures. The original study instead resampled empirical exposure rows. Substituting that generator would change the experiment and must not be labeled an exact reproduction. The NIEHS and binary examples linked from the 2018 software paper are simulated; they are not observed-data applications.

Exact reproduction remains open until the original inputs and sufficient settings are lawfully available. The ozone example is a separate analysis and does not close that gap. Broad repeated-dataset studies also require a separately reviewed design and suitable compute resources.

## Platform evidence

The installed 0.3.0 runtime passed 484 tests in the recorded local Python 3.14/current and Python 3.11/minimum-dependency environments. The configured hosted matrix includes Linux/Python 3.10 with minimum dependencies, Linux/Python 3.14 and Windows/Python 3.13. The latest [requested run](https://github.com/nilson01/kernelmix/actions/runs/40239319446) was blocked before any runner steps by the account billing/spending-limit restriction. No hosted platform passed and no package compatibility defect was observed. Python 3.10 and Windows runtime behavior remain unverified.
