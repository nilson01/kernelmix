# Checks and development

## Run checks

```sh
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests tools
python tools/check_public.py --history
python -m build
python -m twine check dist/*
```

The continuous integration (CI) configuration builds an installable wheel and tests it outside the source checkout on Linux with Python
3.10 and 3.14, and Windows with Python 3.13. The minimum-dependency job pins
NumPy 1.24.0, SciPy 1.10.0 and Matplotlib 3.7.0. See the repository's Actions
results for the outcome on each commit; configuration alone does not establish
that a platform passed.

## Numerical evidence

| Check | Purpose and limit |
| --- | --- |
| Independent kernel and conditional-normal calculations | Check covariance, prediction and joint contrasts |
| Installed R `bkmr` 0.2.2 | Compare constructed parameter states for Gaussian and probit models: means/covariances with those states fixed, with/without clusters, posterior inclusion probabilities (PIPs) and prior densities |
| Integrated Gaussian posterior | Compare sampled amplitude percentiles with numerical integration over one variable, after accounting analytically for residual variance |
| Probit orthant posterior | Compare sampled amplitude percentiles with an analytic likelihood for the signs of two underlying normal outcomes |
| Uninformative selection case | Check the beta-binomial count distribution when exposure selection cannot change the kernel |
| Count and survival likelihoods | Compare Poisson/NB2 mass with SciPy and survival density/censoring terms with independent arithmetic and interval splitting |
| Elliptical slice posterior | Check a normal posterior with a known exact answer |
| Poisson scalar posterior | Compare the posterior of a constant exposure function with numerical integration over one variable |
| NB2 dispersion / survival baseline | Check one-parameter/Gamma posterior distributions while other underlying model variables are held fixed; these do not check the full joint posterior |
| Seeded compatibility with 0.1.0 | Saved arrays agree bitwise with 0.1.0 in six Gaussian/probit selection cases |
| Diagnostic examples | Detect shifted chains, autocorrelation and constant traces |
| Structured likelihoods | Independently check kernel and frequency calculations, joint Gaussian densities, residual predictions, censoring, cumulative incidence functions (CIFs) and predictor paths |
| Structured posterior targets | Constant-GP Gaussian, Poisson and delayed-entry survival targets against independent integration; amplitude held fixed for these checks |
| Finite-basis storage | Instrumented PP/RFF operations avoid training-size square matrices; this is not a large-dataset benchmark |
| Package and input checks | Exercise saved objects, preprocessing, errors, distribution contents and installed examples |

We compare sampled posterior summaries with independently calculated answers, using fixed seeds and tolerances that allow for sampling error based on effective sample size. These tests check small cases, not repeated-dataset performance, general interval coverage, original-study reproduction or full R API equivalence.

### Reproduce the R fixtures

R is needed only to regenerate fixtures. Use an isolated R library with
`bkmr` **0.2.2**, `jsonlite` and the dependencies declared by that version of
`bkmr` (including `fields`, `truncnorm` and `tmvtnorm`). Its CRAN archive has
SHA256 `7ed29f74c4c78f3b8c663eac5ec417b4ea9bc586d789e8049e3ddc51541af9af`.
Set `R_LIBS` to that library, then run:

```sh
Rscript tools/export_reference.R tests/fixtures/r_reference.json
Rscript tools/export_cluster_reference.R tests/fixtures/r_cluster_reference.json
python -m pytest tests/test_reference.py tests/test_clusters.py
```

The exporters record R/package versions and call installed `ComputePostmeanHnew`, `ExtractPIPs` and internal kernel/prior/covariance helpers. Fixtures use **constructed parameter states, not fitted R chains**, with explicit preprocessing and intercept columns.

R's public summary does not accept arbitrary new-cluster labels, so Python $h+u$ prediction is checked by independent block-normal calculations. Count and survival have no matching R `bkmr` family; their checks are separate.

The elementwise tolerance is `rtol=2e-11, atol=2e-12`. R's exact summary uses
sample covariance between conditional means (denominator $S-1$); Python's
finite empirical mixture uses denominator $S$, where $S$ is the retained state
count. The comparison adjusts this convention explicitly. Sampling proposals
and random-number streams differ; equal seeds do not produce matching chains.

## Observed data and original studies

The [ozone example](airquality.md) supplies data, scripts, tables and residual
checks. The [evidence summary](reproduction.md) describes the analysis limits, missing inputs for the original studies and unresolved platform checks.

## Build the manual

The editable sources are the README and Markdown pages in `docs`.
Install the documentation extra and a local `pdflatex` distribution:

```sh
python -m pip install ".[docs]"
python tools/build_manual.py
```

Pandoc and LaTeX build the PDF from these sources; intermediate TeX is temporary. The README's illustrative curve values are in `docs/figures/response.csv`. Their pointwise intervals do not establish repeated-dataset coverage.