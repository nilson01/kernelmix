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

CI builds a wheel and tests it outside the source checkout on Linux with Python
3.10 and 3.14, and Windows with Python 3.13. The minimum-dependency job pins
NumPy 1.24.0, SciPy 1.10.0 and Matplotlib 3.7.0. See the repository's Actions
results for the outcome on each commit; configuration alone does not establish
that a platform passed.

## Numerical evidence

| Check | Purpose and limit |
| --- | --- |
| Independent kernel and conditional-normal calculations | Check covariance, prediction and joint contrasts |
| Installed R `bkmr` 0.2.2 | Compare constructed saved states for Gaussian and probit models, exact conditional moments, PIPs and prior densities |
| Integrated Gaussian posterior | Compare sampled amplitude quantiles with one-dimensional quadrature after integrating out residual variance |
| Probit orthant posterior | Compare sampled amplitude quantiles with the analytic two-observation normal orthant likelihood |
| Uninformative selection case | Check the beta-binomial count distribution when exposure selection cannot change the kernel |
| Diagnostic examples | Detect shifted chains, autocorrelation and constant traces |
| Package and input checks | Exercise saved objects, preprocessing, errors, distribution contents and installed examples |

The posterior-target tests use fixed seeds and Monte Carlo tolerances based on
effective sample size. They are small correctness checks, not a repeated-dataset
performance study. No general coverage, paper-study reproduction or complete R
API equivalence is claimed.

### Reproduce the R fixtures

R is needed only to regenerate fixtures. Use an isolated R library with
`bkmr` **0.2.2**, `jsonlite` and the dependencies declared by that version of
`bkmr` (including `fields`, `truncnorm` and `tmvtnorm`). Its CRAN archive has
SHA256 `7ed29f74c4c78f3b8c663eac5ec417b4ea9bc586d789e8049e3ddc51541af9af`.
Set `R_LIBS` to that library, then run:

```sh
Rscript tools/export_reference.R tests/fixtures/r_reference.json
python -m pytest tests/test_reference.py
```

The exporter verifies the installed version and records R/package versions.
It calls installed `ComputePostmeanHnew` and `ExtractPIPs`, plus internal kernel
and prior helpers. The fixtures contain **constructed states, not fitted R MCMC
chains**. Preprocessing and intercept columns are supplied explicitly.

The elementwise tolerance is `rtol=2e-11, atol=2e-12`. R's exact summary uses
sample covariance between conditional means (denominator $S-1$); Python's
finite empirical mixture uses denominator $S$, where $S$ is the retained state
count. The comparison adjusts this convention explicitly. Sampling proposals
and random-number streams differ; equal seeds do not produce matching chains.

## Build the manual

The editable sources are the README and Markdown pages in `docs`.
Install the documentation extra and a local `pdflatex` distribution:

```sh
python -m pip install ".[docs]"
python tools/build_manual.py
```

Pandoc and LaTeX build the PDF from Markdown. Intermediate TeX stays in a temporary
directory. The small README example generates the response figure; its plotted
values are also available in `docs/figures/response.csv`. The figure shows
pointwise credible limits from short illustrative chains, not a performance
benchmark.
