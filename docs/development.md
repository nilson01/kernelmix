# Development checks

From a source checkout:

```sh
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests tools
python tools/check_public.py --history
python -m build
python -m twine check dist/*
```

CI tests one installed wheel per platform on pull requests and `main`, avoiding duplicate branch-push runs. The matrix covers Linux with Python 3.10 and 3.14, and Windows with Python 3.13. The minimum-dependency job pins NumPy 1.24.0, SciPy 1.10.0 and Matplotlib 3.7.0. Long R comparisons and posterior-validation runners are separate from routine CI. Actual platform results are in the [evidence summary](reproduction.md#platform-evidence).

## R reference checks

R is required only for these checks. Install R `bkmr` **0.2.2**, `jsonlite` and their declared dependencies in an isolated R library. The CRAN archive SHA256 is `7ed29f74c4c78f3b8c663eac5ec417b4ea9bc586d789e8049e3ddc51541af9af`.

To regenerate the constructed-state fixtures, set `R_LIBS` to that library:

```sh
Rscript tools/export_reference.R tests/fixtures/r_reference.json
Rscript tools/export_cluster_reference.R tests/fixtures/r_cluster_reference.json
python -m pytest tests/test_reference.py tests/test_clusters.py
```

These exporters use installed prediction, inclusion, kernel, prior and covariance functions. The elementwise tolerance is `rtol=2e-11, atol=2e-12`. R uses denominator $S-1$ for covariance between conditional means; Python's finite empirical mixture uses $S$, where $S$ is the retained state count. The comparison adjusts this convention.

To compare fitted posterior summaries for the four fixed cases, first run a timing smoke check, then the four fixed cases. Use a new output directory for each attempt:

```sh
python tools/check_r_posteriors.py --r-library /path/to/r-library --output r-smoke --smoke
python tools/check_r_posteriors.py --r-library /path/to/r-library --output r-comparison --r-draws 120000 --python-draws 60000
```

Use `--rscript /path/to/Rscript` when R is not on `PATH`; `--case` selects one case. The checker records inputs, priors, package versions, source hashes, raw draws, diagnostics and Monte Carlo errors. It rejects an existing output directory and returns a nonzero exit code for a failed or imprecise comparison. The R calls carry state and the random-number stream between shorter segments. A short continuation check verifies identical states and random-number output across segment boundaries. Seeds and proposals differ between implementations; identical R and Python chains are not expected.

## Independent posterior targets

```sh
python tools/check_extension_posteriors.py --output extension-results.json
```

This runner compares fixed NB2, survival and structured-Gaussian fits with independent numerical integration. Its saved-draw counts, tolerances and target definitions are explicit in the script. It retains raw draws and source identities. See [results and limitations](reproduction.md) before interpreting a pass.

Documentation build commands are in [Contributing](https://github.com/nilson01/kernelmix/blob/main/CONTRIBUTING.md).
