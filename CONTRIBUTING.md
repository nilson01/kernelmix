# Contributing

Report bugs with a small example, package/dependency versions, the full error and
the expected behavior. Remove confidential data before sharing an example.

For a source checkout:

```sh
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests tools
```

Include a regression test for a corrected defect. Method changes need an independent mathematical or numerical check and updated documentation. Software comparisons must state input transformations, priors and effect scales. Record example-data sources and licenses; exclude confidential data, fitted archives, credentials and local build files.

## Build documentation

The maintained sources are the README and Markdown guides in `docs`.

Install the optional documentation tools:

```sh
python -m pip install ".[docs]"
python -m mkdocs build --strict
```

The static website is written to `site/`. This command builds files locally; it does not deploy them. The site groups the maintained API chapter by task, adds search and renders equations with MathJax. Browser equation rendering requires access to the pinned MathJax CDN resource.

To rebuild `docs/manual.pdf`, install a local `pdflatex` distribution, then run:

```sh
python tools/build_manual.py
```

Pandoc and LaTeX build the PDF; intermediate TeX is temporary. Inspect the resulting website and PDF after content changes, including links, equations, figures and page layout. Keep numerical evidence synchronized with the checks that actually ran.


## Refresh the source example data

The ozone source CSV was exported from R 4.5.0. To export another copy for a documented comparison:

```sh
Rscript tools/export_airquality.R destination.csv
```

Keep the source version, license, hash, units and row exclusions with any replacement. When source data change, regenerate affected results and record the new version.
