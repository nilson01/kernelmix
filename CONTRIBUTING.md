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