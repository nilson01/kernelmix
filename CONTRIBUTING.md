# Contributing

Report bugs with a small example, package/dependency versions, the full error and
the expected behavior. Remove confidential data before sharing an example.

For a source checkout:

```sh
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check src tests tools
```

Include a focused regression test for a corrected defect. Method changes also need
an independent mathematical or numerical check and updated documentation. Describe
preprocessing, priors and effect scales when comparing software. Do not commit
research data, saved fitted objects, credentials or local build files.
