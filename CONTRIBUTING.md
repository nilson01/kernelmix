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
preprocessing, priors and effect scales when comparing software. Document the
source and license of public example data. Keep confidential or restricted data,
fitted archives, credentials and local build files out of the repository.
