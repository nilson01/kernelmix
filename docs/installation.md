# Installation

Python 3.10 or later is required. KernelMix uses NumPy and SciPy; R is not required for fitting or prediction. Matplotlib is an optional plotting dependency.

With access to the repository:

```sh
python -m pip install "kernelmix[plots] @ git+https://github.com/nilson01/kernelmix.git@main"
```

Or, from a downloaded source folder:

```sh
python -m pip install ".[plots]"
```

Omit `[plots]` if plots are not needed. Repository access is required for the Git installation route while the repository is private.

Start with the [analysis tutorial](tutorial.md). It explains the required arrays, fitting, diagnostics, prediction and exposure summaries. The [ozone example](airquality.md) includes observed data and saved results.
