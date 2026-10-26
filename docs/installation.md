# Installation

Python 3.10 or later is required. KernelMix uses NumPy and SciPy; R is not required for fitting or prediction.

## From PyPI

```sh
python -m pip install kernelmix
```

For plots, install Matplotlib through the plotting extra:

```sh
python -m pip install "kernelmix[plots]"
```

## From GitHub

```sh
python -m pip install "kernelmix[plots] @ git+https://github.com/nilson01/kernelmix.git@v0.3.1"
```

Start with the [analysis tutorial](tutorial.md) for fitting, diagnostics, prediction and exposure summaries. The [ozone example](airquality.md) includes observed data and saved results.
