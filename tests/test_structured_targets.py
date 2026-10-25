"""Small posterior targets with nuisance parameters integrated independently."""

import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

from kernelmix import fit


def _target(log_density):
    normalizer = quad(lambda h: math.exp(log_density(h)), -np.inf, np.inf, epsabs=1e-10)[0]
    assert normalizer > 0

    def cdf(value):
        return (
            quad(lambda h: math.exp(log_density(h)), -np.inf, value, epsabs=1e-10)[0] / normalizer
        )

    median = brentq(lambda h: cdf(h) - 0.5, -10, 10)
    return median


def _check_median(draws, median):
    indicators = np.asarray(draws < median, dtype=float)
    batches = indicators.reshape(indicators.shape[0], 20, -1).mean(axis=-1).ravel()
    mcse = batches.std(ddof=1) / math.sqrt(len(batches))
    assert mcse < 0.035
    assert abs(indicators.mean() - 0.5) < 0.075
    assert abs(indicators.mean() - 0.5) < 0.018 + 6 * mcse


@pytest.mark.parametrize("family", ["gaussian", "poisson", "survival"])
def test_structured_constant_gp_posterior_against_integrated_density(family):
    """Freeze amplitude at1 by a sub-ulp proposal; sample all other selected blocks."""
    z = np.arange(4.0)[:, None]
    options = dict(
        approximation="rff",
        n_components=1,
        r_fixed=[0.0],
        intercept=False,
        draws=4000,
        warmup=1000,
        chains=2,
        seed=194,
        proposal_scales={"lambda": 1e-300},
    )
    if family == "gaussian":
        y = np.array([0.1, 0.4, -0.2, 0.6])

        def density(h):
            return float(norm.logpdf(h)) - (2 + len(y) / 2) * math.log(
                1 + 0.5 * float(np.sum((y - h) ** 2))
            )
    elif family == "poisson":
        y = np.array([0.0, 1.0, 1.0, 2.0])

        def density(h):
            if h > 700:
                return -np.inf
            return float(norm.logpdf(h)) + float(y.sum()) * h - len(y) * math.exp(h)
    else:
        y = np.array([1.0, 1.5, 2.0, 3.0])
        entry = np.array([0.2, 0.5, 0.0, 1.0])
        event = np.array([1, 0, 1, 0])
        options.update(entry=entry, event=event, baseline_breaks=[])
        options["proposal_scales"]["baseline_hazard"] = 0.7
        duration = float(np.sum(y - entry))
        events = float(event.sum())

        def density(h):
            return (
                float(norm.logpdf(h))
                + events * h
                - (0.1 + events) * float(logsumexp([math.log(0.1), math.log(duration) + h]))
            )

    median = _target(density)
    result = fit(y, z, family=family, **options)
    np.testing.assert_array_equal(result.samples["lambda"], 1.0)
    h = result.samples["gp_weight"][..., 0, 0]
    _check_median(h, median)
    if family == "gaussian":
        assert np.isfinite(result.samples["sigma2"]).all()
    if family == "survival":
        assert (result.samples["baseline_hazard"] > 0).all()
