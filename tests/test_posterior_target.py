"""Small posterior targets obtained by one-dimensional numerical integration.

These fixed cases test transitions, not repeated-dataset calibration.
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import gamma

from kernelmix import Prior, diagnose, fit


def check_cdf(samples, density):
    normalizer = quad(density, 0, np.inf, epsabs=1e-10)[0]
    assert normalizer > 0
    for probability in (.25, .5, .75):
        threshold = brentq(lambda x: quad(density, 0, x)[0] / normalizer - probability,
                           1e-8, 100)
        indicators = (samples <= threshold).astype(float)
        ess = diagnose(indicators)['ess_bulk']
        assert np.isfinite(ess) and ess > 50
        tolerance = max(.025, 6 * np.sqrt(probability * (1-probability) / ess))
        assert abs(indicators.mean() - probability) < tolerance


@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
def test_marginal_lambda_posterior(family):
    prior = Prior(lambda_mean=2, lambda_sd=1, sigma2_shape=3, sigma2_rate=1)
    if family == 'gaussian':
        y = np.array([-.7, .2, .9, -.1, 1.1, -.3, .4, .6])
        n = len(y)
        def density(value):
            # Sherman-Morrison for V = I + value * 11^T, integrating sigma2.
            quadratic = y @ y - value * y.sum()**2 / (1 + n*value)
            return gamma.pdf(value, a=4, scale=.5) / np.sqrt(1+n*value) * (
                1 + quadratic / 2)**(-(3+n/2))
    else:
        y = np.array([0., 1.])
        def density(value):
            rho = value / (1+value)
            # Opposite-sign bivariate-normal orthant probability.
            return gamma.pdf(value, a=4, scale=.5) * (.25-np.arcsin(rho)/(2*np.pi))
    result = fit(y, np.zeros((len(y), 1)), intercept=False, standardize=False,
                 r_fixed=[0], prior=prior, family=family, seed=410,
                 chains=4, warmup=500, draws=2000, proposal_scales={'lambda': .8})
    check_cdf(result.samples['lambda'], density)


def test_selection_prior_is_preserved_when_kernel_cannot_depend_on_selection():
    result = fit(np.array([-.5, .2, .9, -.1]), np.zeros((4, 3)),
                 standardize=False, intercept=False, selection='component',
                 prior=Prior(r_distribution='gamma'), chains=4,
                 draws=1000, warmup=200, seed=75)
    counts = result.samples['delta'].sum(axis=-1)
    # Integrating Beta(1,1) gives a uniform count on {0,...,M}.
    for count in range(4):
        indicators = (counts == count).astype(float)
        ess = diagnose(indicators)['ess_bulk']
        assert ess > 50
        assert abs(indicators.mean() - .25) < max(.035, 6*np.sqrt(.1875/ess))
