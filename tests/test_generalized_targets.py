"""Bounded independent posterior targets; these are not calibration studies."""
import math

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad

import kernelmix as km
import kernelmix._generalized as engine


def test_elliptical_slice_conjugate_gaussian_target():
    # N(0,I) prior times N(observed | latent, sd^2 I) likelihood.
    # Conjugacy gives mean observed/(1+sd^2), variance sd^2/(1+sd^2).
    observed = np.array([1., -.5])
    sd = .8
    expected_mean = observed / (1 + sd**2)
    expected_variance = sd**2 / (1 + sd**2)

    def log_likelihood(value):
        return -.5 * np.sum(((observed - value) / sd)**2)

    rng = np.random.default_rng(618)
    state = np.zeros(2)
    loglik = log_likelihood(state)
    saved = []
    evaluations = 0
    for iteration in range(4500):
        state, loglik, used = engine.elliptical_slice(state, log_likelihood, rng, current_loglik=loglik)
        evaluations += used
        if iteration >= 500:
            saved.append(state.copy())
    saved = np.asarray(saved)
    assert_allclose(saved.mean(axis=0), expected_mean, atol=.07, rtol=0)
    assert_allclose(np.cov(saved, rowvar=False), np.eye(2)*expected_variance, atol=.06, rtol=0)
    assert evaluations >= 4500


def test_elliptical_slice_returns_proposal_and_fails_on_invalid_bracket():
    rng = np.random.default_rng(84)
    # Constant likelihood accepts the first angle. Recreate the random ellipse
    # from a second generator without reproducing bracket shrink logic.
    current = np.array([.4, -.2])
    state, loglik, evaluations = engine.elliptical_slice(current, lambda value: -1., rng, current_loglik=-1.)
    oracle = np.random.default_rng(84)
    direction = oracle.normal(size=2)
    oracle.random()  # Slice level.
    theta = oracle.uniform(0, 2*np.pi)
    assert_allclose(state, current*np.cos(theta)+direction*np.sin(theta), atol=1e-14)
    assert loglik == -1
    assert evaluations == 1
    import pytest
    with pytest.raises(RuntimeError):
        engine.elliptical_slice(current, lambda value: -np.inf, np.random.default_rng(1), current_loglik=-1., max_steps=3)
    with pytest.raises(ValueError):
        engine.elliptical_slice(current, lambda value: -np.inf, np.random.default_rng(1))


def _constant_gp_poisson_oracle(y):
    """Integrate the common h posterior after analytically integrating lambda.

    lambda~Gamma(4,rate4), h|lambda~N(0,lambda). The half-integer Bessel
    expression for the marginal prior reduces, up to a cancelling constant, to
    exp(-c|h|)*(|h|^3+6|h|^2/c+15|h|/c^2+15/c^3), c=sqrt(8).
    This oracle uses no KernelMix kernel, likelihood, MH or prediction helper.
    """
    n, total = len(y), sum(y)
    c = math.sqrt(8)

    def weight(h):
        absolute = abs(h)
        polynomial = absolute**3 + 6*absolute**2/c + 15*absolute/c**2 + 15/c**3
        log_weight = total*h - n*math.exp(h) - c*absolute + math.log(polynomial)
        return math.exp(log_weight)

    # On these fixed inputs, [-40,10] is much wider than the posterior support.
    denominator, den_error = quad(weight, -40., 10., points=[0.], epsabs=1e-11, epsrel=1e-10)
    numerator, num_error = quad(lambda h: math.exp(h)*weight(h), -40., 10., points=[0.], epsabs=1e-11, epsrel=1e-10)
    below_zero, cdf_error = quad(weight, -40., 0., epsabs=1e-11, epsrel=1e-10)
    assert denominator > 0
    assert max(den_error, num_error, cdf_error) < 1e-8
    return numerator/denominator, below_zero/denominator


def _batch_mcse(values, batch_size=100):
    """Between-batch MCSE of a mean, preserving each chain's batches."""
    values = np.asarray(values)
    assert values.ndim == 2 and values.shape[1] % batch_size == 0
    means = values.reshape(values.shape[0], -1, batch_size).mean(axis=-1).ravel()
    return float(means.std(ddof=1)/np.sqrt(len(means)))


def test_public_poisson_constant_gp_posterior_against_one_dimensional_quadrature():
    y = [0, 1, 1, 2]
    expected_rate, expected_cdf = _constant_gp_poisson_oracle(y)
    # Fixing r=0 makes h common to all observations. No beta or cluster term.
    # Default new-family amplitude prior is exactly Gamma(shape4,rate4).
    fitted = km.fit(y, np.arange(4.)[:, None], family='poisson', intercept=False,
                    r_fixed=[0.], draws=2000, warmup=500, chains=2, seed=733)
    h = fitted.samples['h'][..., 0]
    assert_allclose(fitted.samples['h'], np.broadcast_to(h[..., None], fitted.samples['h'].shape), atol=1e-13)
    rates = np.exp(h)
    indicators = (h <= 0).astype(float)
    mean_error = abs(rates.mean()-expected_rate)
    cdf_error = abs(indicators.mean()-expected_cdf)
    rate_mcse, cdf_mcse = _batch_mcse(rates), _batch_mcse(indicators)
    # Both fixed bounds and an MCSE bound must pass. Low effective information
    # therefore cannot make a wrong result pass by inflating its tolerance.
    assert rate_mcse < .035
    assert cdf_mcse < .035
    assert mean_error < .12
    assert cdf_error < .10
    assert mean_error < max(.04, 6*rate_mcse)
    assert cdf_error < max(.04, 6*cdf_mcse)
    assert fitted.settings['prior']['lambda_mean'] == 1
    assert fitted.settings['prior']['lambda_sd'] == .5


def test_conditional_survival_baseline_draws_against_gamma_moments(monkeypatch):
    # Deliberately hold all Gaussian latent blocks at zero, isolating the
    # conditional baseline Gibbs update. This is not a joint-posterior test.
    def hold_zero(current, log_likelihood, rng, *, current_loglik=None, max_steps=10000):
        assert_array_equal(current, np.zeros_like(current))
        return current.copy(), float(log_likelihood(current)), 1

    monkeypatch.setattr(engine, 'elliptical_slice', hold_zero)
    prior = km.Prior(lambda_mean=1., lambda_sd=.5, cluster_lambda_mean=1.,
                     cluster_lambda_sd=.5, baseline_shape=2., baseline_rate=1.)
    fitted = km.fit([1., 2., 3., 4.], np.arange(4.)[:, None], family='survival',
                    event=[1, 1, 0, 1], baseline_breaks=[2.], intercept=False,
                    r_fixed=[0.], prior=prior, draws=2500, warmup=0, chains=1, seed=681)
    # Explicit interval totals are7 and3; event counts are2 and1.
    shapes, rates = np.array([4., 3.]), np.array([8., 4.])
    expected_mean = shapes/rates
    expected_variance = shapes/rates**2
    draws = fitted.samples['baseline_hazard'][0]
    assert_allclose(draws.mean(axis=0), expected_mean,
                    atol=6*float(np.sqrt(np.max(expected_variance)/len(draws))), rtol=0)
    assert_allclose(draws.var(axis=0, ddof=1), expected_variance, rtol=.15, atol=.005)
    assert_array_equal(fitted.samples['h'], np.zeros((1, 2500, 4)))


def test_nb2_conditional_dispersion_posterior_against_quadrature(monkeypatch):
    """Hold eta=0 and check dispersion MH against an independent scalar target."""
    from scipy.optimize import brentq
    from scipy.stats import gamma, nbinom

    def hold_zero(current, log_likelihood, rng, *, current_loglik=None, max_steps=10000):
        assert_array_equal(current, np.zeros_like(current))
        return current.copy(), float(log_likelihood(current)), 1

    monkeypatch.setattr(engine, 'elliptical_slice', hold_zero)
    y = np.array([0, 0, 1, 3])

    def density(size):
        if size <= 0:
            return 0.
        return np.exp(nbinom.logpmf(y, size, size/(size+1)).sum() + gamma.logpdf(size, a=2, scale=10))

    normalizer = quad(density, 0, np.inf, epsabs=1e-11)[0]
    median = brentq(lambda value: quad(density, 0, value)[0]/normalizer-.5, 1e-8, 300)
    fitted = km.fit(y, np.arange(4.)[:, None], family='negative_binomial',
                    intercept=False, r_fixed=[0.], draws=2000, warmup=500, chains=2,
                    proposal_scales={'dispersion': .8}, seed=583)
    indicators = (fitted.samples['dispersion'] <= median).astype(float)
    mcse = _batch_mcse(indicators)
    assert mcse < .035
    assert abs(indicators.mean()-.5) < .07
    assert abs(indicators.mean()-.5) < max(.04, 6*mcse)
    assert_array_equal(fitted.samples['h'], np.zeros((2, 2000, 4)))
