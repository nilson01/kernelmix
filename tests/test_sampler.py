"""Bounded sampler integration and deterministic latent-update invariants.

These checks establish execution/state invariants, not posterior calibration.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import norm

from kernelmix import Prior, fit, model


@pytest.fixture
def small_data():
    index = np.arange(10, dtype=float)
    x = np.array([-1., 0., 1., -1., 0., 1., -1., 0., 1., -1.])[:, None]
    z = np.column_stack((index / 3, np.sin(index), index ** 2 / 20))
    binary = np.array([0., 1., 0., 1., 0., 1., 1., 0., 1., 0.])
    continuous = np.array([.2, 1.3, -.5, .7, -.2, 1.5, .8, -.3, 1.2, -.7])
    return continuous, binary, z, x


@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
@pytest.mark.parametrize('selection', ['none', 'component', 'group'])
@pytest.mark.parametrize('distribution', ['gamma', 'unif', 'invunif'])
def test_all_selected_sampler_paths_have_consistent_states(small_data, family, selection, distribution):
    continuous, binary, z, x = small_data
    y = binary if family == 'binomial' else continuous
    prior = Prior(r_distribution=distribution, r_lower=.1, r_upper=3.)
    fitted = fit(y, z, x, family=family, selection=selection,
                 groups=['metals', 'metals', 'other'] if selection == 'group' else None,
                 prior=prior, chains=2, draws=4, warmup=2, seed=927)
    samples = fitted.posterior_samples()
    expected = {'beta', 'r', 'lambda', 'sigma2', 'delta', 'pi'}
    assert set(samples) == (expected | {'ystar'} if family == 'binomial' else expected)
    assert samples['beta'].shape == (2, 4, 2)
    assert samples['r'].shape == samples['delta'].shape == (2, 4, 3)
    for parameter in ('lambda', 'sigma2', 'pi'):
        assert samples[parameter].shape == (2, 4)
    for values in samples.values():
        assert np.isfinite(values).all()
    assert np.all(samples['lambda'] > 0)
    assert np.all(samples['sigma2'] > 0)
    active = samples['delta'].astype(bool)
    assert_array_equal(samples['r'][~active], np.zeros((~active).sum()))
    assert np.all(samples['r'][active] > 0)
    for relevance in samples['r'][active]:
        assert np.isfinite(prior.log_r(relevance))
    if selection == 'none':
        assert active.all()
        assert_array_equal(samples['pi'], np.ones((2, 4)))
    else:
        assert np.all((samples['pi'] > 0) & (samples['pi'] < 1))
    if selection == 'group':
        assert np.all(active[..., :2].sum(axis=-1) <= 1)
        pips = fitted.inclusion_probabilities()
        assert len(pips) == 3
        assert all(0 <= row['group_pip'] <= 1 for row in pips)
    if family == 'binomial':
        assert_array_equal(samples['sigma2'], np.ones((2, 4)))
        assert samples['ystar'].shape == (2, 4, 10)
        assert np.all(samples['ystar'][..., y == 1] > 0)
        assert np.all(samples['ystar'][..., y == 0] < 0)
    assert fitted.settings['prior']['r_distribution'] == distribution
    assert fitted.sampling_info['runtime_seconds'] >= 0


def test_reproducible_independent_chains_do_not_mutate_global_rng_or_inputs(small_data):
    continuous, _, z, x = small_data
    originals = [a.copy() for a in (continuous, z, x)]
    before = np.random.get_state()
    arguments = dict(chains=2, draws=5, warmup=3, seed=812, thin=2)
    first = fit(continuous, z, x, **arguments)
    second = fit(continuous, z, x, **arguments)
    after = np.random.get_state()
    assert before[0] == after[0]
    assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    for parameter, values in first.posterior_samples().items():
        assert_array_equal(values, second.posterior_samples(parameter))
    assert not np.array_equal(first.samples['beta'][0], first.samples['beta'][1])
    assert first.sampling_info['seed_entropy'] == 812
    assert [info['seed_spawn_key'] for info in first.sampling_info['chains']] == [[0], [1]]
    for actual, original in zip((continuous, z, x), originals):
        assert_array_equal(actual, original)
    for info in first.sampling_info['chains']:
        counter = info['acceptance']['lambda']
        assert counter['proposed'] == 10  # 5 retained draws, thin=2
        assert counter['warmup_proposed'] == 3
        assert 0 <= counter['accepted'] <= counter['proposed']
        assert counter['rate'] == counter['accepted'] / counter['proposed']
        assert info['acceptance']['r']['proposed'] == 30
        assert info['acceptance']['selection']['proposed'] == 0


@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
def test_zero_fixed_kernel_and_empty_linear_design(small_data, family):
    continuous, binary, z, _ = small_data
    y = binary if family == 'binomial' else continuous
    fitted = fit(y, z, family=family, intercept=False, r_fixed=[0., 0., 0.],
                 chains=2, draws=4, warmup=2, seed=871)
    assert fitted.samples['beta'].shape == (2, 4, 0)
    assert fitted.data['X'].shape == (10, 0)
    assert_array_equal(fitted.samples['r'], np.zeros((2, 4, 3)))
    assert not fitted.samples['delta'].any()
    assert_array_equal(fitted.samples['pi'], np.ones((2, 4)))
    assert all(info['acceptance']['r']['proposed'] == 0
               for info in fitted.sampling_info['chains'])
    assert fitted.predict_h(z[:2], seed=3).draws.shape == (2, 4, 2)


@pytest.mark.parametrize('label', [0., 1.])
def test_one_class_probit_without_flat_beta_is_a_supported_proper_prior_model(label):
    z = np.arange(4, dtype=float)[:, None]
    fitted = fit(np.full(4, label), z, family='binomial', intercept=False,
                 r_fixed=[0.], chains=1, draws=2, warmup=0, seed=109)
    latent = fitted.samples['ystar']
    assert np.all(latent > 0) if label else np.all(latent < 0)
    assert_array_equal(fitted.samples['sigma2'], [[1., 1.]])


@pytest.mark.parametrize('covariate', [
    [-3., -2., -1., 1., 2., 3.],  # complete separation
    [-2., -1., 0., 0., 1., 2.],  # quasi separation, both labels at zero
])
def test_flat_beta_probit_separation_is_rejected(covariate):
    with pytest.raises(ValueError, match='separated'):
        fit([0., 0., 0., 1., 1., 1.], np.arange(6.)[:, None],
            np.asarray(covariate)[:, None], family='binomial',
            chains=1, draws=2, warmup=0, seed=14)


def test_one_class_probit_with_flat_intercept_is_rejected():
    with pytest.raises(ValueError, match='separated'):
        fit(np.ones(4), np.arange(4.)[:, None], family='binomial',
            chains=1, draws=2, warmup=0, seed=14)


def test_latent_sweep_uses_correlated_conditional_not_marginal_normal(monkeypatch):
    # For covariance [[2,1],[1,2]], coordinate conditional variance=3/2,
    # and conditional mean=mu_i+(z_other-mu_other)/2. The second update
    # must use the just-updated first coordinate. No Monte Carlo threshold.
    class OrderedRng:
        def permutation(self, size):
            assert size == 2
            return np.array([0, 1])

    calls = []

    def conditional_median(a, b, *, loc, scale, random_state):
        calls.append((a, b, loc, scale))
        # Exact conditional truncated-normal median supplies deterministic
        # valid values without making the production implementation its oracle.
        probability = (norm.cdf(a) + norm.cdf(b)) / 2
        return loc + scale * norm.ppf(probability)

    monkeypatch.setattr(model.truncnorm, 'rvs', conditional_median)
    initial = np.array([.8, -.4])
    means = np.array([.1, -.2])
    lower = np.linalg.cholesky([[2., 1.], [1., 2.]])
    updated = model._latent_sweep(np.array([1, 0]), initial.copy(), means, lower, OrderedRng())
    sd = np.sqrt(1.5)
    expected_first_mean = 0.0  # .1 + (-.4 -(-.2))/2
    assert_allclose(calls[0][2:], [expected_first_mean, sd], atol=1e-14)
    assert_allclose(calls[1][2:], [-.2 + (updated[0] - .1) / 2, sd], atol=1e-14)
    assert_allclose(calls[0][0], -expected_first_mean / sd, atol=1e-14)
    assert np.isposinf(calls[0][1]) and np.isneginf(calls[1][0])
    assert updated[0] > 0 and updated[1] < 0


@pytest.mark.parametrize('changes', [
    {'draws': 1}, {'warmup': -1}, {'chains': 0}, {'thin': 0}, {'seed': -1},
    {'draws': True}, {'chains': 1.5}, {'selection': 'unknown'},
    {'selection': 'group'}, {'groups': [1, 1, 2]},
    {'r_fixed': [-1, 1, 1]}, {'r_fixed': [1]}, {'r_fixed': [1, np.inf, 1]},
    {'r_fixed': [1, 1, 1], 'selection': 'component'},
    {'proposal_scales': {'r': 0}}, {'proposal_scales': {'lambda': np.nan}},
    {'proposal_scales': {'typo': 1}}, {'prior': object()},
    {'initial': {'unknown': 2}}, {'initial': {'beta': [1]}},
    {'initial': {'r': [0, 1, 1]}}, {'initial': {'pi': .5}},
    {'initial': {'ystar': np.ones(10)}},
    {'selection': 'component', 'initial': {'pi': 1}},
    {'selection': 'group', 'groups': [1, 1, 2],
     'initial': {'delta': [1, 1, 0]}},
])
def test_invalid_fit_options_are_rejected_before_sampling(small_data, changes):
    continuous, _, z, x = small_data
    arguments = dict(chains=1, draws=2, warmup=0, seed=11)
    arguments.update(changes)
    with pytest.raises((ValueError, TypeError)):
        fit(continuous, z, x, **arguments)


def test_fixed_mixed_relevance_is_preserved_and_prior_dictionary_is_accepted(small_data):
    continuous, _, z, _ = small_data
    fixed = np.array([0., .4, 2.])
    fitted = fit(continuous, z, r_fixed=fixed, prior={'r_distribution': 'gamma'},
                 chains=2, draws=3, warmup=1, seed=72)
    assert_allclose(fitted.samples['r'], np.broadcast_to(fixed, (2, 3, 3)), atol=0)
    assert_array_equal(fitted.samples['delta'], np.broadcast_to([False, True, True], (2, 3, 3)))
    assert_array_equal(fixed, [0., .4, 2.])


def test_giant_retained_sample_request_fails_before_allocating_or_sampling(small_data):
    continuous, _, z, _ = small_data
    with pytest.raises(ValueError, match='safety budget'):
        fit(continuous, z, chains=2, draws=10 ** 12, warmup=0, seed=4)


def test_component_selection_accepts_valid_empty_initial_state(small_data):
    continuous, _, z, _ = small_data
    initial = {'r': np.zeros(3), 'delta': np.zeros(3, dtype=bool), 'pi': .2,
               'beta': np.array([0.]), 'lambda': 1., 'sigma2': 1.}
    originals = {name: value.copy() if isinstance(value, np.ndarray) else value
                 for name, value in initial.items()}
    fitted = fit(continuous, z, selection='component', initial=initial,
                 chains=2, draws=3, warmup=1, seed=21)
    assert np.isfinite(fitted.samples['r']).all()
    for name in ('r', 'delta', 'beta'):
        assert_array_equal(initial[name], originals[name])


def test_marginal_likelihood_ratios_match_scipy_joint_normal():
    from scipy.stats import multivariate_normal

    covariance = np.array([[2., .3], [.3, 1.]])
    lower = np.linalg.cholesky(covariance)
    residual = np.array([.7, -1.2])
    zero = np.zeros(2)
    # The private target omits constants that cancel within MH proposals.
    actual = (model._log_likelihood(lower, residual, 1.7)
              - model._log_likelihood(lower, zero, 1.7))
    expected = (multivariate_normal.logpdf(residual, cov=1.7 * covariance)
                - multivariate_normal.logpdf(zero, cov=1.7 * covariance))
    assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)
