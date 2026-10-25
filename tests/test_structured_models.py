"""Public structured-model workflows with independent fixed-state oracles.

Tiny fits exercise integration only. Constructed states isolate transformations,
Gaussian conditional identities, and competing-risk probability calculations.
No posterior calibration or convergence assertion is made.
"""
import json
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import ndtr

from kernelmix import BKMRFit, StructuredFit, fit, overall_effect

TOL = {'rtol': 2e-10, 'atol': 2e-12}


def inputs():
    row = np.arange(8, dtype=float)
    return np.column_stack((row / 3, np.cos(row))), np.array([.4, -.3, 1.2, -.5, .8, -.2, .7, 1.])


def constant_state(fitted, beta=-.5):
    samples = {name: values.copy() for name, values in fitted.samples.items()}
    for name in ('gp_weight', 'gp_anchor', 'beta', 'random_coeff', 'gamma'):
        if name in samples:
            samples[name].fill(0.)
    if samples['beta'].shape[-1]:
        samples['beta'][..., 0] = beta
    samples['lambda'].fill(1.)
    samples['sigma2'].fill(2. if fitted.family == 'gaussian' else 1.)
    return StructuredFit(fitted.data, samples, fitted.settings, fitted.sampling_info)


@pytest.mark.parametrize('family', ['gaussian', 'binomial', 'poisson', 'negative_binomial', 'survival'])
@pytest.mark.parametrize('basis,selection,kernel', [
    ('dense', 'none', 'matern32'),
    ('pp', 'component', 'gaussian'),
    ('rff', 'group', 'rational_quadratic'),
])
def test_all_families_bases_and_selection_modes_reach_public_workflow(family, basis, selection, kernel):
    z, gaussian = inputs()
    y = (gaussian if family == 'gaussian' else np.array([0., 1., 0., 1., 1., 0., 1., 0.])
         if family == 'binomial' else np.array([0., 1., 3., 0., 2., 1., 0., 2.])
         if family in ('poisson', 'negative_binomial') else np.linspace(.5, 2., 8))
    options = dict(family=family, approximation=basis, kernel=kernel, n_components=3,
                   draws=2, warmup=1, chains=1, seed=408, selection=selection,
                   groups=['all', 'all'] if selection == 'group' else None,
                   intercept=family != 'survival')
    if family == 'survival':
        options.update(event=[1, 0, 1, 0, 1, 0, 1, 0], baseline_breaks=[])
    if basis == 'rff':
        options.update(cluster=['a'] * 4 + ['b'] * 4,
                       random_slopes=[-1., 0., 1., 2.] * 2,
                       random_covariance=[[.4, .1], [.1, .3]])
    fitted = fit(y, z, **options)
    assert isinstance(fitted, StructuredFit)
    assert fitted.settings['amplitude_convention'] == 'direct_tau'
    assert fitted.settings['beta_prior'] == 'normal'
    assert fitted.settings['approximation'] == basis
    for value in fitted.samples.values():
        assert np.isfinite(value).all()
    assert fitted.samples['beta'].shape[:3] == (1, 2, 1)
    assert fitted.samples['lambda'].shape == (1, 2, 1)
    assert_array_equal(fitted.samples['r'] > 0, fitted.samples['delta'])
    if selection == 'group':
        assert np.all(fitted.samples['delta'].sum(axis=-1) <= 1)
    if selection != 'none':
        assert len(fitted.inclusion_probabilities()) == 2
    assert fitted.summary_table()
    assert isinstance(fitted.diagnostics()['messages'], list)
    latent = fitted.predict(z[:2], scale='latent', seed=19)
    response = fitted.predict(z[:2], scale='response', seed=19)
    assert latent.draws.shape == response.draws.shape == (1, 2, 2)
    if basis == 'rff':
        coefficients = fitted.random_effects(['a', 'b'], seed=17).draws
        assert_array_equal(coefficients, fitted.samples['random_coeff'][:, :, 0])
        adjusted = fitted.predict(z[:2], scale='latent', cluster_new=['a', 'a'],
                                   random_slopes_new=[-1., 0.], seed=19)
        contribution = coefficients[..., 0, 0, None] + coefficients[..., 0, 1, None] * [-1., 0.]
        assert_allclose(adjusted.draws - latent.draws, contribution, **TOL)
        expected_cov = np.broadcast_to([[.4, .1], [.1, .3]], (1, 2, 1, 2, 2))
        assert_allclose(fitted.samples['random_covariance'], expected_cov, **TOL)
    if family == 'gaussian':
        assert_array_equal(response.draws, latent.draws)
    elif family == 'binomial':
        assert_allclose(response.draws, ndtr(latent.draws), **TOL)
    else:
        assert_allclose(response.draws, np.exp(latent.draws), **TOL)
    if family in ('poisson', 'negative_binomial'):
        observations = fitted.predict(z[:2], kind='observation', seed=2).draws
        assert np.all(observations >= 0)
        assert_array_equal(observations, np.floor(observations))
    if family == 'survival':
        with pytest.raises(ValueError):
            fitted.predict(z[:2], kind='observation')
        probability = fitted.predict_survival(z[:2], times=[0., 1.], seed=4)
        assert_array_equal(probability.draws[..., 0], np.ones((1, 2, 2)))
    contrast = overall_effect(fitted, max_draws=2, seed=14)
    assert_array_equal(contrast.draws[..., 1], np.zeros((1, 2)))


@pytest.fixture(scope='module')
def linear_state():
    z, y = inputs()
    template = fit(y, z, approximation='rff', n_components=2, r_fixed=[0., 0.],
        cluster=['a'] * 4 + ['b'] * 4, random_slopes=[-1., 0., 1., 2.] * 2,
        random_covariance=[[.4, .1], [.1, .3]], variance_design=np.linspace(-.7, .7, 8)[:, None],
        draws=2, warmup=1, chains=1, seed=881)
    state = constant_state(template)
    samples = {name: values.copy() for name, values in state.samples.items()}
    samples['random_coeff'][..., 0, 0, :] = [.3, .7]
    samples['random_coeff'][..., 0, 1, :] = [-.2, .4]
    samples['gamma'].fill(math.log(4))
    return StructuredFit(state.data, samples, state.settings, state.sampling_info)


def test_gaussian_mean_is_linear_and_omitted_random_effects_are_zero(linear_state):
    znew = np.array([[0., 0.], [1., 3.], [4., -2.]])
    got = linear_state.predict(znew, seed=29)
    assert_allclose(got.draws, np.full((1, 2, 3), -.5), **TOL)
    assert np.all(got.draws < 0)  # exponentiating a Gaussian mean would fail
    selected = linear_state.predict(znew, cluster_new=['a', 'a', 'b'],
                                     random_slopes_new=[0., 2., -1.], seed=29)
    expected = np.array([-.5 + .3, -.5 + .3 + 2 * .7, -.5 - .2 - .4])
    assert_allclose(selected.draws, np.broadcast_to(expected, (1, 2, 3)), **TOL)
    with pytest.raises(ValueError, match='random_slopes_new'):
        linear_state.predict(znew, cluster_new=['a', 'a', 'b'])


def test_public_random_coefficients_known_values_and_new_label_sharing(linear_state):
    labels = ['a', 'new', 'new', 'b']
    coefficients = linear_state.random_effects(labels, seed=30)
    assert coefficients.coefficients == ('Intercept', 'Slope1')
    assert coefficients.labels == tuple(labels)
    assert_allclose(coefficients.draws[..., 0, :], np.broadcast_to([.3, .7], (1, 2, 2)), **TOL)
    assert_allclose(coefficients.draws[..., 3, :], np.broadcast_to([-.2, .4], (1, 2, 2)), **TOL)
    assert_array_equal(coefficients.draws[..., 1, :], coefficients.draws[..., 2, :])
    assert_array_equal(linear_state.random_intercepts(labels, seed=30).draws,
                       coefficients.draws[..., 0])
    assert_array_equal(linear_state.random_effects(labels, seed=30).draws, coefficients.draws)
    typed = linear_state.random_effects([1, '1'], seed=42).draws
    assert not np.array_equal(typed[..., 0, :], typed[..., 1, :])


def test_heteroscedastic_variance_and_observation_covariance_have_physical_units(linear_state, monkeypatch):
    import kernelmix.structured_result as result_module

    znew = np.zeros((3, 2))
    variance = linear_state.predict_residual_variance(znew, variance_design_new=[-1., 0., 1.])
    assert_allclose(variance.draws, np.broadcast_to([.5, 2., 8.], (1, 2, 3)), **TOL)
    captured = []

    def conditional_mean(mean, covariance, rng):
        captured.append((mean.copy(), covariance.copy()))
        return mean.copy()

    monkeypatch.setattr(result_module, '_normal_draw', conditional_mean)
    observation = linear_state.predict(znew, kind='observation',
        variance_design_new=[-1., 0., 1.], seed=33)
    assert_allclose(observation.draws, -.5, **TOL)
    assert len(captured) == 2
    for mean, covariance in captured:
        assert_array_equal(mean, np.zeros(3))
        assert_allclose(covariance, np.diag([.5, 2., 8.]), **TOL)
    with pytest.raises(ValueError, match='variance_design_new'):
        linear_state.predict(znew, kind='observation')


@pytest.fixture(scope='module', params=['ar1', 'exponential', 'fixed'])
def correlated_state(request):
    z, y = inputs()
    times = np.tile(np.arange(4.), 2)
    options = dict(cluster=['a'] * 4 + ['b'] * 4,
        random_slopes=[-1., 0., 1., 2.] * 2, random_covariance=[[.4, .1], [.1, .3]],
        variance_design=np.linspace(-.7, .7, 8)[:, None], approximation='rff',
        n_components=2, r_fixed=[0., 0.], draws=2, warmup=1, chains=1, seed=121)
    if request.param == 'fixed':
        matrix = np.zeros((8, 8))
        block = .5 ** np.abs(np.arange(4)[:, None] - np.arange(4)[None, :])
        matrix[:4, :4] = matrix[4:, 4:] = block
        options['correlation'] = matrix
    else:
        options.update(correlation=request.param, observation_time=times,
                       correlation_parameter=.5 if request.param == 'ar1' else 1.)
    state = constant_state(fit(y, z, **options))
    samples = {name: value.copy() for name, value in state.samples.items()}
    samples['random_coeff'][..., 0, 0, :] = [.3, .7]
    samples['random_coeff'][..., 0, 1, :] = [-.2, .4]
    samples['gamma'].fill(math.log(4))
    return StructuredFit(state.data, samples, state.settings, state.sampling_info)


def test_correlated_training_prediction_is_the_same_observed_process(correlated_state):
    labels = ['a'] * 4 + ['b'] * 4
    for kind in ('mean', 'observation'):
        actual = correlated_state.predict(cluster_new=labels, kind=kind, seed=712)
        assert_allclose(actual.draws, np.broadcast_to(correlated_state.data['y'], (1, 2, 8)), **TOL)
    # Omission requests zero random effects and no known-process residual mean.
    omitted = correlated_state.predict(seed=712)
    assert_allclose(omitted.draws, -.5, **TOL)


def test_correlated_future_mean_and_variance_match_markov_normal_oracle(correlated_state, monkeypatch):
    import kernelmix.structured_result as result_module

    kind = correlated_state.settings['correlation']
    rho = math.exp(-1.) if kind == 'exponential' else .5
    options = dict(cluster_new=['a'], random_slopes_new=[3.],
                   variance_design_new=[.9], seed=41)
    if kind == 'fixed':
        options.update(correlation_cross=np.array([[rho ** 4, rho ** 3, rho ** 2, rho, 0., 0., 0., 0.]]),
                       correlation_new=np.ones((1, 1)))
    else:
        options['observation_time_new'] = [4.]
    # Training cluster a ends at t3 with eta1.2, y-.5 and log-variance V=-.1.
    # The new eta is1.9 at t4; exp((.9-(-.1))*log(4)/2)=2.
    expected_mean = 1.9 + rho * 2 * (-.5 - 1.2)
    expected_variance = 2 * 4 ** .9 * (1 - rho ** 2)
    mean = correlated_state.predict([[0., 0.]], **options)
    assert_allclose(mean.draws, expected_mean, **TOL)
    captured = []

    def conditional_mean(location, covariance, rng):
        captured.append(covariance.copy())
        return location.copy()

    monkeypatch.setattr(result_module, '_normal_draw', conditional_mean)
    observations = correlated_state.predict([[0., 0.]], kind='observation', **options)
    assert_allclose(observations.draws, expected_mean, **TOL)
    assert len(captured) == 2
    for covariance in captured:
        assert_allclose(covariance, [[expected_variance]], **TOL)
    if kind == 'fixed':
        independent_new = correlated_state.predict([[0., 0.]], cluster_new=['a'],
            random_slopes_new=[3.], variance_design_new=[.9], seed=41)
        assert_allclose(independent_new.draws, 1.9, **TOL)


def test_structured_gaussian_gp_amplitude_is_tau_not_residual_variance_ratio():
    z, y = inputs()
    template = fit(y, z, n_components=3, r_fixed=[.4, .9],
                   draws=2, warmup=1, chains=1, seed=73)
    state = constant_state(template)
    samples = {name: value.copy() for name, value in state.samples.items()}
    samples['lambda'].fill(4.)
    state = StructuredFit(state.data, samples, state.settings, {})
    new_original = np.array([[.15, .2]])
    new = (new_original - state.data['z_center']) / state.data['z_scale']
    train = state.data['Z']
    # Independent scalar Gaussian-kernel expression, not production basis code.
    weights = np.array([.4, .9])
    gram = np.exp(-np.sum(weights * (train[:, None, :] - train[None, :, :]) ** 2, axis=-1))
    cross = np.exp(-np.sum(weights * (new[:, None, :] - train[None, :, :]) ** 2, axis=-1))
    expected = 4 * (1 - (cross @ np.linalg.solve(gram, cross.T)).item())
    moments = state.posterior_moments_h(new_original)
    assert_allclose(moments['mean'], [0.], **TOL)
    assert_allclose(moments['covariance'], [[expected]], **TOL)
    samples['sigma2'].fill(50.)
    changed_residual = StructuredFit(state.data, samples, state.settings, {})
    assert_array_equal(state.predict_h(new_original, seed=19).draws,
                       changed_residual.predict_h(new_original, seed=19).draws)


@pytest.fixture(scope='module')
def competing_state():
    z, _ = inputs()
    template = fit([.8, .4, 1.2, .5, 1.5, 0., 1., .7], z,
        family='survival', event=[1, 2, 0, -1, 2, 1, 1, 0],
        entry=[.1, 0., .2, .1, .5, 0., 0., .2],
        upper=[.8, 1., np.inf, 1.3, 1.5, .6, 1., np.inf],
        n_causes=2, baseline_breaks=[.6], intercept=False,
        approximation='rff', n_components=2, r_fixed=[0., 0.],
        draws=2, warmup=1, chains=1, seed=916)
    state = constant_state(template)
    samples = {name: value.copy() for name, value in state.samples.items()}
    samples['baseline_hazard'].fill(1.)
    return StructuredFit(state.data, samples, state.settings, state.sampling_info)


def test_competing_cause_survival_cif_and_entry_have_independent_probability_oracles(competing_state):
    state = competing_state
    assert state.settings['n_causes'] == 2
    assert state.settings['entrant_cohort_conditional']
    assert_array_equal(state.data['is_interval'], [False, True, False, True, False, True, False, False])
    times = np.array([0., .25, .5, 1., 2.])
    survival = state.predict_survival([[0., 0.]], times=times, entry=.25, seed=20)
    cif = state.predict_cumulative_incidence([[0., 0.]], times=times, entry=.25, seed=20)
    expected = np.exp(-2 * np.maximum(times - .25, 0))
    assert_allclose(survival.draws, np.broadcast_to(expected, (1, 2, 1, 5)), **TOL)
    for cause in (1, 2):
        expected_cif = np.broadcast_to((1 - expected) / 2, (1, 2, 1, 5))
        assert_allclose(cif.for_cause(cause).draws, expected_cif, **TOL)
        direct = state.predict_cumulative_incidence([[0., 0.]], cause=cause,
                                                    times=times, entry=.25, seed=20)
        assert_allclose(direct.draws, expected_cif, **TOL)
    assert_allclose(survival.draws + cif.draws.sum(axis=-1), 1., **TOL)
    assert state.settings['active_cause'] == 1
    assert state.for_cause(2).settings['active_cause'] == 2
    with pytest.raises(ValueError):
        state.for_cause(3)
    with pytest.raises(ValueError):
        state.predict(kind='observation')


def test_public_survival_path_integrates_supplied_rows_and_requires_coverage(competing_state):
    times = np.array([0., .25, .5, 1., 2.])
    options = dict(start=[.25, .7, 1.3], stop=[.7, 1.3, 2.],
                   subject_new=['path', 'path', 'path'], times=times, seed=68)
    znew = np.array([[0., 0.], [2., -1.], [10., 3.]])
    survival = competing_state.predict_survival_path(znew, **options)
    cif = competing_state.predict_survival_path(znew, kind='cumulative_incidence', **options)
    expected = np.exp(-2 * np.maximum(times - .25, 0))
    assert_allclose(survival.draws, np.broadcast_to(expected, (1, 2, 1, 5)), **TOL)
    assert_allclose(cif.draws[..., 0], np.broadcast_to((1 - expected) / 2, (1, 2, 1, 5)), **TOL)
    assert_allclose(survival.draws + cif.draws.sum(axis=-1), 1., **TOL)
    with pytest.raises(ValueError, match='beyond'):
        competing_state.predict_survival_path(znew, **{**options, 'times': [0., 3.]})


@pytest.mark.parametrize('missing_field', ['terminal_event', 'terminal_exposure'])
def test_missing_terminal_row_drops_whole_subject_history(missing_field):
    z = np.column_stack((np.arange(6.), np.cos(np.arange(6.))))
    event = np.array([0., 1., 0., 1., 0., 2.])
    if missing_field == 'terminal_event':
        event[3] = np.nan
    else:
        z[3, 0] = np.nan
    fitted = fit([1., 2., 1., 2., 1., 2.], z, family='survival', event=event,
        entry=[0., 1., 0., 1., 0., 1.], subject=['a', 'a', 'b', 'b', 'c', 'c'],
        n_causes=2, baseline_breaks=[], intercept=False, approximation='rff',
        n_components=2, missing='drop', draws=2, warmup=1, chains=1, seed=94)
    assert_array_equal(fitted.data['rows'], [0, 1, 4, 5])
    assert fitted.data['original_n'] == 6
    assert fitted.data['subject_levels'] == ['a', 'c']
    assert fitted.data['dropped_subject_levels'] == ['b']
    assert_array_equal(fitted.data['event'], [0, 1, 0, 2])
    assert_array_equal(fitted.data['terminal'], [False, True, False, True])


@pytest.mark.parametrize('options', [
    {'variance_design': np.ones((8, 1))},
    {'variance_design': np.column_stack((np.arange(8.), 1 - np.arange(8.)))},
    {'random_slopes': np.ones((8, 1)), 'cluster': ['a'] * 4 + ['b'] * 4},
    {'random_slopes': np.column_stack((np.arange(8.), 2 * np.arange(8.))), 'cluster': ['a'] * 4 + ['b'] * 4},
    {'random_slopes': np.arange(8.), 'cluster': ['a'] * 4 + ['b'] * 4,
     'random_covariance': [[1., 2.], [2., 1.]]},
    {'correlation': 'ar1', 'observation_time': [0., 0., 1., 2.] * 2,
     'correlation_parameter': .5, 'cluster': ['a'] * 4 + ['b'] * 4},
    {'random_slopes': np.arange(8.)},
])
def test_invalid_structured_designs_fail_before_sampling(options):
    z, y = inputs()
    with pytest.raises(ValueError):
        fit(y, z, approximation='rff', n_components=2, draws=2, warmup=0, chains=1,
            seed=4, **options)


@pytest.fixture(scope='module', params=['dense', 'pp', 'rff'])
def archived_basis_state(request):
    z, y = inputs()
    return fit(y, z, approximation=request.param, n_components=3,
               r_fixed=[.4, .9], draws=2, warmup=1, chains=1, seed=401)


def test_safe_bkmr_load_dispatches_all_structured_basis_archives(archived_basis_state, tmp_path):
    fitted = archived_basis_state
    target = tmp_path / 'structured.npz'
    fitted.save(target)
    restored = BKMRFit.load(target)
    assert type(restored) is StructuredFit
    assert restored.settings == fitted.settings
    for name, value in fitted.samples.items():
        assert_array_equal(restored.samples[name], value)
    znew = np.array([[.1, .2], [1., -.3], [.1, .2]])
    assert_array_equal(restored.predict_h(znew, seed=7).draws,
                       fitted.predict_h(znew, seed=7).draws)
    with np.load(target, allow_pickle=False) as stored:
        assert all(stored[key].dtype != object for key in stored.files)


def test_invalid_archived_basis_fails_closed(archived_basis_state, tmp_path):
    target = tmp_path / 'invalid-basis.npz'
    archived_basis_state.save(target)
    with np.load(target, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    mode = archived_basis_state.settings['approximation']
    if mode == 'dense':
        arrays['data__basis_train_Z'][0, 0] += 1
    elif mode == 'pp':
        arrays['data__basis_knots'] = arrays['data__basis_knots'][:-1]
    else:
        arrays['data__basis_base_frequencies'][0, 0] = np.nan
    np.savez_compressed(target, **arrays)
    with pytest.raises(ValueError):
        BKMRFit.load(target)


def test_unknown_basis_metadata_and_object_arrays_are_not_loaded(linear_state, tmp_path):
    target = tmp_path / 'bad-metadata.npz'
    linear_state.save(target)
    with np.load(target, allow_pickle=False) as stored:
        arrays = {key: stored[key].copy() for key in stored.files}
    metadata = json.loads(arrays['metadata'].item())
    metadata['data']['basis_meta']['approximation'] = 'unknown'
    arrays['metadata'] = np.asarray(json.dumps(metadata))
    np.savez_compressed(target, **arrays)
    with pytest.raises(ValueError):
        BKMRFit.load(target)
    metadata['data']['basis_meta']['approximation'] = linear_state.settings['approximation']
    arrays['metadata'] = np.asarray(json.dumps(metadata))
    arrays['data__unsafe'] = np.asarray([object()], dtype=object)
    np.savez_compressed(target, **arrays)
    with pytest.raises(ValueError):
        BKMRFit.load(target)


def test_legacy_default_fit_and_archive_remain_bkmrfit(tmp_path):
    z, y = inputs()
    legacy = fit(y, z, draws=2, warmup=0, chains=1, seed=88)
    assert type(legacy) is BKMRFit
    target = tmp_path / 'legacy.npz'
    legacy.save(target)
    restored = BKMRFit.load(target)
    assert type(restored) is BKMRFit
    assert_array_equal(legacy.predict_h(z[:2], seed=8).draws,
                       restored.predict_h(z[:2], seed=8).draws)
