"""Independent likelihood arithmetic and generalized public-API contracts."""
from dataclasses import asdict

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import nbinom, poisson

import kernelmix as km
from kernelmix._generalized import _gp_factor
from kernelmix._likelihoods import (
    baseline_posterior_parameters,
    interval_durations,
    negative_binomial_loglik,
    poisson_loglik,
    prepare_survival,
    survival_loglik,
)


@pytest.mark.parametrize('eta', [np.array([-20., -1., .5, 4.]), np.array([0., 1., 2., 3.])])
def test_poisson_loglik_matches_scipy_normalized_mass(eta):
    y = np.array([0, 1, 4, 17])
    assert_allclose(poisson_loglik(y, eta), poisson.logpmf(y, np.exp(eta)).sum(), rtol=2e-13, atol=1e-12)


@pytest.mark.parametrize('dispersion', [.08, 1., 3.5, 50., 10000.])
def test_nb2_loglik_matches_scipy_mean_dispersion_parameterization(dispersion):
    y = np.array([0, 1, 4, 17])
    eta = np.array([-3., -1., .5, 4.])
    mu = np.exp(eta)
    expected = nbinom.logpmf(y, dispersion, dispersion / (dispersion + mu)).sum()
    assert_allclose(negative_binomial_loglik(y, eta, dispersion), expected, rtol=2e-10, atol=2e-10)


def test_piecewise_survival_equals_event_density_times_censor_survivor():
    time = [1., 2., 3.5]
    event = [1, 0, 1]
    eta = np.log([2., .5, 1.5])
    pieces = prepare_survival(time, event, [2.])
    # Independent integrated hazards: 1*.2, 2*.2, 2*.2+1.5*.7.
    expected = np.log(.2 * 2) - .2 * 2 - .4 * .5 + np.log(.7 * 1.5) - 1.45 * 1.5
    actual = survival_loglik(event, pieces['event_bin'], pieces['durations'], eta, [.2, .7])
    assert_allclose(actual, expected, atol=1e-13)
    # Poisson interval splitting differs only by event log-duration factors.
    split_y = np.array([[1, 0], [0, 0], [0, 1]])
    positive = pieces['durations'] > 0
    split_mu = np.exp(eta[:, None]) * pieces['durations'] * np.array([.2, .7])
    split = poisson.logpmf(split_y[positive], split_mu[positive]).sum()
    adjustment = np.log(pieces['durations'][0, 0]) + np.log(pieces['durations'][2, 1])
    assert_allclose(actual, split - adjustment, atol=1e-13)


def test_cut_event_is_assigned_positive_own_duration_and_bounded_density():
    pieces = prepare_survival([1., 2., 3.], [1, 1, 0], [1., 2.])
    assert_array_equal(pieces['event_bin'], [0, 1, 2])
    assert_array_equal(pieces['durations'], [[1, 0, 0], [1, 1, 0], [1, 1, 1]])
    rows = np.flatnonzero(pieces['event'])
    own = pieces['durations'][rows, pieces['event_bin'][rows]]
    assert np.all(own > 0)
    # Regression counterexample: assigning a cut event to the right gives D=0.
    wrong_bins = np.searchsorted([1., 2.], [1., 2., 3.], side='right')
    assert_array_equal(pieces['durations'][rows, wrong_bins[rows]], [0., 0.])
    # Every observed density r*exp(-rD) is bounded by 1/(eD).
    rates = np.array([.01, .2, 1., 5., 100.])[:, None]
    density = rates * np.exp(-rates * own)
    assert np.all(density <= 1 / (np.e * own) + 1e-14)


def test_survival_default_cuts_empty_baseline_and_gamma_parameters():
    default = prepare_survival([1., 2., 3., 4.], [1, 0, 1, 0])
    assert np.all((default['baseline_breaks'] > 0) & (default['baseline_breaks'] < 4))
    assert len(default['baseline_breaks']) <= 9
    assert np.all(np.diff(default['baseline_breaks']) > 0)
    censored = prepare_survival([1., 2.], [0, 0])
    assert len(censored['baseline_breaks']) == 0
    pieces = prepare_survival([1., 2., 3., 4.], [1, 1, 0, 1], [2.])
    shapes, rates = baseline_posterior_parameters(pieces['event'], pieces['event_bin'], pieces['durations'], np.zeros(4), 2., 1.)
    assert_array_equal(shapes, [4., 3.])
    assert_array_equal(rates, [8., 4.])
    assert_array_equal(interval_durations([0., 1., 2., 5.], [2.]), [[0, 0], [1, 0], [2, 0], [2, 3]])


@pytest.mark.parametrize('family', ['poisson', 'negative_binomial', 'survival'])
@pytest.mark.parametrize('selection', ['none', 'component', 'group'])
def test_new_families_selection_shapes_and_saved_state_identity(family, selection):
    z = np.array([[10., 1.], [20., 2.], [30., 1.5], [40., 3.]])
    kwargs = dict(family=family, selection=selection, draws=4, warmup=2, chains=1,
                  seed=94, cluster=['a', 'a', 'b', 'b'], intercept=False)
    if selection == 'group':
        kwargs['groups'] = ['mixture', 'mixture']
    if family == 'survival':
        kwargs.update(event=[1, 0, 1, 0], baseline_breaks=[2.])
        y = [1., 2., 3., 4.]
    else:
        y = [0, 1, 2, 1]
    fitted = km.fit(y, z, **kwargs)
    assert fitted.samples['h'].shape == (1, 4, 4)
    assert fitted.samples['latent_white'].shape == (1, 4, 4)
    assert fitted.samples['u'].shape == (1, 4, 2)
    assert_array_equal(fitted.samples['sigma2'], np.ones((1, 4)))
    assert np.all(fitted.samples['lambda_cluster'] > 0)
    assert np.all(fitted.samples['r'][~fitted.samples['delta']] == 0)
    if selection == 'group':
        assert np.all(fitted.samples['delta'].sum(axis=-1) <= 1)
    if selection != 'none':
        assert len(fitted.inclusion_probabilities()) == 2
    else:
        assert_array_equal(fitted.samples['pi'], np.ones((1, 4)))
    if family == 'negative_binomial':
        assert np.all(fitted.samples['dispersion'] > 0)
    if family == 'survival':
        assert fitted.samples['baseline_hazard'].shape == (1, 4, 2)
    for draw in range(4):
        root = _gp_factor(fitted.data['Z'], fitted.samples['r'][0, draw])
        expected = np.sqrt(fitted.samples['lambda'][0, draw]) * root @ fitted.samples['latent_white'][0, draw]
        assert_allclose(fitted.samples['h'][0, draw], expected, atol=1e-12)
    information = fitted.sampling_info['chains'][0]
    assert information['likelihood_evaluations']['sampling'] >= 4
    assert information['acceptance']['ess']['accepted'] == 4
    assert all(row['parameter'] != 'sigma2' for row in fitted.summary_table())


@pytest.mark.parametrize('family', ['poisson', 'negative_binomial', 'survival'])
def test_new_family_reproducibility_archive_and_effect_routes(family, tmp_path):
    z = np.array([[10., 100.], [20., 150.], [30., 200.], [40., 300.]])
    kwargs = dict(family=family, draws=4, warmup=2, chains=1, seed=119, intercept=False)
    y = [0, 1, 2, 1]
    if family == 'survival':
        y = [1., 2., 3., 4.]
        kwargs.update(event=[1, 0, 1, 0], baseline_breaks=[])
    first = km.fit(y, z, **kwargs)
    second = km.fit(y, z, **kwargs)
    for name in first.samples:
        assert_array_equal(first.samples[name], second.samples[name])
    path = tmp_path / 'state.npz'
    first.save(path)
    restored = km.BKMRFit.load(path)
    for name in first.samples:
        assert_array_equal(first.samples[name], restored.samples[name])
    assert_allclose(first.predict(z, seed=19).draws, restored.predict(z, seed=19).draws)
    effect = km.overall_effect(restored, scale='response', seed=3)
    assert_array_equal(effect.draws[..., 1], np.zeros((1, 4)))
    assert effect.metadata['random_intercept'] == 'zero'
    assert effect.metadata['response_quantity'] == ('relative_hazard' if family == 'survival' else 'unit-exposure count mean')
    assert np.isfinite(km.univariate_response(restored, variable=0, points=3, seed=3).mean).all()


@pytest.mark.parametrize('family', ['poisson', 'survival'])
def test_extra_missingness_joins_original_rows_transform_and_cluster_mask(family):
    z = np.column_stack((np.arange(7.) + 10, np.array([1., 3., 2., 4., 5., 8., 6.])))
    z[1, 0] = np.nan
    x = np.arange(7.)[:, None]
    x[3, 0] = np.nan
    cluster = ['a', 'a', 'b', 'b', 'c', None, 'c']
    kwargs = dict(family=family, X=x, cluster=cluster, missing='drop',
                  draws=2, warmup=0, chains=1, seed=44, intercept=False)
    if family == 'poisson':
        y = np.array([0., 1., 0., 2., 1., 0., 1.])
        kwargs['offset'] = [0., 0., np.nan, 0., .2, 0., .4]
    else:
        y = np.arange(1., 8.)
        kwargs.update(event=[1., 0., np.nan, 0., 1., 0., 0.], baseline_breaks=[])
    fitted = km.fit(y, z, **kwargs)
    assert_array_equal(fitted.data['rows'], [0, 4, 6])
    assert fitted.data['original_n'] == 7
    assert_array_equal(fitted.data['y'], y[[0, 4, 6]])
    assert_array_equal(fitted.data['cluster_codes'], [0, 1, 1])
    assert fitted.data['cluster_levels'] == ['a', 'c']
    assert_allclose(fitted.data['z_center'], z[[0, 4, 6]].mean(axis=0))
    if family == 'poisson':
        assert_array_equal(fitted.data['offset'], [0., .2, .4])
    else:
        assert_array_equal(fitted.data['event'], [1, 1, 0])


def test_model_specific_default_priors_explicit_prior_and_recorded_warnings():
    z = np.array([[0.], [1.], [2.]])
    default = km.fit([0, 0, 0], z, family='poisson', draws=2, warmup=0, chains=1, seed=1)
    assert default.settings['prior']['lambda_mean'] == 1
    assert default.settings['prior']['lambda_sd'] == .5
    assert default.settings['prior']['cluster_lambda_mean'] == 1
    assert default.settings['prior']['cluster_lambda_sd'] == .5
    assert not default.sampling_info['warnings']
    prior = km.Prior(dispersion_shape=.8)
    with pytest.warns(RuntimeWarning):
        explicit = km.fit([0, 1, 0], z, family='negative_binomial', prior=prior,
                          draws=2, warmup=0, chains=1, seed=2)
    assert explicit.settings['prior'] == asdict(prior)
    assert len(explicit.sampling_info['warnings']) >= 2
    with pytest.warns(RuntimeWarning):
        censored = km.fit([1., 2., 3.], z, family='survival', event=[0, 0, 0],
                          draws=2, warmup=0, chains=1, seed=3)
    assert len(censored.data['baseline_breaks']) == 0
    assert len(censored.sampling_info['warnings']) >= 2
    for message in censored.sampling_info['warnings']:
        assert message in censored.summary()


@pytest.mark.parametrize('family,kwargs,y', [
    ('poisson', {}, [-1, 0, 1]), ('poisson', {}, [0, .5, 1]),
    ('poisson', {'offset': [0, np.inf, 0]}, [0, 1, 1]),
    ('poisson', {'offset': [0, np.nan, 0]}, [0, 1, 1]),
    ('poisson', {'offset': [0, 0]}, [0, 1, 1]),
    ('poisson', {'event': [1, 0, 1]}, [0, 1, 1]),
    ('negative_binomial', {'baseline_breaks': []}, [0, 1, 1]),
    ('survival', {}, [1, 2, 3]),
    ('survival', {'event': [1, 2, 0]}, [1, 2, 3]),
    ('survival', {'event': [1, 0, 1]}, [0, 2, 3]),
    ('survival', {'event': [1, 0, 1], 'offset': [0, 0, 0]}, [1, 2, 3]),
    ('survival', {'event': [1, 0, 1], 'baseline_breaks': [2, 1]}, [1, 2, 3]),
    ('poisson', {'initial': {'ystar': [1, 1, 1]}}, [0, 1, 1]),
    ('poisson', {'initial': {'dispersion': 1}}, [0, 1, 1]),
    ('poisson', {'initial': {'lambda_cluster': 1}}, [0, 1, 1]),
    ('negative_binomial', {'initial': {'dispersion': -1}}, [0, 1, 1]),
    ('poisson', {'proposal_scales': {'dispersion': .3}}, [0, 1, 1]),
    ('poisson', {'proposal_scales': {'cluster_lambda': .3}}, [0, 1, 1]),
    ('poisson', {'r_fixed': [0], 'selection': 'component'}, [0, 1, 1]),
])
def test_new_family_input_errors(family, kwargs, y):
    with pytest.raises(ValueError):
        km.fit(y, [[0.], [1.], [2.]], family=family, draws=2, warmup=0, chains=1, seed=8, **kwargs)


def saved_generalized(family='poisson', clustered=True, offset_used=False):
    """Direct finite saved states; useful for prediction identities, not inference."""
    raw = np.array([[0.], [1.], [2.], [3.]])
    center, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
    y = np.array([1., 2., 3., 4.]) if family == 'survival' else np.array([0., 1., 2., 1.])
    data = dict(y=y, Z=(raw-center)/scale, Z_original=raw, X=np.ones((4, 1)),
                X_original=np.empty((4, 0)), z_center=center, z_scale=scale,
                rows=np.arange(4), original_n=4, exposure_names=['Dose'],
                covariate_names=['Intercept'], groups=None,
                cluster_codes=np.array([0, 0, 1, 1]) if clustered else None,
                cluster_levels=['a', 'b'] if clustered else [])
    samples = dict(beta=np.array([[[-.2], [0.], [.3], [.6]]]), r=np.zeros((1, 4, 1)),
                   h=np.zeros((1, 4, 4)), latent_white=np.zeros((1, 4, 4)),
                   lambda_=np.ones((1, 4)), sigma2=np.ones((1, 4)),
                   delta=np.zeros((1, 4, 1), dtype=bool), pi=np.ones((1, 4)))
    samples['lambda'] = samples.pop('lambda_')
    if clustered:
        samples.update(lambda_cluster=np.array([[.2, .4, .6, .8]]),
                       u=np.array([[[.1, -.2], [.2, -.1], [.3, 0.], [.4, .1]]]))
    if family == 'survival':
        data.update(prepare_survival(y, [1, 0, 1, 0], [2.]))
        samples['baseline_hazard'] = np.array([[[.2, .4], [.3, .6], [.4, .8], [.5, 1.]]])
    else:
        data['offset'] = np.log([1., 2., 3., 4.]) if offset_used else np.zeros(4)
    if family == 'negative_binomial':
        samples['dispersion'] = np.array([[1., 2., 3., 4.]])
    settings = dict(family=family, chains=1, draws=4, intercept=True, selection='none',
                    clustered=clustered, offset_used=offset_used, standardize=True)
    return km.BKMRFit(data, samples, settings, {})


@pytest.mark.parametrize('family', ['poisson', 'negative_binomial'])
def test_count_mean_offsets_and_observation_scale(family):
    fitted = saved_generalized(family, offset_used=True)
    training = fitted.predict(seed=2)
    expected = np.exp(fitted.samples['beta'][..., 0, None]) * np.array([1., 2., 3., 4.])
    assert_allclose(training.draws, expected, rtol=2e-7, atol=2e-7)
    new = np.array([[.5], [.5]])
    with pytest.raises(ValueError):
        fitted.predict(new, seed=2)
    explicit = fitted.predict(new, offset=np.log([1., 3.]), seed=2)
    assert_allclose(explicit.draws[..., 1], 3 * explicit.draws[..., 0], rtol=1e-13)
    observed = fitted.predict(new, offset=np.zeros(2), kind='observation', seed=2)
    assert np.all(observed.draws >= 0)
    assert_array_equal(observed.draws, np.floor(observed.draws))
    with pytest.raises(ValueError):
        fitted.predict(new, offset=np.zeros(2), kind='observation', scale='latent')
    effect = km.overall_effect(fitted, scale='response', seed=2)
    assert_array_equal(effect.draws, np.zeros((1, 4, 3)))


@pytest.mark.parametrize('family', ['poisson', 'negative_binomial', 'survival'])
def test_known_and_unknown_cluster_draws_share_labels_and_default_is_zero(family):
    fitted = saved_generalized(family)
    new = np.array([[.5], [.5], [.5]])
    labels = ['a', 'new', 'new']
    known = fitted.random_intercepts(['a', 'a', 'b'], seed=6)
    assert_array_equal(known.draws[..., 0], fitted.samples['u'][..., 0])
    assert_array_equal(known.draws[..., 0], known.draws[..., 1])
    assert_array_equal(known.draws[..., 2], fitted.samples['u'][..., 1])
    random = fitted.random_intercepts(labels, seed=6)
    assert_array_equal(random.draws[..., 1], random.draws[..., 2])
    assert not np.array_equal(random.draws[..., 0], random.draws[..., 1])
    unclustered = fitted.predict(new, scale='latent', seed=9)
    cluster = fitted.predict(new, scale='latent', cluster_new=labels, seed=9)
    assert_allclose(cluster.draws[..., 0] - unclustered.draws[..., 0], fitted.samples['u'][..., 0], atol=1e-13)
    assert_array_equal(cluster.draws[..., 1], cluster.draws[..., 2])
    assert_allclose(unclustered.draws, np.broadcast_to(fitted.samples['beta'], (1, 4, 3)), atol=2e-7)


def test_survival_probability_hazard_boundaries_and_pointwise_records():
    fitted = saved_generalized('survival')
    times = np.array([0., 1., 2., 3., 8.])
    result = fitted.predict_survival([[1.]], times=times, seed=10)
    assert result.draws.shape == (1, 4, 1, 5)
    assert_array_equal(result.draws[..., 0], np.ones((1, 4, 1)))
    assert np.all((result.draws >= 0) & (result.draws <= 1))
    assert np.all(np.diff(result.draws, axis=-1) <= 0)
    hazard = fitted.predict_survival([[1.]], times=times, kind='hazard', seed=10)
    baseline = fitted.samples['baseline_hazard'][0]
    relative = np.exp(fitted.samples['beta'][0, :, 0])
    expected_cumulative = np.column_stack((np.zeros(4), baseline[:, 0], 2*baseline[:, 0],
                                            2*baseline[:, 0]+baseline[:, 1], 2*baseline[:, 0]+6*baseline[:, 1]))
    assert_allclose(result.draws[0, :, 0], np.exp(-relative[:, None]*expected_cumulative), atol=2e-7)
    expected_hazard = relative[:, None] * baseline[:, [0, 0, 0, 1, 1]]
    assert_allclose(hazard.draws[0, :, 0], expected_hazard, atol=2e-7)
    assert len(result.to_records()) == 5
    assert result.to_records()[-1]['time'] == 8
    with pytest.raises(ValueError):
        fitted.predict(kind='observation')
    for invalid in ([], [-1., 0.], [0., 0.], [1., 0.], [np.nan]):
        with pytest.raises(ValueError):
            fitted.predict_survival(times=invalid)


def test_survival_plot_data_and_probability_limits():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    fitted = saved_generalized('survival')
    for kind in ('survival', 'hazard'):
        result = fitted.predict_survival([[1.]], times=[0., 1., 2., 3.], kind=kind, seed=31)
        _, ax = plt.subplots()
        assert km.plot_survival(result, ax=ax) is ax
        assert_array_equal(ax.lines[0].get_xdata(), result.times)
        assert_allclose(ax.lines[0].get_ydata(), result.mean[0])
        if kind == 'survival':
            assert_allclose(ax.get_ylim(), [0, 1])
        else:
            # Point estimates must not imply a step between arbitrary grid times.
            assert ax.lines[0].get_linestyle() in ('None', 'none', '', ' ')
        ax.figure.canvas.draw()
    plt.close('all')
