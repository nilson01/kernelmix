"""Prediction oracles from constructed saved states, not fitted MCMC output."""
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from test_results_diagnostics import saved_fit

from kernelmix import BKMRFit, overall_effect, plot_survival
from kernelmix._likelihoods import interval_durations
from kernelmix._prediction import conditional_saved_h


def generalized_saved(family='poisson', clustered=False):
    original = saved_fit(adjusted=False, zero=True)
    data, settings = deepcopy(original.data), deepcopy(original.settings)
    samples = original.posterior_samples()
    settings.update(family=family, clustered=clustered, offset_used=True)
    data['y'] = np.array([0., 1., 2., 1.])
    data['offset'] = np.log(np.arange(1, 5))
    samples['sigma2'][:] = 1
    samples['beta'][..., 0] = np.log(2.)
    samples['h'] = np.zeros((2, 6, 4))
    samples['latent_white'] = np.zeros((2, 6, 4))
    if family == 'negative_binomial':
        samples['dispersion'] = np.ones((2, 6)) * 3
    if clustered:
        data['cluster_codes'] = np.array([0, 0, 1, 1])
        data['cluster_levels'] = ['a', 'b']
        samples['lambda_cluster'] = np.ones((2, 6)) * .7
        samples['u'] = np.broadcast_to(np.log([2., 3.]), (2, 6, 2)).copy()
    if family == 'survival':
        data['y'] = np.arange(1., 5.)
        data['event'] = np.array([1, 0, 1, 1])
        data['baseline_breaks'] = np.array([2.])
        data['durations'] = interval_durations(data['y'], data['baseline_breaks'])
        data['event_bin'] = np.searchsorted(data['baseline_breaks'], data['y'], side='left')
        samples['baseline_hazard'] = np.broadcast_to([.1, .2], (2, 6, 2)).copy()
        data.pop('offset')
        settings['offset_used'] = False
    return BKMRFit(data, samples, settings, {})


@pytest.mark.parametrize('family', ['poisson', 'negative_binomial'])
def test_count_offsets_and_unit_exposure_contrasts(family):
    fit = generalized_saved(family)
    assert_allclose(fit.predict(seed=3).mean, [2, 4, 6, 8])
    with pytest.raises(ValueError, match='offset is required'):
        fit.predict([[1, 2]])
    pred = fit.predict([[1, 2], [3, 4]], offset=np.log([2., 5.]), seed=3)
    assert_allclose(pred.mean, [4, 10])
    # The helper deliberately compares unit-exposure profiles, not training offsets.
    effect = overall_effect(fit, scale='response', seed=3)
    assert_array_equal(effect.draws, np.zeros_like(effect.draws))
    assert effect.metadata['response_quantity'] == 'unit-exposure count mean'
    obs = fit.predict([[1, 2]], offset=[0.], kind='observation', seed=3)
    assert np.all(obs.draws >= 0) and np.all(obs.draws == np.floor(obs.draws))
    with pytest.raises(ValueError, match='require scale'):
        fit.predict([[1, 2]], offset=[0.], kind='observation', scale='latent')


def test_known_and_new_cluster_draws_and_default_zero_policy():
    fit = generalized_saved(clustered=True)
    profiles = [[0, 0], [0, 0], [0, 0]]
    assert_allclose(fit.predict(profiles, offset=[0, 0, 0], seed=2).mean, [2, 2, 2])
    known = fit.predict(profiles, offset=[0, 0, 0], cluster_new=['a', 'b', 'a'], seed=2)
    assert_allclose(known.mean, [4, 6, 4])
    new = fit.predict(profiles, offset=[0, 0, 0], cluster_new=['new', 'a', 'new'], seed=2)
    assert_array_equal(new.draws[..., 0], new.draws[..., 2])
    u = fit.random_intercepts(['a', 'new', 'new'], seed=2)
    assert_allclose(u.draws[..., 0], np.log(2.))
    assert_array_equal(u.draws[..., 1], u.draws[..., 2])
    assert any(row['parameter'] == "u['a']" for row in fit.summary_table())
    assert any(row['parameter'] == "u['a']" for row in fit.diagnostics()['parameters'])


def test_piecewise_survival_and_hazard_values_at_cut_and_zero():
    fit = generalized_saved('survival')
    times = [0., 1., 2., 3., 6.]
    pred = fit.predict_survival([[0, 0]], times=times, seed=1)
    cumulative = np.array([0, .1, .2, .4, 1.])
    expected = np.exp(-2*cumulative)
    assert_allclose(pred.mean[0], expected)
    assert_allclose(pred.lower[0], expected)
    assert_array_equal(pred.draws[..., 0], np.ones((2, 6, 1)))
    hazard = fit.predict_survival([[0, 0]], times=times, kind='hazard', seed=1)
    assert_allclose(hazard.mean[0], [.2, .2, .2, .4, .4])
    assert len(pred.to_records()) == len(times)
    with pytest.raises(ValueError, match='observation'):
        fit.predict([[0, 0]], kind='observation')
    with pytest.raises(ValueError, match='survival fit'):
        generalized_saved().predict_survival(times=times)
    with pytest.raises(ValueError, match='strictly increasing'):
        fit.predict_survival(times=[0, 1, 1])


def test_hazard_plot_does_not_invent_step_locations():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fit = generalized_saved('survival')
    pred = fit.predict_survival([[0, 0]], times=[0, 1, 3, 5], kind='hazard', seed=1)
    ax = plot_survival(pred)
    assert_array_equal(ax.lines[0].get_xdata(), pred.times)
    assert_allclose(ax.lines[0].get_ydata(), pred.mean[0])
    assert ax.lines[0].get_linestyle() == 'None'
    plt.close(ax.figure)


def test_archive_round_trip_and_latent_support_validation(tmp_path):
    fit = generalized_saved('survival', clustered=True)
    path = tmp_path/'survival.npz'
    fit.save(path)
    restored = BKMRFit.load(path)
    options = dict(times=[0, 1, 4], cluster_new=['new'], seed=4)
    assert_array_equal(restored.predict_survival([[0, 0]], **options).draws,
                       fit.predict_survival([[0, 0]], **options).draws)
    samples = fit.posterior_samples()
    samples['h'][0, 0, 0] = 1
    with pytest.raises(ValueError, match='kernel support'):
        BKMRFit(fit.data, samples, fit.settings, {})
    with pytest.raises(ValueError, match='kernel support'):
        conditional_saved_h(np.array([[0.], [1.]]), np.array([[.5]]),
                            np.array([0., 1.]), [0.], 1.)


def test_new_survival_memory_guard_precedes_output_allocation():
    fit = generalized_saved('survival')
    with pytest.raises(ValueError, match='working-memory'):
        fit.predict_survival(np.zeros((100, 2)), times=np.arange(10000.))



def test_archive_validation_does_not_depend_on_eigenvector_signs():
    fit = generalized_saved()
    samples = fit.posterior_samples()
    # Auxiliary white coordinates use the sampler's basis; another legitimate
    # eigensolver may choose opposite signs without changing h or its support.
    samples['h'][:] = 1.
    samples['latent_white'][:] = 0.
    samples['latent_white'][..., -1] = -1/np.sqrt(1.4)
    restored = BKMRFit(fit.data, samples, fit.settings, {})
    assert_allclose(restored.predict(offset=np.zeros(4), seed=1).mean, 2*np.e)



@pytest.mark.parametrize('family', ['poisson', 'negative_binomial', 'survival'])
def test_partial_prior_dictionary_keeps_family_specific_amplitude_defaults(family):
    import kernelmix as km
    kwargs = dict(event=[1, 0, 1, 0], baseline_breaks=[]) if family == 'survival' else {}
    y = [1., 2., 3., 4.] if family == 'survival' else [0, 1, 2, 1]
    result = km.fit(y, np.arange(4.)[:, None], family=family, intercept=False,
                    prior={'r_distribution': 'gamma'}, draws=2, warmup=0, chains=1, seed=82, **kwargs)
    assert result.settings['prior']['lambda_mean'] == 1
    assert result.settings['prior']['lambda_sd'] == .5
