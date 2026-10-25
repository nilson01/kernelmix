"""Independent exposure-contrast arithmetic and optional plotting contracts.

No MCMC fits or statistical studies: direct saved states exercise integration.
"""
import os
import subprocess
import sys
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import ndtr

from kernelmix import (
    BKMRFit,
    bivariate_response,
    interaction_effects,
    overall_effect,
    plot_overall,
    plot_pips,
    plot_response,
    plot_surface,
    plot_trace,
    single_exposure_effects,
    univariate_response,
)


class StubFit:
    """Known polynomial draws on visibly different original-unit columns.

    h_s(A,B,C) = offset_s + a_s A/10 + b_s B/100 + c_s C/4
                 + t_s (A/10)(B/100).
    Prediction calls are recorded so separate reference sampling cannot pass.
    """

    exposure_names = ['A', 'B', 'C']
    a = np.array([-.4, .1, .3, .5, .7, 1.1])
    b = np.array([1., .7, .3, -.2, .2, -.6])
    c = np.array([.3, -.1, .2, .2, -.3, .4])
    t = np.array([0., .1, .2, .3, .4, .5])
    offset = np.array([-2., -1., 0., .5, 1., 2.])
    adjustment = np.array([-.5, -.2, .1, .2, .4, .8])

    def __init__(self, family='gaussian'):
        self.family = family
        self.data = {
            'Z_original': np.array([[10., 100., -4.], [20., 200., 0.], [30., 300., 4.]]),
            # Deliberately unrelated transformed matrix catches use of wrong units.
            'Z': np.array([[-1., -1., -1.], [0., 0., 0.], [1., 1., 1.]]),
            'X_original': np.array([[2.], [4.], [9.]]),
        }
        self.calls = []

    def latent(self, z):
        z = np.asarray(z)
        a, b, c = z[:, 0] / 10, z[:, 1] / 100, z[:, 2] / 4
        return (self.offset[:, None] + self.a[:, None] * a + self.b[:, None] * b
                + self.c[:, None] * c + self.t[:, None] * a * b)

    def predict_h(self, Znew, **options):
        self.calls.append(('h', np.array(Znew, copy=True), None, options))
        return SimpleNamespace(draws=self.latent(Znew).reshape(2, 3, -1))

    def predict(self, Znew, Xnew, *, kind, scale, **options):
        self.calls.append(('response', np.array(Znew, copy=True),
                           np.array(Xnew, copy=True), dict(options, kind=kind, scale=scale)))
        assert kind == 'mean'
        assert scale == 'response'
        values = self.latent(Znew) + self.adjustment[:, None] * np.asarray(Xnew)[:, 0]
        if self.family == 'binomial':
            values = ndtr(values)
        return SimpleNamespace(draws=values.reshape(2, 3, -1))

    def posterior_samples(self, parameter):
        if parameter == 'lambda':
            return np.array([[1., 2., 4.], [3., 2., 1.]])
        if parameter == 'beta':
            return np.array([[[1., 5.], [2., 6.], [4., 7.]],
                             [[3., 9.], [2., 8.], [1., 7.]]])
        raise ValueError('unknown parameter')

    def inclusion_probabilities(self):
        return [dict(variable='A', pip=.25, group='g1', group_pip=.75, conditional_pip=1/3),
                dict(variable='B', pip=.5, group='g1', group_pip=.75, conditional_pip=2/3),
                dict(variable='C', pip=.2, group='g2', group_pip=.2, conditional_pip=1.)]


@pytest.fixture
def stub():
    return StubFit()


def test_overall_arithmetic_joint_covariance_and_reference_zero(stub):
    result = overall_effect(stub, probability=.8, max_draws=3, seed=127)
    # All raw exposures move a half unit after their displayed unit conversion.
    expected_low = -.5 * (stub.a + stub.b + stub.c) - 1.75 * stub.t
    expected_high = .5 * (stub.a + stub.b + stub.c) + 2.25 * stub.t
    expected = np.column_stack((expected_low, np.zeros(6), expected_high))
    assert_allclose(result.draws.reshape(6, 3), expected, atol=1e-14)
    assert_array_equal(result.draws[..., 1], np.zeros((2, 3)))
    assert_allclose(result.mean, expected.mean(axis=0), atol=1e-14)
    assert_allclose(result.sd, expected.std(axis=0, ddof=1), atol=1e-14)
    assert_allclose(result.lower, np.quantile(expected, .1, axis=0), atol=1e-14)
    assert_allclose(result.upper, np.quantile(expected, .9, axis=0), atol=1e-14)
    assert len(stub.calls) == 1
    assert stub.calls[0][3] == dict(probability=.8, max_draws=3, seed=127)
    assert len(np.unique(stub.calls[0][1], axis=0)) == len(stub.calls[0][1])
    # Difference variance includes covariance; independent reference noise differs.
    raw = stub.latent([[25., 250., 2.], [20., 200., 0.]])
    covariance = np.cov(raw, rowvar=False, ddof=1)
    assert_allclose(result.sd[2] ** 2, covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1])
    assert not np.isclose(result.sd[2] ** 2, covariance[0, 0] + covariance[1, 1])
    records = result.to_records()
    assert len(records) == 3
    assert records[0]['quantile'] == .25
    assert_allclose(records[2]['mean'], expected_high.mean())


@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
@pytest.mark.parametrize('given_x', [None, [1.25]])
def test_response_transforms_profile_draws_before_contrast(family, given_x):
    fit = StubFit(family)
    result = overall_effect(fit, quantiles=[.75], scale='response', X=given_x)
    x = 5. if given_x is None else 1.25
    low = fit.offset + 2 * fit.a + 2 * fit.b + 4 * fit.t + x * fit.adjustment
    high = fit.offset + 2.5 * fit.a + 2.5 * fit.b + .5 * fit.c + 6.25 * fit.t + x * fit.adjustment
    expected = ndtr(high) - ndtr(low) if family == 'binomial' else high - low
    assert_allclose(result.draws.reshape(6), expected, atol=1e-14)
    assert len(fit.calls) == 1
    assert fit.calls[0][0] == 'response'
    assert_allclose(fit.calls[0][2], np.full((len(fit.calls[0][1]), 1), x))
    assert_array_equal(result.metadata['covariates'], [x])
    if family == 'binomial':
        assert not np.allclose(expected, ndtr(high - low))
        assert np.all(np.abs(result.draws) <= 1)


def test_univariate_names_units_background_and_median_center(stub):
    # C is at its 25th percentile (-2), but centering remains all-medians.
    result = univariate_response(stub, 'A', grid=[10., 20., 30.], fixed_quantile=.25)
    # B/100=1.5, C/4=-.5. Reference is (A/10,B/100,C/4)=(2,2,0).
    expected = np.column_stack([
        (u - 2) * stub.a - .5 * stub.b - .5 * stub.c + (1.5 * u - 4) * stub.t
        for u in (1., 2., 3.)
    ])
    assert_allclose(result.draws.reshape(6, 3), expected, atol=1e-14)
    assert [row['variable'] for row in result.locations] == ['A'] * 3
    assert [row['value'] for row in result.locations] == [10., 20., 30.]
    profiles = stub.calls[0][1]
    assert any(np.array_equal(row, [20., 200., 0.]) for row in profiles)
    assert any(np.array_equal(row, [10., 150., -2.]) for row in profiles)
    assert len(stub.calls) == 1


def test_all_univariate_curves_are_joint_and_uncentered_values_are_preserved(stub):
    result = univariate_response(stub, points=3, center=False)
    assert len(stub.calls) == 1
    assert len(result.locations) == 9
    for index, (name, values) in enumerate((('A', [10, 20, 30]), ('B', [100, 200, 300]), ('C', [-4, 0, 4]))):
        selected = result.locations[index * 3:(index + 1) * 3]
        assert [row['variable'] for row in selected] == [name] * 3
        assert [row['value'] for row in selected] == values
    expected_a_first = stub.offset + stub.a + 2 * stub.b + 2 * stub.t
    assert_allclose(result.draws[..., 0].ravel(), expected_a_first)
    assert_array_equal(result.draws[..., 1], result.draws[..., 4])
    assert_array_equal(result.draws[..., 4], result.draws[..., 7])


def test_bivariate_surface_ordering_and_background(stub):
    result = bivariate_response(stub, ('B', 'A'), grids=([100., 300.], [10., 20., 30.]), fixed_quantile=.75)
    assert result.variables == ('B', 'A')
    assert result.shape == (2, 3)
    assert [(row['value1'], row['value2']) for row in result.locations] == [
        (100., 10.), (100., 20.), (100., 30.), (300., 10.), (300., 20.), (300., 30.)]
    expected = np.column_stack([
        (a - 2) * stub.a + (b - 2) * stub.b + .5 * stub.c + (a * b - 4) * stub.t
        for b in (1., 3.) for a in (1., 2., 3.)
    ])
    assert_allclose(result.draws.reshape(6, 6), expected, atol=1e-14)
    assert len(stub.calls) == 1


def test_single_exposure_high_low_signs_at_backgrounds(stub):
    result = single_exposure_effects(stub, fixed_quantiles=[.25, .75])
    expected = np.column_stack([
        stub.a + 1.5 * stub.t, stub.b + 1.5 * stub.t, stub.c,
        stub.a + 2.5 * stub.t, stub.b + 2.5 * stub.t, stub.c,
    ])
    assert_allclose(result.draws.reshape(6, 6), expected, atol=1e-14)
    assert [row['variable'] for row in result.locations] == ['A', 'B', 'C'] * 2
    assert [row['fixed_quantile'] for row in result.locations] == [.25] * 3 + [.75] * 3
    assert len(stub.calls) == 1


def test_pairwise_interaction_cancels_additive_terms_and_has_expected_sign(stub):
    result = interaction_effects(stub)
    expected = np.column_stack((stub.t, np.zeros(6), np.zeros(6)))
    assert_allclose(result.draws.reshape(6, 3), expected, atol=1e-14)
    assert [(row['variable1'], row['variable2']) for row in result.locations] == [('A', 'B'), ('A', 'C'), ('B', 'C')]
    reverse = interaction_effects(StubFit(), variables=['B', 'A'])
    assert_allclose(reverse.draws.ravel(), stub.t, atol=1e-14)
    assert len(stub.calls) == 1


@pytest.mark.parametrize('function,kwargs', [
    (overall_effect, {'quantiles': []}),
    (overall_effect, {'quantiles': [-.1]}),
    (overall_effect, {'reference': np.nan}),
    (overall_effect, {'probability': 1}),
    (overall_effect, {'probability': [0.8]}),
    (overall_effect, {'scale': 'odds'}),
    (overall_effect, {'X': [1]}),
    (overall_effect, {'scale': 'response', 'X': [1, 2]}),
    (overall_effect, {'scale': 'response', 'X': [np.nan]}),
    (univariate_response, {'variable': 'missing'}),
    (univariate_response, {'variable': True}),
    (univariate_response, {'variable': -1}),
    (univariate_response, {'grid': [1, 1]}),
    (univariate_response, {'grid': [2, 1]}),
    (univariate_response, {'grid': [np.nan]}),
    (univariate_response, {'points': True}),
    (univariate_response, {'points': 1}),
    (univariate_response, {'center': 'yes'}),
    (univariate_response, {'fixed_quantile': 2}),
    (bivariate_response, {'variables': ['A', 'A']}),
    (bivariate_response, {'variables': ['A']}),
    (bivariate_response, {'grids': [[1, 2]]}),
    (single_exposure_effects, {'quantiles': [.75, .25]}),
    (single_exposure_effects, {'fixed_quantiles': []}),
    (interaction_effects, {'variables': ['A']}),
    (interaction_effects, {'variables': ['A', 'A']}),
    (interaction_effects, {'quantiles': [0, .5, 1]}),
])
def test_invalid_inputs_raise_before_predicting(function, kwargs, stub):
    with pytest.raises(ValueError):
        function(stub, **kwargs)
    assert not stub.calls


@pytest.mark.parametrize('function,kwargs', [
    (univariate_response, {'points': 10**9}),
    # Only two short 65-entry grids exist before the joint-size guard.
    (bivariate_response, {'points': 65}),
])
def test_memory_guards_reject_huge_output_without_joint_prediction(function, kwargs, stub):
    with pytest.raises(ValueError):
        function(stub, **kwargs)
    assert not stub.calls


def saved_state_fit(family='gaussian', adjustment=False):
    """Small finite state fixture, not a claim of posterior convergence."""
    raw = np.array([[10., 100.], [12., 180.], [20., 140.], [35., 310.]])
    center, scale = raw.mean(axis=0), raw.std(axis=0, ddof=1)
    xraw = np.array([[1.], [2.], [3.], [5.]]) if adjustment else np.empty((4, 0))
    design = np.column_stack((np.ones(4), xraw))
    beta = np.zeros((1, 4, design.shape[1]))
    beta[0, :, 0] = [-.4, -.1, .2, .4]
    if adjustment:
        beta[0, :, 1] = [.1, -.2, .3, -.1]
    y = np.array([0., 1., 0., 1.]) if family == 'binomial' else np.array([.2, -.5, 1., .3])
    data = dict(y=y, Z=(raw-center)/scale, Z_original=raw, X=design, X_original=xraw,
                z_center=center, z_scale=scale, rows=np.arange(4), original_n=4,
                exposure_names=['A', 'B'], covariate_names=['Intercept', 'X'] if adjustment else ['Intercept'], groups=None)
    samples = dict(beta=beta, r=np.array([[[.3, .5], [.4, .4], [.6, .2], [.5, .3]]]),
                   lambda_=np.array([[.5, .8, 1., 1.2]]), sigma2=np.ones((1, 4)),
                   delta=np.ones((1, 4, 2), dtype=bool), pi=np.ones((1, 4)))
    samples['lambda'] = samples.pop('lambda_')
    if family == 'binomial':
        samples['ystar'] = np.array([[[-.3, .5, -.8, .7], [-.2, .8, -.4, 1.],
                                      [-.6, .6, -.5, .4], [-.4, .9, -.2, .3]]])
    settings = dict(family=family, chains=1, draws=4, intercept=True, selection='none', standardize=True)
    return BKMRFit(data, samples, settings, {})


@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
@pytest.mark.parametrize('adjustment', [False, True])
def test_direct_saved_states_integrate_with_joint_prediction(family, adjustment):
    fit = saved_state_fit(family, adjustment)
    before = {key: values.copy() for key, values in fit.samples.items()}
    effects = overall_effect(fit, scale='response', seed=34)
    raw = np.quantile(fit.data['Z_original'], [.25, .5, .75], axis=0)
    mean_x = fit.data['X_original'].mean(axis=0)
    prediction = fit.predict(raw, Xnew=np.tile(mean_x, (3, 1)), kind='mean', scale='response', seed=34)
    expected = prediction.draws - prediction.draws[..., 1:2]
    assert_allclose(effects.draws, expected, atol=1e-13)
    assert_array_equal(effects.draws[..., 1], np.zeros((1, 4)))
    for key, values in before.items():
        assert_array_equal(fit.samples[key], values)
    if family == 'binomial':
        assert np.all((prediction.draws >= 0) & (prediction.draws <= 1))
        assert np.any((prediction.draws > 0) & (prediction.draws < 1))


def test_saved_state_latent_curves_and_surfaces_are_reachable():
    fit = saved_state_fit()
    outputs = [univariate_response(fit, 'A', points=3, seed=9),
               bivariate_response(fit, points=2, seed=9),
               single_exposure_effects(fit, fixed_quantiles=[.5], seed=9),
               interaction_effects(fit, seed=9)]
    assert [output.draws.shape for output in outputs] == [(1, 4, 3), (1, 4, 4), (1, 4, 2), (1, 4, 1)]
    assert all(np.isfinite(output.draws).all() for output in outputs)


@pytest.fixture
def pyplot():
    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg', force=True)
    import matplotlib.pyplot as plt
    yield plt
    plt.close('all')


def test_public_import_is_lazy_for_optional_matplotlib():
    # A fresh process distinguishes package imports from pytest's plot fixtures.
    code = "import sys, kernelmix; assert not any(n == 'matplotlib' or n.startswith('matplotlib.') for n in sys.modules)"
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join(map(str, sys.path))
    subprocess.run([sys.executable, '-c', code], check=True, env=environment, timeout=30)


def test_trace_artists_use_saved_draws_and_explicit_parameter_indices(pyplot, stub):
    _, ax = pyplot.subplots()
    assert plot_trace(stub, 'beta', indices=1, ax=ax) is ax
    assert len(ax.lines) == 2
    for chain, line in enumerate(ax.lines):
        assert_array_equal(line.get_xdata(), [1, 2, 3])
        assert_array_equal(line.get_ydata(), stub.posterior_samples('beta')[chain, :, 1])
    assert ax.lines[0].get_linestyle() != ax.lines[1].get_linestyle()
    with pytest.raises(ValueError):
        plot_trace(stub, 'beta')
    with pytest.raises(ValueError):
        plot_trace(stub, 'beta', indices=-1)
    with pytest.raises(ValueError):
        plot_trace(stub, indices=0)


def test_pips_positions_probabilities_and_group_shapes(pyplot, stub):
    ax = plot_pips(stub)
    assert_allclose(ax.get_xlim(), [0, 1])
    expected = [[.25, .5, .2], [.75, .75, .2], [1/3, 2/3, 1.]]
    assert len(ax.collections) == 3
    for collection, positions in zip(ax.collections, expected):
        assert_allclose(collection.get_offsets()[:, 0], positions)
    shapes = [collection.get_paths()[0].vertices for collection in ax.collections]
    assert shapes[0].shape != shapes[1].shape or not np.array_equal(shapes[0], shapes[1])
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ['A', 'B', 'C']


def test_response_and_overall_artists_show_estimates_and_intervals(pyplot, stub):
    curve = univariate_response(stub, 'A', grid=[10., 20., 30.])
    ax = plot_response(curve)
    assert_array_equal(ax.lines[0].get_xdata(), [10., 20., 30.])
    assert_allclose(ax.lines[0].get_ydata(), curve.mean)
    vertices = ax.collections[0].get_paths()[0].vertices
    for x, low, high in zip([10., 20., 30.], curve.lower, curve.upper):
        at_x = vertices[np.isclose(vertices[:, 0], x), 1]
        assert_allclose([at_x.min(), at_x.max()], [low, high])
    effect = overall_effect(stub, quantiles=[.75, .25, .5])
    ax = plot_overall(effect)
    assert_array_equal(ax.lines[0].get_xdata(), [.25, .5, .75])
    assert_allclose(ax.lines[0].get_ydata(), effect.mean[[1, 2, 0]])
    for segment, location, low, high in zip(ax.collections[0].get_segments(), effect.locations, effect.lower, effect.upper):
        assert_allclose(segment, [[location['quantile'], low], [location['quantile'], high]])
    assert any('80' in item.get_text() for item in plot_overall(overall_effect(stub, probability=.8)).get_legend().get_texts())


def test_surface_artist_preserves_grid_orientation_and_zero_center(pyplot, stub):
    surface = bivariate_response(stub, grids=([10., 30.], [100., 200., 300.]))
    ax = plot_surface(surface)
    mesh = ax.collections[0]
    assert_allclose(np.asarray(mesh.get_array()).reshape(3, 2), surface.mean.reshape(2, 3).T)
    assert_allclose(mesh.norm(0), .5)
    assert len(ax.figure.axes) == 2
    assert ax.get_xlabel() and ax.get_ylabel()
    assert ax.figure.axes[1].get_ylabel()


def test_plots_return_supplied_axes_without_display_save_or_style_changes(pyplot, stub, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('display/save must be controlled by caller')
    monkeypatch.setattr(pyplot, 'show', forbidden)
    monkeypatch.setattr(pyplot, 'savefig', forbidden)
    from matplotlib.figure import Figure
    monkeypatch.setattr(Figure, 'savefig', forbidden)
    curve = univariate_response(stub, 'A', points=3)
    overall = overall_effect(stub)
    surface = bivariate_response(stub, points=2)
    specifications = [(plot_trace, stub), (plot_pips, stub), (plot_response, curve),
                      (plot_overall, overall), (plot_surface, surface)]
    baseline = deepcopy(dict(pyplot.rcParams))
    for function, argument in specifications:
        figure, ax = pyplot.subplots()
        assert function(argument, ax=ax) is ax
        figure.canvas.draw()
    assert dict(pyplot.rcParams) == baseline


@pytest.mark.parametrize('function,producer', [
    (plot_response, overall_effect),
    (plot_overall, univariate_response),
    (plot_surface, overall_effect),
])
def test_plot_rejects_incompatible_result_kinds(pyplot, stub, function, producer):
    with pytest.raises(ValueError):
        function(producer(stub))


def test_equal_quantile_and_collapsed_exposure_contrasts_are_exact_zero(stub):
    same = single_exposure_effects(stub, quantiles=(.5, .5))
    assert_array_equal(same.draws, np.zeros((2, 3, 9)))
    stub.data['Z_original'][:, 2] = 1.
    collapsed = single_exposure_effects(stub, fixed_quantiles=[.5])
    assert_array_equal(collapsed.draws[..., 2], np.zeros((2, 3)))
    interactions = interaction_effects(stub, variables=['A', 'C'])
    assert_array_equal(interactions.draws, np.zeros((2, 3, 1)))


@pytest.mark.parametrize('bad_draws', [np.ones((2, 2)), np.full((2, 3, 3), np.nan)])
def test_prediction_boundary_rejects_malformed_or_nonfinite_joint_draws(stub, bad_draws):
    stub.predict_h = lambda *args, **kwargs: SimpleNamespace(draws=bad_draws)
    with pytest.raises(ValueError):
        overall_effect(stub)


def test_missing_retained_training_values_rejected(stub):
    stub.data['Z_original'][0, 0] = np.nan
    with pytest.raises(ValueError):
        overall_effect(stub)
    assert not stub.calls
