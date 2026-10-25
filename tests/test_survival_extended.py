"""Independent likelihood and probability identities for extended survival.

These are conditional-on-predictor calculations, not population frailty or
posterior calibration studies. No production duration/integration helper is
used to construct the analytic or numerical quadrature oracles.
"""
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad
from scipy.stats import expon

from kernelmix._survival import (
    extended_survival_loglik,
    prepare_extended_survival,
    survival_curves,
    survival_path_curves,
)

TOL = {'rtol': 2e-12, 'atol': 2e-13}


@pytest.mark.parametrize('event,lower,upper,entry', [
    (1, .8, .8, 0.), (2, .8, .8, .2), (0, .8, np.inf, .2),
    (1, .3, 1.1, 0.), (2, .3, 1.1, .1), (-1, .3, 1.1, .1),
    (1, 0., .7, 0.), (2, 0., .7, 0.), (-1, 0., .7, 0.),
    (1, .2, .9, .2), (-1, .2, .9, .2),
])
def test_constant_competing_cause_likelihoods_match_exponential_law(event, lower, upper, entry):
    data = prepare_extended_survival([lower], [event], upper=[upper], entry=[entry],
                                     n_causes=2, baseline_breaks=[])
    actual = extended_survival_loglik([[0., 0.]], [[2.], [3.]], data)
    if event == 0:
        expected = expon.logsf(lower - entry, scale=1 / 5)
    elif upper == lower:
        expected = expon.logpdf(lower - entry, scale=1 / 5) + math.log([2., 3.][event - 1] / 5)
    else:
        probability = (expon.cdf(upper - entry, scale=1 / 5)
                       - expon.cdf(lower - entry, scale=1 / 5))
        expected = math.log(probability) + (0. if event == -1 else math.log([2., 3.][event - 1] / 5))
    assert_allclose(actual, expected, **TOL)


def test_entry_factor_is_conditional_on_latent_predictors_and_shared_cause_risk():
    # Two cause rates are1 and.75 after applying different predictor effects.
    eta = np.log([[2., .5]])
    baseline = [[.5], [1.5]]
    entered = prepare_extended_survival([1.2], [1], entry=[.4], n_causes=2,
                                        baseline_breaks=[])
    unentered = prepare_extended_survival([1.2], [1], n_causes=2, baseline_breaks=[])
    got_entered = extended_survival_loglik(eta, baseline, entered)
    got_unentered = extended_survival_loglik(eta, baseline, unentered)
    assert_allclose(got_entered, -(1 + .75) * (1.2 - .4), **TOL)
    assert_allclose(got_entered - got_unentered, 1.75 * .4, **TOL)
    # A cause1 event is still at risk from cause2 before the observed event.
    assert not np.isclose(got_entered, -(1.2 - .4))


@pytest.mark.parametrize('event', [1, 2, -1])
def test_piecewise_interval_likelihood_matches_independent_quadrature(event):
    cuts = [.4, 1.]
    baseline = np.array([[.5, 2., .7], [1.2, .3, 1.4]])
    eta = np.array([.2, -.3])
    entry, lower, upper = .1, .3, 1.6

    def cause_rate(time, cause):
        # Literal cases define a second, independent baseline implementation.
        column = 0 if time <= .4 else 1 if time <= 1. else 2
        return baseline[cause, column] * math.exp(eta[cause])

    def total_integral(time):
        return quad(lambda t: cause_rate(t, 0) + cause_rate(t, 1), entry, time,
                    points=[cut for cut in cuts if entry < cut < time],
                    epsabs=1e-12, epsrel=1e-12)[0]

    def density(time):
        hazard = sum(cause_rate(time, cause) for cause in range(2)) if event == -1 else cause_rate(time, event - 1)
        return math.exp(-total_integral(time)) * hazard

    probability, error = quad(density, lower, upper,
                              points=[cut for cut in cuts if lower < cut < upper],
                              epsabs=1e-12, epsrel=1e-12)
    data = prepare_extended_survival([lower], [event], entry=[entry], upper=[upper],
                                     n_causes=2, baseline_breaks=cuts)
    actual = extended_survival_loglik(eta[None, :], baseline, data)
    assert error < 1e-10
    assert_allclose(actual, math.log(probability), rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize('large', [1000., 1e308])
@pytest.mark.parametrize('event', [1, 2, -1])
def test_large_common_predictors_keep_finite_interval_cause_fractions(large, event):
    data = prepare_extended_survival([0.], [event], upper=[1.], n_causes=2,
                                     baseline_breaks=[])
    actual = extended_survival_loglik([[large, large]], [[2.], [3.]], data)
    expected = 0. if event == -1 else math.log([2., 3.][event - 1] / 5)
    assert_allclose(actual, expected, **TOL)


@pytest.mark.parametrize('event', [1, 2, -1])
def test_tiny_interval_probability_remains_finite_on_log_scale(event):
    data = prepare_extended_survival([0.], [event], upper=[.5], n_causes=2,
                                     baseline_breaks=[])
    actual = extended_survival_loglik([[-1000., -1000.]], [[2.], [3.]], data)
    # Rate*width is below representable probability, but its log is finite.
    multiplier = 5. if event == -1 else [2., 3.][event - 1]
    assert_allclose(actual, -1000 + math.log(multiplier * .5), rtol=0, atol=2e-12)


def test_large_hazard_before_interval_is_zero_probability_not_nan():
    data = prepare_extended_survival([.1], [-1], upper=[1.], n_causes=2,
                                     baseline_breaks=[])
    assert extended_survival_loglik([[1000., 1000.]], [[1.], [1.]], data) == -np.inf


@pytest.mark.parametrize('interval', [False, True])
def test_subject_history_splitting_preserves_full_conditional_likelihood(interval):
    baseline = [[.7, 1.1, .3], [1.3, .4, 1.2]]
    options = dict(n_causes=2, baseline_breaks=[1., 2.5])
    one = prepare_extended_survival([3.], [2], entry=[.5],
        upper=[4.] if interval else None, **options)
    history = prepare_extended_survival([1.2, 2., 3.], [0, 0, 2],
        entry=[.5, 1.2, 2.], upper=[np.inf, np.inf, 4.] if interval else None,
        subject=['a', 'a', 'a'], **options)
    eta = np.array([[.2, -.1]])
    assert_allclose(extended_survival_loglik(eta, baseline, one),
                    extended_survival_loglik(np.repeat(eta, 3, axis=0), baseline, history), **TOL)


@pytest.mark.parametrize('entry,time,event', [
    ([0., 1.5], [1., 2.], [0, 1]),  # gap
    ([0., .5], [1., 2.], [0, 1]),  # overlapping risk intervals
    ([1., 0.], [2., 1.], [0, 1]),  # reversed history
    ([0., 1.], [1., 2.], [1, 0]),  # event followed by another risk row
])
def test_invalid_subject_histories_are_rejected_instead_of_reordered(entry, time, event):
    with pytest.raises(ValueError):
        prepare_extended_survival(time, event, entry=entry, subject=['a', 'a'], baseline_breaks=[])


def test_interleaved_subject_rows_keep_original_row_order_and_typed_ids():
    data = prepare_extended_survival([1., 1., 2., 2.], [0, 0, 1, 2],
        entry=[0., 0., 1., 1.], subject=[1, '1', 1, '1'], baseline_breaks=[])
    assert data['subject_levels'] == [1, '1']
    assert_array_equal(data['subject_codes'], [0, 1, 0, 1])
    assert_array_equal(data['time'], [1., 1., 2., 2.])
    assert_array_equal(data['terminal'], [False, False, True, True])


@pytest.mark.parametrize('entry', [0., .5])
def test_equal_constant_causes_have_analytic_survival_and_cumulative_incidence(entry):
    times = np.array([0., .25, .5, 1., 2.])
    survival, cif, hazard = survival_curves([[0., 0.]], [[1.], [1.]], [], times, entry=entry)
    expected = np.exp(-2 * np.maximum(times - entry, 0))
    assert_allclose(survival[0], expected, **TOL)
    assert_allclose(cif[0, :, 0], (1 - expected) / 2, **TOL)
    assert_allclose(cif[0, :, 1], (1 - expected) / 2, **TOL)
    expected_hazard = np.repeat((times >= entry)[:, None], 2, axis=1)
    assert_array_equal(hazard[0], expected_hazard)


def assert_probability_curves(survival, cif):
    assert np.all(survival >= 0) and np.all(survival <= 1)
    assert np.all(cif >= -2e-14) and np.all(cif <= 1 + 2e-14)
    assert np.all(np.diff(survival, axis=1) <= 2e-14)
    assert np.all(np.diff(cif, axis=1) >= -2e-14)
    assert_allclose(survival + cif.sum(axis=-1), np.ones_like(survival), rtol=0, atol=3e-13)


def test_static_and_time_varying_path_curves_preserve_competing_risk_mass():
    times = [0., .25, .5, .75, 1., 1.5, 2.]
    baseline = [[.5, 2., .7], [1.2, .3, 1.4]]
    cuts = [.4, 1.]
    survival, cif, hazard = survival_curves([[.2, -.3], [-.4, .5]], baseline, cuts,
                                           times, entry=[0., .25])
    assert_probability_curves(survival, cif)
    assert hazard.shape == (2, 7, 2)
    levels, survival, cif, hazard = survival_path_curves(
        [[.2, -.3], [-.4, .5], [.1, .2], [.8, -.2]], baseline, cuts, times,
        start=[0., 0., .75, 1.], stop=[.75, 1., 2., 2.], subject=[1, '1', 1, '1'])
    assert levels == [1, '1']
    assert_probability_curves(survival, cif)
    assert hazard.shape == (2, 7, 2)


def test_constant_predictor_path_splits_match_static_curves():
    times = [0., .1, .25, .5, .7, 1., 1.3, 2.]
    eta = np.array([[.2, -.4]])
    baseline, cuts = [[.7, 1., .3], [.2, .5, 1.2]], [.5, 1.]
    static = survival_curves(eta, baseline, cuts, times, entry=.25)
    levels, *path = survival_path_curves(np.repeat(eta, 3, axis=0), baseline, cuts, times,
        start=[.25, .7, 1.3], stop=[.7, 1.3, 2.], subject=['a', 'a', 'a'])
    assert levels == ['a']
    for actual, expected in zip(path, static):
        assert_allclose(actual, expected, **TOL)


def test_hazards_at_baseline_and_covariate_ties_use_right_closed_intervals():
    s, _, hazard = survival_curves([[0.]], [[1., 4.]], [1.], [0., 1., 1.5])
    assert_allclose(s[0], np.exp(-np.array([0., 1., 3.])), **TOL)
    assert_array_equal(hazard[0, :, 0], [1., 1., 4.])
    _, s, _, hazard = survival_path_curves([[0.], [math.log(2)]], [[1.]], [],
        [0., 1., 1.5, 2.], start=[0., 1.], stop=[1., 2.], subject=['a', 'a'])
    assert_allclose(s[0], np.exp(-np.array([0., 1., 2., 3.])), **TOL)
    assert_array_equal(hazard[0, :, 0], [1., 1., 2., 2.])


def test_future_path_values_cannot_change_past_curve_values():
    args = dict(baseline=[[1.]], breaks=[], times=[0., .5, 1.],
                start=[0., 1.], stop=[1., 2.], subject=['a', 'a'])
    first = survival_path_curves([[0.], [math.log(2)]], **args)
    changed_future = survival_path_curves([[0.], [math.log(20)]], **args)
    for actual, expected in zip(first[1:], changed_future[1:]):
        assert_array_equal(actual, expected)


def test_path_coverage_is_required_and_static_baseline_extrapolation_is_explicit():
    with pytest.raises(ValueError, match='beyond'):
        survival_path_curves([[0.]], [[1.]], [], [0., 2.],
                             start=[0.], stop=[1.], subject=['a'])
    s, _, hazard = survival_curves([[0.]], [[1., 2.]], [1.], [3.])
    assert_allclose(s, [[math.exp(-5)]], **TOL)
    assert_array_equal(hazard, [[[2.]]])


@pytest.mark.parametrize('start,stop', [([0., 1.2], [1., 2.]),
                                      ([0., .5], [1., 2.]),
                                      ([1., 0.], [2., 1.]),
                                      ([0., 1.], [1., np.inf])])
def test_path_gaps_overlaps_reversal_and_unspecified_infinite_followup_fail(start, stop):
    with pytest.raises(ValueError):
        survival_path_curves([[0.], [0.]], [[1.]], [], [.5],
                             start=start, stop=stop, subject=['a', 'a'])


def test_exact_event_at_baseline_cut_has_positive_left_bin_risk_duration():
    data = prepare_extended_survival([1.], [1], entry=[.5], baseline_breaks=[1.])
    assert_array_equal(data['event_bin'], [0])
    assert_allclose(data['durations'], [[.5, 0.]])
    actual = extended_survival_loglik([[0.]], [[2., 1e20]], data)
    assert_allclose(actual, math.log(2) - 1, **TOL)


@pytest.mark.parametrize('options', [
    {'time': [0.], 'event': [1]}, {'time': [1.], 'event': [1], 'entry': [1.]},
    {'time': [1.], 'event': [0], 'entry': [1.]},
    {'time': [.5], 'event': [1], 'entry': [.7], 'upper': [1.]},
    {'time': [.5], 'event': [0], 'upper': [1.]},
    {'time': [.5], 'event': [1], 'upper': [np.inf]},
    {'time': [1.], 'event': [1], 'upper': [.5]},
    {'time': [.5], 'event': [-1], 'upper': [.5], 'n_causes': 2},
    {'time': [.5], 'event': [-1], 'upper': [1.]},
    {'time': [.5], 'event': [2], 'n_causes': 1},
    {'time': [.5], 'event': [1], 'n_causes': True},
    {'time': [.5], 'event': [.5]}, {'time': [np.nan], 'event': [1]},
    {'time': [1.], 'event': [1], 'entry': [-1.]},
    {'time': [1.], 'event': [1], 'baseline_breaks': [1., 1.]},
    {'time': [1.], 'event': [1], 'baseline_breaks': [0.]},
])
def test_invalid_censoring_and_entry_endpoints(options):
    with pytest.raises(ValueError):
        prepare_extended_survival(**options)


@pytest.mark.parametrize('eta,baseline', [([[np.nan]], [[1.]]), ([[np.inf]], [[1.]]),
                                       ([[0.]], [[0.]]), ([[0.]], [[-1.]]),
                                       ([[0., 0.]], [[1.]]), ([[0.]], [[np.inf]])])
def test_invalid_likelihood_candidates_return_negative_infinity(eta, baseline):
    data = prepare_extended_survival([1.], [1], baseline_breaks=[])
    assert extended_survival_loglik(eta, baseline, data) == -np.inf


def test_changing_predictors_have_row_specific_risk_and_terminal_cause_density():
    data = prepare_extended_survival([1., 3.], [0, 1], entry=[0., 1.],
                                     subject=['a', 'a'], n_causes=2, baseline_breaks=[])
    eta = np.log([[1., 1.], [2., .5]])
    actual = extended_survival_loglik(eta, [[.7], [1.3]], data)
    expected = -(.7 + 1.3) * 1 + math.log(1.4) - (1.4 + .65) * 2
    assert_allclose(actual, expected, **TOL)


@pytest.mark.parametrize('times', [[], [1., 0.], [0., 0.], [-1., 0.], [np.nan]])
def test_prediction_time_grid_is_explicit_finite_and_chronological(times):
    with pytest.raises(ValueError):
        survival_curves([[0.]], [[1.]], [], times)
    with pytest.raises(ValueError):
        survival_path_curves([[0.]], [[1.]], [], times,
                             start=[0.], stop=[2.], subject=['a'])
