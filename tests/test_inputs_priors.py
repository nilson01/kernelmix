"""Independent input, density, and Gaussian-conditioning checks."""
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import gamma, uniform

from kernelmix import Prior
from kernelmix._input import prediction_matrix, prepare_data
from kernelmix._kernel import (
    conditional_h,
    covariance_components,
    gaussian_kernel,
    psd_factor,
)


class NamedMatrix:
    """DataFrame-like protocol without making pandas a testing dependency."""

    def __init__(self, values, columns):
        self.values = np.asarray(values, dtype=float)
        self.columns = columns

    def __array__(self, dtype=None, copy=None):
        return np.array(self.values, dtype=dtype, copy=True)


def test_complete_cases_use_one_row_mask_and_training_sample_sd():
    y = np.arange(6, dtype=float)
    z = np.column_stack((2 * y, 2 * y + 2))
    x = (y ** 2)[:, None]
    y[1], z[3, 0], x[5, 0] = np.nan, np.nan, np.nan
    originals = [a.copy() for a in (y, z, x)]
    data = prepare_data(y, z, x, missing='drop')
    assert_array_equal(data['rows'], [0, 2, 4])
    assert data['original_n'] == 6
    assert_array_equal(data['y'], [0, 2, 4])
    assert_array_equal(data['Z_original'], [[0, 2], [4, 6], [8, 10]])
    assert_array_equal(data['X_original'], [[0], [4], [16]])
    assert_allclose(data['z_center'], [4, 6])
    assert_allclose(data['z_scale'], [4, 4])
    assert_allclose(data['Z'], [[-1, -1], [0, 0], [1, 1]])
    assert_array_equal(data['X'], [[1, 0], [1, 4], [1, 16]])
    for actual, original in zip((y, z, x), originals):
        assert_array_equal(actual, original)
    data['Z_original'][0, 0] = 100
    assert_array_equal(z, originals[1])


def test_names_are_inferred_and_prediction_order_is_exact():
    z = NamedMatrix([[0, 2], [1, 4], [2, 8]], ['lead', 'cadmium'])
    x = NamedMatrix([[0], [1], [0]], ['age'])
    data = prepare_data([0., 1., 0.], z, x)
    assert data['exposure_names'] == ['lead', 'cadmium']
    assert data['covariate_names'] == ['Intercept', 'age']
    assert_array_equal(prediction_matrix(z, data['exposure_names']), z.values)
    reordered = NamedMatrix(z.values[:, ::-1], ['cadmium', 'lead'])
    with pytest.raises(ValueError, match='order'):
        prediction_matrix(reordered, data['exposure_names'])
    for invalid in (np.empty((0, 2)), [[1]], [[1, np.inf]], [1, 2]):
        with pytest.raises(ValueError):
            prediction_matrix(invalid, ['lead', 'cadmium'])


def test_no_transform_no_intercept_and_constant_exposure_are_explicit():
    z = np.array([[2., 7.], [2., 9.], [2., 11.]])
    data = prepare_data([1., 0., 1.], z, standardize=False, intercept=False)
    assert_array_equal(data['Z'], z)
    assert_array_equal(data['z_center'], [0, 0])
    assert_array_equal(data['z_scale'], [1, 1])
    assert data['X'].shape == (3, 0)
    assert data['covariate_names'] == []
    with pytest.raises(ValueError, match='nonconstant'):
        prepare_data([1., 0., 1.], z)


@pytest.mark.parametrize('changes', [
    {'y': [[0], [1], [0]]}, {'y': [0, 1]}, {'Z': [0, 1, 2]},
    {'Z': np.empty((3, 0))}, {'X': [[1], [2]]}, {'X': np.ones((3, 1))},
    {'Z': [[0], [np.nan], [2]]}, {'Z': [[0], [np.inf], [2]], 'missing': 'drop'},
    {'missing': 'impute'}, {'family': 'poisson'}, {'standardize': 1},
    {'intercept': 'yes'}, {'exposure_names': ['a', 'a']},
    {'exposure_names': ['']}, {'covariate_names': ['extra']},
    {'groups': [None]}, {'groups': [1, 2]},
    {'family': 'binomial', 'y': [0, 0.5, 1]},
])
def test_invalid_data_are_rejected(changes):
    kwargs = {'y': [0., 1., 0.], 'Z': [[0.], [1.], [2.]]}
    kwargs.update(changes)
    with pytest.raises(ValueError):
        prepare_data(**kwargs)


def test_default_priors_are_pinned_and_frozen():
    prior = Prior()
    assert prior.r_distribution == 'invunif'
    assert (prior.r_lower, prior.r_upper) == (0, 100)
    assert (prior.r_mean, prior.r_sd) == (5, 5)
    assert (prior.lambda_mean, prior.lambda_sd) == (10, 10)
    assert (prior.sigma2_shape, prior.sigma2_rate) == (.001, .001)
    assert (prior.inclusion_alpha, prior.inclusion_beta) == (1, 1)
    with pytest.raises(FrozenInstanceError):
        prior.r_upper = 1


@pytest.mark.parametrize('distribution', ['gamma', 'unif', 'invunif'])
def test_prior_density_matches_scipy_with_change_of_variable(distribution):
    prior = Prior(r_distribution=distribution, r_mean=3, r_sd=2,
                  r_lower=.2, r_upper=4)
    for value in (.01, .25, .7, 2., 4., 5., 20.):
        if distribution == 'gamma':
            expected = gamma.logpdf(value, a=2.25, scale=4 / 3)
        elif distribution == 'unif':
            expected = uniform.logpdf(value, loc=.2, scale=3.8)
        else:
            # Density transformation from u=1/r: |du/dr|=1/r².
            expected = uniform.logpdf(1 / value, loc=.2, scale=3.8) - 2 * math.log(value)
        assert_allclose(prior.log_r(value), expected, rtol=1e-13, atol=1e-13)
    for invalid in (0., -1., np.inf, np.nan):
        assert prior.log_r(invalid) == -np.inf


def test_inverse_uniform_sampling_uses_reciprocal_bounds_not_uniform_r():
    prior = Prior(r_distribution='invunif', r_lower=.25, r_upper=2)
    observed_rng = np.random.default_rng(42)
    oracle_rng = np.random.default_rng(42)
    observed = [prior.sample_r(observed_rng) for _ in range(8)]
    expected = 1 / oracle_rng.uniform(.25, 2, size=8)
    assert_array_equal(observed, expected)
    assert np.all((np.asarray(observed) >= .5) & (np.asarray(observed) <= 4))


@pytest.mark.parametrize('changes', [
    {'r_distribution': 'normal'}, {'r_lower': -1}, {'r_upper': 0},
    {'r_lower': 3, 'r_upper': 2}, {'r_upper': np.inf}, {'r_lower': np.nan},
    {'r_mean': 0}, {'r_sd': -1}, {'lambda_mean': np.nan}, {'lambda_sd': 0},
    {'sigma2_shape': 0}, {'sigma2_rate': np.inf},
    {'inclusion_alpha': -1}, {'inclusion_beta': 0}, {'r_mean': True},
])
def test_invalid_prior_parameters(changes):
    with pytest.raises(ValueError):
        Prior(**changes)


def test_kernel_analytic_values_and_covariance_factor():
    z = np.array([[0., 0.], [1., 1.]])
    expected = np.array([[1., 1 / 6], [1 / 6, 1.]])
    assert_allclose(gaussian_kernel(z, r=[math.log(2), math.log(3)]), expected)
    assert_array_equal(gaussian_kernel(z, r=[0, 0]), np.ones((2, 2)))
    kernel, covariance, lower = covariance_components(z, [math.log(2), math.log(3)], 3.)
    assert_allclose(kernel, expected)
    assert_allclose(covariance, [[4, .5], [.5, 4]])
    assert_allclose(lower @ lower.T, covariance)
    assert_array_equal(np.triu(lower, 1), np.zeros((2, 2)))
    assert_allclose(gaussian_kernel(z, [[0., 1.]], [math.log(2), math.log(3)]), [[1/3], [1/2]])


def test_conditional_h_matches_independent_joint_normal_block_conditioning():
    # Build the entire joint law of noisy training outcomes and noiseless hnew.
    # SciPy multivariate-normal covariance blocks, using literal kernel entries,
    # avoid using the production kernel or its covariance helpers as an oracle.
    log2 = math.log(2)
    joint = np.array([
        [6., .25, 2., 2.],
        [.25, 6., 2., 2.],
        [2., 2., 4., 4.],
        [2., 2., 4., 4.],
    ])
    observed = np.array([1., -2.])
    mean, covariance = conditional_h(
        [[0.], [2.]], [[1.], [1.]], observed, np.empty((2, 0)),
        np.empty(0), [log2], 2., 2.,
    )
    # exp(-log(2)*4)=1/16 gives training off-diagonal4/16;
    # a unit new/train distance gives cross-covariance4/2.
    expected_mean = joint[2:, :2] @ np.linalg.solve(joint[:2, :2], observed)
    expected_cov = joint[2:, 2:] - joint[2:, :2] @ np.linalg.solve(joint[:2, :2], joint[:2, 2:])
    assert_allclose(mean, expected_mean, rtol=1e-12, atol=1e-12)
    assert_allclose(covariance, expected_cov, rtol=1e-12, atol=1e-12)
    assert_allclose(covariance[:, 0], covariance[:, 1], atol=1e-13)


def test_psd_draw_factor_handles_rank_deficiency_and_rejects_indefiniteness():
    covariance = np.ones((3, 3)) * 2
    factor = psd_factor(covariance)
    assert_allclose(factor @ factor.T, covariance, atol=1e-13)
    with pytest.raises(np.linalg.LinAlgError, match='indefinite'):
        psd_factor([[1., 2.], [2., 1.]])
    with pytest.raises(ValueError):
        psd_factor([[np.nan]])


@pytest.mark.parametrize('weights', [[-1], [np.nan], [1, 2]])
def test_invalid_kernel_relevance_is_rejected(weights):
    with pytest.raises(ValueError):
        gaussian_kernel([[0.], [1.]], r=weights)


@pytest.mark.parametrize('value', [0., -1., np.nan, np.inf])
def test_covariance_rejects_invalid_kernel_amplitude(value):
    with pytest.raises(ValueError):
        covariance_components([[0.], [1.]], [1.], value)


@pytest.mark.parametrize('units', [1e-200, 1e-12, 1., 1e12, 1e200])
@pytest.mark.parametrize('values', [
    [-3., -2., -1., 1., 2., 3.],
    [-2., -1., 0., 0., 1., 2.],
])
def test_flat_prior_probit_separation_does_not_depend_on_covariate_units(units, values):
    from kernelmix import fit

    # Both complete and quasi separation must be rejected before sampling,
    # even when the intercept and raw covariate differ greatly in magnitude.
    x = units * np.asarray(values)[:, None]
    with pytest.raises(ValueError, match='separated'):
        fit([0., 0., 0., 1., 1., 1.], np.arange(6.)[:, None], x,
            family='binomial', chains=1, draws=2, warmup=0, seed=8)


@pytest.mark.parametrize('units', [1e-200, 1e-12, 1., 1e12, 1e200])
def test_rank_and_nonseparation_diagnostics_preserve_units_and_original_design(units):
    from kernelmix.model import _check_probit_design

    # Interleaved labels cannot be separated by an intercept and a linear
    # covariate. Exercise the diagnostic without running a numerical fit.
    x = units * np.array([-3., -2., -1., 1., 2., 3.])[:, None]
    y = np.array([0., 1., 0., 1., 0., 1.])
    data = prepare_data(y, np.arange(6.)[:, None], x, family='binomial')
    expected_design = np.column_stack((np.ones(6), x))
    assert_array_equal(data['X'], expected_design)
    assert_array_equal(data['X_original'], x)
    assert _check_probit_design(y, data['X']) is None
    assert_array_equal(data['X'], expected_design)


@pytest.mark.parametrize('units', [1e-200, 1e-12, 1., 1e12, 1e200])
def test_column_scaled_rank_check_still_rejects_dependent_design(units):
    x = units * np.arange(6.)
    dependent = np.column_stack((x, 2 * x))
    with pytest.raises(ValueError, match='full column rank'):
        prepare_data(np.arange(6.), np.arange(6.)[:, None], dependent)
