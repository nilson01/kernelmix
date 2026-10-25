"""Independent kernel, finite-feature and Gaussian block-covariance checks."""
import tracemalloc
from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import sqrtm
from scipy.stats import multivariate_normal

import kernelmix._basis as basis_module
from kernelmix._basis import basis_matrix, basis_predict, kernel_matrix, prepare_basis
from kernelmix._structure import correlation_blocks, gaussian_loglik, residual_prediction


def scalar_kernel(a, b, r, kernel, alpha=1.):
    distance = sum(weight*(float(x)-float(y))**2 for x, y, weight in zip(a, b, r))
    if kernel == 'gaussian':
        return np.exp(-distance)
    if kernel == 'matern32':
        scaled = np.sqrt(6*distance)
        return (1+scaled)*np.exp(-scaled)
    return (1+distance/alpha)**(-alpha)


def oracle_matrix(a, b, r, kernel, alpha=1.):
    return np.array([[scalar_kernel(x, y, r, kernel, alpha) for y in b] for x in a])


@pytest.mark.parametrize('kernel', ['gaussian', 'matern32', 'rational_quadratic'])
def test_exact_kernel_formulas_symmetry_and_exposure_exclusion(kernel):
    z = np.array([[0., 2.], [.5, -1.], [2., .7]])
    new = np.array([[.2, 3.], [1.1, -2.]])
    r, alpha = [.7, .3], 1.7
    expected = oracle_matrix(z, new, r, kernel, alpha)
    assert_allclose(kernel_matrix(z, new, r, kernel=kernel, alpha=alpha), expected, rtol=2e-14, atol=2e-15)
    gram = kernel_matrix(z, r=r, kernel=kernel, alpha=alpha)
    assert_array_equal(np.diag(gram), np.ones(3))
    assert_allclose(gram, gram.T, atol=0)
    assert np.linalg.eigvalsh(gram).min() >= -1e-13
    assert_array_equal(kernel_matrix(z, r=[0, 0], kernel=kernel, alpha=alpha), np.ones((3, 3)))
    excluded = kernel_matrix([[.5, -100.], [.5, 200.]], r=[.7, 0], kernel=kernel)
    assert_array_equal(excluded, np.ones((2, 2)))


def test_rational_quadratic_limit_approaches_existing_gaussian():
    z = np.array([[0.], [.3], [1.]])
    gaussian = oracle_matrix(z, z, [1.], 'gaussian')
    errors = [np.max(np.abs(kernel_matrix(z, r=[1.], kernel='rational_quadratic', alpha=alpha)-gaussian))
              for alpha in (1., 100., 10000., 1e8)]
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < 3e-9


class FixedSpectrum(np.random.Generator):
    """Controlled spectral ingredients expose shared scale across coordinates."""
    def __init__(self):
        super().__init__(np.random.PCG64(7))
        self.calls = []

    def normal(self, size=None):
        self.calls.append(('normal', size))
        assert size == (2, 2)
        return np.array([[1., 2.], [3., 4.]])

    def chisquare(self, df, size=None):
        self.calls.append(('chisquare', df, size))
        assert df == 3 and size == 2
        return np.array([3., 12.])

    def gamma(self, shape, scale=1., size=None):
        self.calls.append(('gamma', shape, scale, size))
        assert shape == 2.5 and scale == 1/2.5 and size == 2
        return np.array([4., .25])


@pytest.mark.parametrize('kernel,scales', [('matern32', [1., .5]), ('rational_quadratic', [2., .5])])
def test_multivariate_spectral_scale_is_shared_per_frequency(kernel, scales):
    z = np.array([[.2, -.4], [.6, .1]])
    rng = FixedSpectrum()
    prepared = prepare_basis(z, approximation='rff', kernel=kernel, alpha=2.5, n_components=2, rng=rng)
    # Student-t3 and RQ mixtures scale an entire frequency vector, not each
    # coordinate independently. The Gaussian components are fixed above.
    frequency = np.array([[1., 2.], [3., 4.]])*np.array(scales)[:, None]*np.sqrt([.8, 1.8])
    phase = z @ frequency.T
    expected = np.column_stack((np.cos(phase), np.sin(phase)))/np.sqrt(2)
    actual = basis_matrix(z, [.4, .9], prepared)
    assert_allclose(actual, expected, atol=2e-15)
    assert_array_equal(prepared['frequency_scales'], scales)
    assert len(rng.calls) == 2


@pytest.mark.parametrize('kernel', ['gaussian', 'matern32', 'rational_quadratic'])
def test_frozen_rff_features_and_zero_relevance_are_stable(kernel):
    z = np.array([[.2, -1.], [.2, 7.], [.7, .3]])
    prepared = prepare_basis(z, approximation='rff', kernel=kernel, n_components=32,
                             rng=np.random.default_rng(92))
    frozen = deepcopy(prepared)
    first = basis_matrix(z, [.7, .2], prepared)
    basis_matrix(z, [.3, 1.1], prepared)
    assert_array_equal(basis_matrix(z, [.7, .2], prepared), first)
    assert_array_equal(prepared['base_frequencies'], frozen['base_frequencies'])
    assert_array_equal(prepared['frequency_scales'], frozen['frequency_scales'])
    assert_allclose(np.sum(first**2, axis=1), np.ones(3), atol=5e-16)
    excluded = basis_matrix(z, [.7, 0.], prepared)
    assert_array_equal(excluded[0], excluded[1])
    zero = basis_matrix(z, [0., 0.], prepared)
    assert_array_equal(zero, np.tile(np.r_[np.ones(32), np.zeros(32)]/np.sqrt(32), (3, 1)))
    weights = np.linspace(-1., 1., 64)
    rng = np.random.default_rng(71)
    before = deepcopy(rng.bit_generator.state)
    prediction = basis_predict(z, [.7, 0.], prepared, weights, 1.3, rng)
    assert rng.bit_generator.state == before
    assert prediction[0] == prediction[1]
    assert_allclose(prediction, np.sqrt(1.3)*(excluded@weights), atol=2e-15)


@pytest.mark.parametrize('kernel', ['gaussian', 'matern32', 'rational_quadratic'])
def test_full_distinct_pp_knots_reproduce_parent_training_covariance(kernel):
    z = np.array([[-1., -.3], [0., .2], [.7, -.4], [1.5, .6]])
    prepared = prepare_basis(z, approximation='pp', kernel=kernel, knots=z)
    features = basis_matrix(z, [.6, .3], prepared)
    expected = oracle_matrix(z, z, [.6, .3], kernel)
    assert_allclose(features@features.T, expected, rtol=2e-12, atol=2e-12)
    assert prepared['residual_correction'] is False


def test_pp_covariance_is_the_projected_model_without_parent_residual():
    knots = np.array([[-1.], [1.]])
    z = np.array([[0.], [.3]])
    prepared = prepare_basis(z, approximation='pp', knots=knots)
    features = basis_matrix(z, [.7], prepared)
    knot_cov = oracle_matrix(knots, knots, [.7], 'gaussian')
    cross = oracle_matrix(z, knots, [.7], 'gaussian')
    expected = cross @ np.linalg.solve(knot_cov, cross.T)
    assert_allclose(features@features.T, expected, rtol=1e-13, atol=1e-13)
    assert np.all(np.diag(expected) < .8)
    anchor = np.array([.2, -.4])
    expected_mean = cross @ np.linalg.solve(knot_cov, anchor)
    rng = np.random.default_rng(68)
    before = deepcopy(rng.bit_generator.state)
    predicted = basis_predict(z, [.7], prepared, np.zeros(2), 1., rng, training_h=anchor)
    assert_allclose(predicted, expected_mean, atol=2e-14)
    assert rng.bit_generator.state == before
    # With weights, independently constructed symmetric sqrt gives knot values.
    weights = np.array([.3, -.7])
    anchors = sqrtm(knot_cov).real @ weights
    assert_allclose(basis_predict(z, [.7], prepared, weights, 1., rng),
                    cross @ np.linalg.solve(knot_cov, anchors), atol=2e-14)


class Directions(np.random.Generator):
    def __init__(self, direction):
        super().__init__(np.random.PCG64(0))
        self.direction = np.asarray(direction)

    def normal(self, size=None):
        assert size == len(self.direction)
        return self.direction.copy()


def test_dense_joint_conditional_gp_moments_and_anchor_identity():
    train = np.array([[-1.], [1.]])
    new = np.array([[-.3], [.4]])
    r, amplitude = [.8], 1.7
    prepared = prepare_basis(train)
    anchor = np.array([.3, -.2])
    ktt = oracle_matrix(train, train, r, 'gaussian')
    knt = oracle_matrix(new, train, r, 'gaussian')
    knn = oracle_matrix(new, new, r, 'gaussian')
    mean = knt @ np.linalg.solve(ktt, anchor)
    covariance = amplitude*(knn-knt@np.linalg.solve(ktt, knt.T))
    # These four directions have population covariance I exactly. This checks
    # joint conditional covariance without a random Monte Carlo tolerance.
    directions = np.sqrt(2)*np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    draws = np.array([basis_predict(new, r, prepared, np.zeros(2), amplitude,
                                   Directions(direction), training_h=anchor)
                      for direction in directions])
    assert_allclose(draws.mean(axis=0), mean, atol=2e-13)
    assert_allclose(np.cov(draws, rowvar=False, ddof=0), covariance, atol=2e-12)
    exact = basis_predict(train, r, prepared, np.zeros(2), amplitude,
                          np.random.default_rng(5), training_h=anchor)
    assert_array_equal(exact, anchor)
    constant = basis_predict(new, [0.], prepared, np.zeros(2), 1.,
                             np.random.default_rng(5), training_h=[.4, .4])
    assert_array_equal(constant, [.4, .4])
    with pytest.raises(ValueError):
        basis_predict(new, [0.], prepared, np.zeros(2), 1., np.random.default_rng(5), training_h=[.4, .5])


def test_automatic_pp_knots_are_deterministic_distinct_and_bounded():
    z = np.array([[0.], [0.], [1.], [2.], [2.]])
    first = prepare_basis(z, approximation='pp', n_components=128)
    second = prepare_basis(z, approximation='pp', n_components=128)
    assert_array_equal(first['knots'], second['knots'])
    assert len(first['knots']) == 3
    assert_array_equal(first['knots'], [[0.], [2.], [1.]])


@pytest.mark.parametrize('approximation', ['pp', 'rff'])
def test_approximate_paths_do_not_allocate_training_square_arrays(approximation, monkeypatch):
    n = 2048
    x = np.linspace(-1., 1., n)
    z = np.column_stack((x, np.sin(3*x)))
    original_kernel = basis_module.kernel_matrix
    observed_shapes = []

    def guarded_kernel(left, right=None, *args, **kwargs):
        shape = (len(left), len(left) if right is None else len(right))
        observed_shapes.append(shape)
        assert shape != (n, n), 'approximation attempted a full training Gram matrix'
        return original_kernel(left, right, *args, **kwargs)

    monkeypatch.setattr(basis_module, 'kernel_matrix', guarded_kernel)
    for name in ('zeros', 'ones', 'empty', 'full'):
        original = getattr(basis_module.np, name)

        def guarded_allocation(shape, *args, _original=original, **kwargs):
            if isinstance(shape, tuple):
                assert shape != (n, n), 'approximation attempted a full training square allocation'
            return _original(shape, *args, **kwargs)

        monkeypatch.setattr(basis_module.np, name, guarded_allocation)
    tracemalloc.start()
    try:
        prepared = prepare_basis(z, approximation=approximation, n_components=16,
                                 rng=np.random.default_rng(2))
        features = basis_matrix(z, [.4, .2], prepared)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert features.shape == (n, 16 if approximation == 'pp' else 32)
    # Full n² float64 storage is32MiB; allow16MiB for the bounded feature path.
    assert peak < n*n*8//2
    if approximation == 'pp':
        assert observed_shapes and all(min(shape) <= 16 for shape in observed_shapes)
    else:
        assert not observed_shapes


@pytest.mark.parametrize('kwargs', [
    {'kernel': 'unknown'}, {'alpha': 0}, {'alpha': np.nan},
    {'n_components': 0}, {'n_components': True}, {'n_components': 1.5},
    {'approximation': 'unknown'}, {'approximation': 'rff', 'knots': [[0.]]},
    {'approximation': 'rff', 'rng': 3},
])
def test_basis_invalid_settings_fail(kwargs):
    with pytest.raises(ValueError):
        prepare_basis([[0.], [1.]], **kwargs)


def test_basis_allocation_guards_fail_before_large_work():
    with pytest.raises(ValueError):
        prepare_basis(np.arange(10000.)[:, None])
    with pytest.raises(ValueError):
        prepare_basis([[0.], [1.]], approximation='rff', n_components=10**12)
    with pytest.raises(ValueError):
        kernel_matrix([[0.]], r=[-1.])
    with pytest.raises(ValueError):
        kernel_matrix([[0.]], [[0., 1.]])


@pytest.mark.parametrize('kind,parameter', [(None, None), ('ar1', -.4), ('exponential', 1.3)])
def test_normalized_heteroscedastic_gaussian_density_matches_scipy(kind, parameter):
    codes = np.array([0, 1, 0, 1])
    times = np.array([0., 1., 2., 4.])
    y = np.array([.3, 2., -1., .7])
    eta = np.array([.1, .4, -.2, .9])
    sigma2 = .7
    logvar = np.log([1., 4., .25, 2.])
    correlation = np.eye(4)
    if kind is not None:
        for i in range(4):
            for j in range(4):
                if codes[i] == codes[j]:
                    gap = abs(times[i]-times[j])
                    correlation[i, j] = parameter**gap if kind == 'ar1' else np.exp(-gap/parameter)
    std = np.sqrt(sigma2*np.exp(logvar))
    covariance = std[:, None]*correlation*std[None, :]
    expected = multivariate_normal.logpdf(y, mean=eta, cov=covariance)
    blocks = correlation_blocks(codes, times, kind, parameter)
    assert_allclose(gaussian_loglik(y, eta, sigma2, logvar, blocks), expected, atol=3e-13)
    if kind is None:
        assert blocks == []


def test_fixed_block_gaussian_density_and_zero_residual_at_tiny_variance():
    codes = np.array([0, 1, 0])
    correlation = np.array([[1., 0., .3], [0., 1., 0.], [.3, 0., 1.]])
    blocks = correlation_blocks(codes, kind='fixed', matrix=correlation)
    y, eta, logvar = np.array([1., -.2, .4]), np.array([0., .1, .2]), np.log([4., 9., 1.])
    std = np.exp(logvar/2)
    expected = multivariate_normal.logpdf(y, mean=eta, cov=std[:, None]*correlation*std[None, :])
    assert_allclose(gaussian_loglik(y, eta, 1., logvar, blocks), expected, atol=2e-13)
    expected_zero = 1000-.5*np.log(2*np.pi)
    assert_allclose(gaussian_loglik([0.], [0.], 1., [-2000.], []), expected_zero, atol=1e-12)
    assert gaussian_loglik([0.], [np.inf], 1., [0.], []) == -np.inf
    assert gaussian_loglik([0.], [0.], -1., [0.], []) == -np.inf


def residual_case(**updates):
    case = dict(y=[2.], eta_train=[0.], sigma2=1., logvar_train=np.log([4.]),
                logvar_new=np.log([9.]), train_codes=np.array([0]), new_codes=np.array([0]),
                train_time=[0.], new_time=[1.], kind='ar1', parameter=.5,
                rng=np.random.default_rng(13))
    case.update(updates)
    return residual_prediction(**case)


def test_heteroscedastic_residual_conditioning_has_cross_scale_factor():
    # Training SD2, new SD3, correlation.5 and training residual2:
    # mean=(3*2*.5)/4*2=1.5; variance=9-(3*2*.5)^2/4=6.75.
    mean, covariance = residual_case()
    assert_allclose(mean, [1.5], atol=2e-14)
    assert_allclose(covariance, [[6.75]], atol=2e-14)
    mean_only, no_noise = residual_case(observation=False)
    assert_allclose(mean_only, [1.5], atol=2e-14)
    assert_array_equal(no_noise, [[0.]])
    at_training, exact_cov = residual_case(new_time=[0.])
    assert_allclose(at_training, [3.], atol=2e-14)
    assert_array_equal(exact_cov, [[0.]])


def test_known_and_new_cluster_residual_blocks_and_rng_policy():
    rng = np.random.default_rng(13)
    before = deepcopy(rng.bit_generator.state)
    mean, covariance = residual_case(logvar_new=np.log([9., 9., 9.]),
                                     new_codes=np.array([0, 1, 1]), new_time=[1., 1., 2.], rng=rng)
    assert_allclose(mean, [1.5, 0., 0.], atol=2e-14)
    assert_allclose(covariance, [[6.75, 0., 0.], [0., 9., 4.5], [0., 4.5, 9.]], atol=3e-14)
    assert rng.bit_generator.state == before
    duplicate_mean, duplicate_cov = residual_case(logvar_new=np.log([9., 9.]),
                                                 new_codes=np.array([0, 0]), new_time=[1., 1.])
    assert_allclose(duplicate_mean, [1.5, 1.5], atol=2e-14)
    assert_allclose(duplicate_cov, np.full((2, 2), 6.75), atol=3e-14)


def test_fixed_custom_cross_conditioning_and_independent_default():
    mean, covariance = residual_case(kind='fixed', parameter=None, matrix=[[1.]],
                                     train_time=None, new_time=None, cross=[[.5]], new_matrix=[[1.]])
    assert_allclose(mean, [1.5], atol=2e-14)
    assert_allclose(covariance, [[6.75]], atol=2e-14)
    independent_mean, independent_cov = residual_case(kind='fixed', parameter=None, matrix=[[1.]],
                                                      train_time=None, new_time=None)
    assert_array_equal(independent_mean, [0.])
    assert_allclose(independent_cov, [[9.]], atol=2e-14)
    with pytest.raises(ValueError):
        residual_case(kind='fixed', parameter=None, matrix=[[1.]], cross=[[1.2]], new_matrix=[[1.]])
    with pytest.raises(ValueError):
        residual_case(kind='fixed', parameter=None, matrix=[[1.]], new_codes=np.array([1]), cross=[[.5]])


def test_near_psd_new_process_allowed_but_material_indefiniteness_rejected():
    mean, covariance = residual_case(kind='fixed', parameter=None, matrix=[[1.]],
                                     logvar_new=np.zeros(2), new_codes=np.array([0, 0]),
                                     cross=[[0.], [0.]], new_matrix=[[1., 1.+1e-15], [1.+1e-15, 1.]])
    assert_array_equal(mean, [0., 0.])
    assert np.linalg.eigvalsh(covariance).min() >= -1e-14
    bad = np.full((3, 3), -.9)
    np.fill_diagonal(bad, 1.)
    with pytest.raises(ValueError):
        residual_case(kind='fixed', parameter=None, matrix=[[1.]], logvar_new=np.zeros(3),
                      new_codes=np.array([0, 0, 0]), new_matrix=bad)


@pytest.mark.parametrize('kwargs', [
    dict(codes=None, times=[0, 1], kind='ar1', parameter=.5),
    dict(codes=[0, 0], times=[0, .5], kind='ar1', parameter=.5),
    dict(codes=[0, 0], times=[1, 1], kind='ar1', parameter=.5),
    dict(codes=[0, 0], times=[1, 1], kind='exponential', parameter=1),
    dict(codes=[0, 0], times=[0, 1], kind='ar1', parameter=1),
    dict(codes=[0, 0], times=[0, 1], kind='ar1', parameter=-1),
    dict(codes=[0, 0], times=[0, 1], kind='exponential', parameter=0),
    dict(codes=[0, 0], kind='fixed', matrix=[[1, 1], [1, 1]]),
    dict(codes=[0, 1], kind='fixed', matrix=[[1, .2], [.2, 1]]),
    dict(codes=[0, 0], kind='fixed', matrix=[[1, .2], [.3, 1]]),
    dict(codes=[0, 0], kind='fixed', matrix=[[2, 0], [0, 1]]),
])
def test_invalid_correlation_inputs_fail(kwargs):
    with pytest.raises(ValueError):
        correlation_blocks(**kwargs)


@pytest.mark.parametrize('kernel', ['gaussian', 'matern32', 'rational_quadratic'])
def test_bounded_spectral_average_matches_target_covariance(kernel):
    # Sixteen frozen64-pair bases give1024 spectral draws, with no model fit.
    # This checks the selected spectral law in addition to controlled scales.
    z = np.array([[.2, -.15], [.6, .1]])
    expected = scalar_kernel(z[0], z[1], [.4, .8], kernel, alpha=1.7)
    values = []
    for seed in range(1280, 1296):
        prepared = prepare_basis(z, approximation='rff', kernel=kernel, alpha=1.7,
                                 n_components=64, rng=np.random.default_rng(seed))
        feature = basis_matrix(z, [.4, .8], prepared)
        values.append(feature[0]@feature[1])
    assert abs(np.mean(values)-expected) < .04


def test_residual_memory_guard_and_malformed_cross_inputs():
    with pytest.raises(ValueError):
        residual_case(logvar_new=np.zeros(10000))
    with pytest.raises(ValueError):
        residual_case(kind='fixed', parameter=None, matrix=[[1.]], cross=[[.5, .2]])
    with pytest.raises(ValueError):
        residual_case(kind='fixed', parameter=None, matrix=[[1.]], new_matrix=[[2.]])
    with pytest.raises(ValueError):
        residual_case(cross=[[.5]])
