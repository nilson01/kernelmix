"""Independent integral and checker tests for optional joint-posterior validation."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.stats import gamma, norm

SPEC = importlib.util.spec_from_file_location(
    'extension_posteriors', Path(__file__).resolve().parents[1]/'tools/check_extension_posteriors.py'
)
checks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checks)


def test_marginal_gp_prior_matches_independent_gamma_normal_integral():
    def integrated(h):
        return quad(lambda tau: norm.pdf(h, scale=np.sqrt(tau))*gamma.pdf(tau, a=4, scale=.25),
                    0, np.inf, epsabs=1e-12, epsrel=1e-12)[0]

    # Ratio cancels the unnormalised density constant.
    for h in [.2, .8, 1.7, -2.]:
        assert_allclose(checks.h_prior(h)/checks.h_prior(.5), integrated(h)/integrated(.5),
                        rtol=1e-9, atol=1e-11)


def test_joint_nb_target_is_stable_to_domain_and_resolution():
    small = checks.nb_target(401)
    large = checks.nb_target(801, expanded=True)
    # The unused discontinuous CDF grid statistic converges more slowly;
    # comparison of full fitted chains uses smooth marginal and cross moments.
    for key in ['h', 'dispersion', 'h_dispersion']:
        assert_allclose(small[key], large[key], rtol=0, atol=1e-5)


def test_joint_structured_target_is_stable_to_domain_and_hermite_order():
    small = checks.structured_target()
    large = checks.structured_target(801, 48, True)
    for key in small:
        assert_allclose(small[key], large[key], rtol=0, atol=1e-5)
    # The random intercept and common exposure component are confounded in
    # the likelihood but distinguished by their explicitly different priors.
    assert small['h_intercept'] < small['h']*small['random_intercept']


def test_survival_target_retains_baseline_exposure_dependence():
    result, error = checks.survival_target()
    assert error < 1e-7
    assert result['h_baseline'] < result['h']*result['baseline']


@pytest.mark.parametrize('values,target', [([1, 2], 0.), ([[1., np.nan], [2, 3]], 0.),
                                          (np.ones((2, 5)), 0.), (np.arange(10.).reshape(2,5), np.inf)])
def test_checker_rejects_malformed_or_uninformative_draws(values, target):
    with pytest.raises(ValueError):
        checks.summarize(values, target)


def test_checker_does_not_accept_large_monte_carlo_error_as_agreement():
    # Chains concentrated at different locations inflate MCSE but must fail.
    values = np.tile(np.linspace(-.01, .01, 200), (2, 1))
    values[1] += 5.
    result = checks.summarize(values, 2.5)
    assert not result['passed']
    assert result['split_rhat'] > 1.05


def test_checker_rejects_wrong_target_even_with_precise_well_mixed_draws():
    values = np.random.default_rng(821).normal(size=(2, 20000))
    result = checks.summarize(values, 10.)
    assert result['mcse'] < .02*result['sd']
    assert result['split_rhat'] < 1.05
    assert not result['passed']


def test_checker_rejects_imprecise_mean_even_when_target_and_split_chains_agree():
    values = np.tile(np.sin(np.linspace(0, 8*np.pi, 8000)), (2, 1))
    result = checks.summarize(values, float(values.mean()))
    assert result['error'] == 0.
    assert result['split_rhat'] < 1.05
    assert result['mcse'] > .02*result['sd']
    assert not result['passed']


def test_centered_covariance_check_detects_broken_dependence():
    rng = np.random.default_rng(823)
    h = rng.normal(size=(2, 20000))
    phi = -.7*h + np.sqrt(1-.7**2)*rng.normal(size=h.shape)
    original = checks.summarize_covariance(h, phi, 0., 0., -.7)
    assert original['passed']
    permuted = np.stack([rng.permutation(chain) for chain in phi])
    broken = checks.summarize_covariance(h, permuted, 0., 0., -.7)
    assert not broken['passed']
    assert not broken['covariance_magnitude_passed']


def test_raw_draw_artifact_round_trip_and_hash(tmp_path):
    import hashlib
    samples = {'h': np.arange(12.).reshape(2, 6), 'delta': np.ones((2, 6), dtype=bool)}
    record = checks._save_draws(tmp_path, 'case', samples)
    path = Path(record['path'])
    assert record['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as loaded:
        assert sorted(loaded.files) == sorted(samples)
        for name in samples:
            assert_allclose(loaded[name], samples[name])
            assert loaded[name].dtype == samples[name].dtype


def test_relative_covariance_precision_requires_nonzero_target():
    with pytest.raises(ValueError, match='nonzero'):
        checks.summarize_covariance(np.arange(12.).reshape(2, 6), np.arange(12.).reshape(2, 6), 0., 0., 0.)


def test_covariance_checker_rejects_unresolved_small_covariance():
    phase = np.linspace(0, 8*np.pi, 8000)
    left = np.tile(np.sin(phase), (2, 1))
    right = np.tile(np.cos(phase)+.02*np.sin(phase), (2, 1))
    result = checks.summarize_covariance(left, right, float(left.mean()),
                                         float(right.mean()), float((left*right).mean()))
    assert result['covariance_magnitude_passed']
    assert not result['covariance_precision_passed']
    assert not result['passed']
