"""Cluster identity, independent block-normal, and installed-R state checks.

Fixtures are constructed posterior states, never MCMC fits. R is not required
when running these tests. Arithmetic tolerances do not imply sampler calibration.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from kernelmix import BKMRFit, Prior, fit
from kernelmix._clusters import canonical_label, cluster_mask, encode_clusters
from kernelmix._input import prepare_data
from kernelmix._kernel import conditional_h, covariance_components

HERE = Path(__file__).parent
FIXTURE = json.loads((HERE / 'fixtures' / 'r_cluster_reference.json').read_text())
TOL = {'rtol': 2e-11, 'atol': 2e-12}


def state_fit(case, selection='component', clustered=True):
    z, x = np.asarray(case['Z']), np.asarray(case['X'])
    data = dict(y=np.asarray(case['y']), Z=z, Z_original=z, X=x,
                X_original=x[:, 1:], z_center=np.zeros(2), z_scale=np.ones(2),
                rows=np.arange(len(z)), original_n=len(z),
                exposure_names=['z1', 'z2'], covariate_names=['intercept', 'x1'],
                groups=[1, 1] if selection == 'group' else None)
    samples = {key: np.asarray(case[key])[None, ...] for key in
               ('beta', 'r', 'lambda', 'sigma2', 'delta')}
    samples['pi'] = np.full((1, 3), .5)
    if clustered:
        data.update(cluster_codes=np.asarray(case['cluster_codes']), cluster_levels=[0, 1])
        samples['lambda_cluster'] = np.asarray(case['lambda_cluster'])[None, ...]
    if case['family'] == 'binomial':
        samples['ystar'] = np.asarray(case['ystar'])[None, ...]
    settings = dict(family=case['family'], chains=1, draws=3, standardize=False,
                    intercept=True, selection=selection, clustered=clustered)
    return BKMRFit(data, samples, settings, {})


def test_typed_identity_and_first_retained_occurrence():
    labels = [1, '1', None, 'other', 1, np.nan, '1']
    assert canonical_label(1) != canonical_label('1')
    assert canonical_label(np.int64(1)) == canonical_label(1)
    assert_array_equal(cluster_mask(labels, 7), [True, True, False, True, True, False, True])
    codes, levels = encode_clusters(labels, 7, rows=np.array([1, 3, 4, 6]))
    assert levels == ['1', 'other', 1]
    assert type(levels[-1]) is int
    assert_array_equal(codes, [0, 1, 2, 0])
    assert encode_clusters(None, 4) == (None, [])
    assert_array_equal(cluster_mask(None, 4), np.ones(4, dtype=bool))
    with pytest.raises(ValueError, match='missing'):
        encode_clusters(labels, 7)


@pytest.mark.parametrize('label', [True, np.bool_(False), '', 1.0, np.inf, [], {}, object()])
def test_invalid_labels_fail_even_on_rows_dropped_for_another_reason(label):
    with pytest.raises(ValueError):
        cluster_mask([1, label, 2], 3)
    with pytest.raises(ValueError):
        encode_clusters([1, label, 2], 3, rows=np.array([0, 2]))


def test_joint_drop_mask_and_cluster_levels_use_retained_rows_only():
    y = np.arange(6, dtype=float)
    y[1] = np.nan
    z = np.column_stack((np.arange(6.), np.arange(6.) ** 2))
    z[2, 0] = np.nan
    labels = ['kept-b', 'dropped', 'dropped', None, 'kept-a', 'kept-b']
    data = prepare_data(y, z, cluster=labels, missing='drop')
    assert_array_equal(data['rows'], [0, 4, 5])
    assert data['original_n'] == 6
    assert data['cluster_levels'] == ['kept-b', 'kept-a']
    assert_array_equal(data['cluster_codes'], [0, 1, 0])
    assert_array_equal(data['y'], [0, 4, 5])
    with pytest.raises(ValueError, match='missing'):
        prepare_data(np.arange(6.), np.column_stack((np.arange(6.), np.arange(6.) ** 2)),
                     cluster=labels, missing='raise')


@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
def test_literal_joint_normal_known_and_unseen_cluster_moments(family):
    # r=0 makes K constant. lambda=2, lambda_cluster=3 and different training
    # clusters give V=[[6,2],[2,6]]. Known cluster0 has crosscov[5,2]; unseen
    # cluster2 has crosscov[2,2]. The two unseen profiles share the SAME u.
    variance = .5 if family == 'gaussian' else 1.
    train_cov = variance * np.array([[6., 2.], [2., 6.]])
    cross = variance * np.array([[5., 2.], [2., 2.], [2., 2.]])
    prior = variance * np.array([[5., 2., 2.], [2., 5., 5.], [2., 5., 5.]])
    # This full block covariance defines the independent normal-law oracle.
    joint = np.block([[train_cov, cross.T], [cross, prior]])
    observed = np.array([1., -2.])
    expected_mean = joint[2:, :2] @ np.linalg.solve(joint[:2, :2], observed)
    expected_cov = joint[2:, 2:] - joint[2:, :2] @ np.linalg.solve(joint[:2, :2], joint[:2, 2:])
    z = np.array([[0.], [1.]])
    new = np.array([[.2], [.3], [.9]])  # all exposure coordinates irrelevant
    kernel, covariance, lower = covariance_components(z, [0.], 2., cluster_codes=[0, 1], lambda_cluster=3.)
    assert_array_equal(kernel, np.ones((2, 2)))
    assert_allclose(variance * covariance, train_cov, **TOL)
    assert_allclose(lower @ lower.T, covariance, **TOL)
    mean, cov = conditional_h(z, new, observed, np.empty((2, 0)), [], [0.], 2., variance,
                              cluster_codes=[0, 1], lambda_cluster=3., cluster_new_codes=[0, 2, 2])
    assert_allclose(mean, expected_mean, **TOL)
    assert_allclose(cov, expected_cov, **TOL)
    assert_allclose(cov[:, 1], cov[:, 2], **TOL)
    # Omitting new codes returns h alone, NOT population-marginal h+u.
    hcross = np.full((3, 2), 2 * variance)
    hmean = hcross @ np.linalg.solve(train_cov, observed)
    hcov = np.full((3, 3), 2 * variance) - hcross @ np.linalg.solve(train_cov, hcross.T)
    actual = conditional_h(z, new, observed, np.empty((2, 0)), [], [0.], 2., variance,
                           cluster_codes=[0, 1], lambda_cluster=3.)
    assert_allclose(actual[0], hmean, **TOL)
    assert_allclose(actual[1], hcov, **TOL)


def test_installed_r_fixture_identity_and_exporter_binding():
    assert FIXTURE['provenance']['reference'] == 'installed bkmr 0.2.2'
    assert 'no MCMC' in FIXTURE['provenance']['fixture_kind']
    assert len(FIXTURE['cases']) == 2
    assert 'no new cluster identities' in FIXTURE['provenance']['cluster_prediction_limit']
    exporter = HERE.parent / 'tools' / 'export_cluster_reference.R'
    assert hashlib.md5(exporter.read_bytes()).hexdigest() == FIXTURE['provenance']['exporter_md5']


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda case: case['family'])
def test_installed_r_clustered_marginal_primitives_and_public_h_moments(case):
    fitted = state_fit(case)
    for draw, state in enumerate(case['states']):
        _, v, lower = covariance_components(case['Z'], case['r'][draw], case['lambda'][draw],
            cluster_codes=case['cluster_codes'], lambda_cluster=case['lambda_cluster'][draw])
        assert_allclose(np.linalg.solve(v, np.eye(len(v))), state['Vinv'], **TOL)
        assert_allclose(-2 * np.log(np.diag(lower)).sum(), state['logdetVinv'], **TOL)
        actual = fitted._condition(np.asarray(case['Znew']), 0, draw)
        assert_allclose(actual[0], state['mean'], **TOL)
        assert_allclose(actual[1], state['covariance'], **TOL)
    moments = fitted.posterior_moments_h(np.asarray(case['Znew']))
    means = np.asarray([state['mean'] for state in case['states']])
    # R exact aggregation uses ddof1; Python finite posterior mixture uses0.
    expected_cov = np.asarray(case['exact']['covariance']) - np.cov(means, rowvar=False, ddof=1) / 3
    assert_allclose(moments['mean'], case['exact']['mean'], **TOL)
    assert_allclose(moments['covariance'], expected_cov, **TOL)
    for mode, key in [('component', 'pips'), ('group', 'grouped_pips')]:
        for actual, expected in zip(state_fit(case, mode).inclusion_probabilities(), case[key]):
            if mode == 'component':
                assert_allclose(actual['pip'], expected['PIP'], **TOL)
            else:
                assert_allclose(actual['group_pip'], expected['groupPIP'], **TOL)
                assert_allclose(actual['conditional_pip'], expected['condPIP'], **TOL)


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda case: case['family'])
def test_public_random_intercept_moments_follow_full_joint_normal(case, monkeypatch):
    import kernelmix.result as result_module

    fitted = state_fit(case)
    recorded = []

    def capture_mean(mean, covariance, rng):
        recorded.append((mean.copy(), covariance.copy()))
        return mean.copy()

    monkeypatch.setattr(result_module, '_normal_draw', capture_mean)
    labels = [0, 'unseen', 'unseen', 1]
    actual = fitted.random_intercepts(labels, seed=12)
    # unique codes are0,1,2; the last has no training crosscovariance.
    membership = np.array([[1., 1., 0., 0.], [0., 0., 1., 1.], [0., 0., 0., 0.]])
    codes = np.array([0, 1, 2])
    train_codes = np.asarray(case['cluster_codes'])
    for draw, (mean, covariance) in enumerate(recorded):
        # Literal membership supplies random-intercept covariance blocks;
        # use installed-R precision independently of Python covariance helpers.
        amplitude = case['lambda_cluster'][draw]
        sigma2 = case['sigma2'][draw]
        cross = sigma2 * amplitude * membership
        prior = sigma2 * amplitude * np.eye(3)
        training_cov = sigma2 * np.linalg.inv(np.asarray(case['states'][draw]['Vinv']))
        outcome = np.asarray(case['y'] if case['family'] == 'gaussian' else case['ystar'][draw])
        residual = outcome - np.asarray(case['X']) @ np.asarray(case['beta'][draw])
        expected_mean = cross @ np.linalg.solve(training_cov, residual)
        expected_cov = prior - cross @ np.linalg.solve(training_cov, cross.T)
        assert_allclose(mean, expected_mean, **TOL)
        assert_allclose(covariance, expected_cov, **TOL)
        assert mean[2] == 0
        assert_allclose(covariance[2, 2], sigma2 * amplitude, **TOL)
    assert len(recorded) == 3
    assert_array_equal(actual.draws[..., 1], actual.draws[..., 2])
    assert_array_equal(codes, [0, 1, 2])
    assert_array_equal(train_codes, [0, 0, 1, 1])


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda case: case['family'])
def test_clustered_prediction_shares_profiles_by_exposure_and_cluster(case):
    fitted = state_fit(case)
    znew = np.repeat(np.asarray(case['Znew'])[:1], 4, axis=0)
    # Equal exposure profiles and equal cluster labels must give exact copies;
    # distinct cluster identities must not be collapsed by exposure dedup.
    prediction = fitted.predict(znew, np.zeros((4, 1)), scale='latent',
                                 cluster_new=[0, 0, 'new', 'new'], seed=19)
    assert_array_equal(prediction.draws[..., 0], prediction.draws[..., 1])
    assert_array_equal(prediction.draws[..., 2], prediction.draws[..., 3])
    assert not np.array_equal(prediction.draws[..., 0], prediction.draws[..., 2])
    repeated = fitted.predict(znew, np.zeros((4, 1)), scale='latent',
                               cluster_new=[0, 0, 'new', 'new'], seed=19)
    assert_array_equal(prediction.draws, repeated.draws)


@pytest.mark.parametrize('case', FIXTURE['cases'], ids=lambda case: case['family'])
def test_legacy_absent_cluster_metadata_keeps_seeded_prediction_and_pips(case):
    legacy = state_fit(case, clustered=False)
    explicit_data = dict(legacy.data, cluster_codes=None, cluster_levels=[])
    explicit = BKMRFit(explicit_data, legacy.samples, legacy.settings, {})
    first = legacy.predict_h(np.asarray(case['Znew']), seed=492)
    second = explicit.predict_h(np.asarray(case['Znew']), seed=492)
    assert_array_equal(first.draws, second.draws)
    assert legacy.inclusion_probabilities() == explicit.inclusion_probabilities()
    with pytest.raises(ValueError, match='clustered'):
        legacy.random_intercepts(seed=7)


def test_cluster_archive_roundtrip_preserves_typed_levels_and_joint_predictions(tmp_path):
    fitted = state_fit(FIXTURE['cases'][0])
    data = dict(fitted.data, cluster_levels=[1, '1'])
    typed = BKMRFit(data, fitted.samples, fitted.settings, {})
    path = tmp_path / 'clustered.npz'
    typed.save(path)
    restored = BKMRFit.load(path)
    assert restored.data['cluster_levels'] == [1, '1']
    assert type(restored.data['cluster_levels'][0]) is int
    before = typed.random_intercepts([1, '1', 'new', 'new'], seed=14)
    after = restored.random_intercepts([1, '1', 'new', 'new'], seed=14)
    assert_array_equal(before.draws, after.draws)
    assert_array_equal(after.draws[..., 2], after.draws[..., 3])


@pytest.mark.parametrize('family', ['gaussian', 'binomial'])
def test_clustered_sampler_state_and_omitted_vs_explicit_no_cluster_seed(family):
    # Root executes this bounded check later; fixture authoring never ran fits.
    z = np.column_stack((np.arange(8.), np.sin(np.arange(8.))))
    y = np.array([0., 1., 0., 1., 1., 0., 1., 0.])
    if family == 'gaussian':
        y = y + np.arange(8.) / 10
    options = dict(family=family, chains=2, draws=3, warmup=2, seed=91,
                   selection='group', groups=['a', 'a'])
    clustered = fit(y, z, cluster=[1, 1, '1', '1', 1, 1, '1', '1'], **options)
    assert clustered.settings['clustered']
    assert clustered.data['cluster_levels'] == [1, '1']
    assert clustered.samples['lambda_cluster'].shape == (2, 3)
    assert np.all(clustered.samples['lambda_cluster'] > 0)
    assert np.all(clustered.samples['delta'].sum(axis=-1) <= 1)
    if family == 'binomial':
        assert_array_equal(clustered.samples['sigma2'], np.ones((2, 3)))
        assert np.all(clustered.samples['ystar'][..., y == 1] > 0)
        assert np.all(clustered.samples['ystar'][..., y == 0] < 0)
    omitted = fit(y, z, **options)
    explicit = fit(y, z, cluster=None, **options)
    for key, value in omitted.samples.items():
        assert_array_equal(value, explicit.samples[key])
    assert 'lambda_cluster' not in omitted.samples
    assert 'cluster_lambda' not in omitted.sampling_info['chains'][0]['acceptance']


@pytest.mark.parametrize('field', ['cluster_lambda_mean', 'cluster_lambda_sd', 'beta_sd',
                                 'dispersion_shape', 'dispersion_rate', 'baseline_shape', 'baseline_rate'])
def test_new_positive_prior_fields_reject_invalid_values(field):
    for value in (0., -1., np.inf, np.nan, True):
        with pytest.raises(ValueError):
            Prior(**{field: value})
