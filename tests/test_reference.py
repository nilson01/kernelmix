"""Deterministic installed-bkmr fixtures; constructed states, no MCMC or R runtime."""
import json
import unittest
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from kernelmix._kernel import conditional_h, covariance_components, gaussian_kernel
from kernelmix.priors import Prior, gamma_logpdf
from kernelmix.result import BKMRFit

FIXTURE = json.loads((Path(__file__).parent / 'fixtures' / 'r_reference.json').read_text())
TOL = dict(rtol=2e-11, atol=2e-12)


def constructed_fit(case, selection='component'):
    z, x = np.array(case['Z']), np.array(case['X'])
    data = dict(y=np.array(case['y']), Z=z, Z_original=z, X=x,
                X_original=x[:, 1:], z_center=np.zeros(z.shape[1]),
                z_scale=np.ones(z.shape[1]), rows=np.arange(len(z)), original_n=len(z),
                exposure_names=['z1', 'z2'], covariate_names=['intercept', 'x1'],
                groups=[1, 1] if selection == 'group' else None)
    samples = {key: np.array(case[key])[None, ...] for key in
               ('beta', 'r', 'lambda', 'sigma2', 'delta')}
    samples['pi'] = np.full((1, 3), .5)
    if case['family'] == 'binomial':
        samples['ystar'] = np.array(case['ystar'])[None, ...]
    settings = dict(family=case['family'], chains=1, draws=3, standardize=False,
                    intercept=True, selection=selection)
    return BKMRFit(data, samples, settings, {})


class InstalledRReference(unittest.TestCase):
    def test_fixture_identity(self):
        self.assertEqual(FIXTURE['provenance']['reference'], 'installed bkmr 0.2.2')
        self.assertIn('no MCMC', FIXTURE['provenance']['fixture_kind'])
        self.assertEqual(len(FIXTURE['cases']), 2)

    def test_kernel_and_covariance_primitives(self):
        for case in FIXTURE['cases']:
            for i, state in enumerate(case['states']):
                with self.subTest(family=case['family'], state=i):
                    z, r, lam = np.array(case['Z']), case['r'][i], case['lambda'][i]
                    k, v, lower = covariance_components(z, r, lam)
                    assert_allclose(k, state['K'], **TOL)
                    assert_allclose(gaussian_kernel(case['Znew'], z, r), state['cross_kernel'], **TOL)
                    assert_allclose(np.linalg.solve(v, np.eye(len(v))), state['Vinv'], **TOL)
                    assert_allclose(-2*np.log(np.diag(lower)).sum(), state['logdetVinv'], **TOL)

    def test_conditional_moments_each_state(self):
        for case in FIXTURE['cases']:
            fit = constructed_fit(case)
            for i, state in enumerate(case['states']):
                with self.subTest(family=case['family'], state=i):
                    y = case['y'] if case['family'] == 'gaussian' else case['ystar'][i]
                    mean, covariance = conditional_h(case['Z'], case['Znew'], y,
                        case['X'], case['beta'][i], case['r'][i],
                        case['lambda'][i], case['sigma2'][i])
                    assert_allclose(mean, state['mean'], **TOL)
                    assert_allclose(covariance, state['covariance'], **TOL)
                    actual = fit._condition(np.array(case['Znew']), 0, i)
                    assert_allclose(actual[0], state['mean'], **TOL)
                    assert_allclose(actual[1], state['covariance'], **TOL)

    def test_public_exact_moment_aggregation(self):
        for case in FIXTURE['cases']:
            with self.subTest(family=case['family']):
                actual = constructed_fit(case).posterior_moments_h(np.array(case['Znew']))
                assert_allclose(actual['mean'], case['exact']['mean'], **TOL)
                means = np.array([s['mean'] for s in case['states']])
                # R uses sample covariance (ddof=1). Python's finite empirical
                # mixture uses ddof=0: remove sample covariance / state count.
                expected = np.array(case['exact']['covariance']) - np.cov(means,rowvar=False,ddof=1)/len(means)
                assert_allclose(actual['covariance'], expected, **TOL)

    def test_public_inclusion_summaries(self):
        for case in FIXTURE['cases']:
            for mode, key in [('component', 'pips'), ('group', 'grouped_pips')]:
                actual = constructed_fit(case, mode).inclusion_probabilities()
                for got, expected in zip(actual, case[key]):
                    if mode == 'component':
                        assert_allclose(got['pip'], expected['PIP'], **TOL)
                    else:
                        self.assertEqual(got['group'], expected['group'])
                        assert_allclose(got['group_pip'], expected['groupPIP'], **TOL)
                        assert_allclose(got['conditional_pip'], expected['condPIP'], **TOL)

    def test_prior_densities(self):
        for case in FIXTURE['priors']:
            prior = Prior(r_distribution=case['distribution'])
            for value, expected in zip(case['values'], case['log_density']):
                with self.subTest(prior=case['distribution'], value=value):
                    if expected is None:
                        self.assertEqual(prior.log_r(value), -np.inf)
                    else:
                        assert_allclose(prior.log_r(value), expected, **TOL)
        case = FIXTURE['lambda_prior']
        assert_allclose([gamma_logpdf(x, 10, 10) for x in case['values']], case['log_density'], **TOL)


if __name__ == '__main__':
    unittest.main()
