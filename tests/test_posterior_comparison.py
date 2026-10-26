"""Fault checks for the optional full-chain reference comparator."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    'posterior_check', Path(__file__).resolve().parents[1]/'tools/check_r_posteriors.py')
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


def draws(seed):
    return np.random.default_rng(seed).normal(size=(4, 12000))


def test_matching_independent_distributions_pass():
    report = CHECKER.compare_arrays(draws(21), draws(22))
    assert report['passed']
    assert all(row['precision_ok'] and row['margin_ok'] for row in report['comparisons'])


def test_shifted_posterior_is_rejected():
    assert not CHECKER.compare_arrays(draws(21), draws(22)+.5)['passed']


def test_bad_chain_mixing_is_rejected_even_if_marginals_match():
    values = draws(21)
    values[0] += 2
    assert not CHECKER.compare_arrays(values, values.copy())['passed']


def test_low_precision_cannot_pass_from_wide_mc_tolerance():
    # Independent across chains, but 120-draw plateaus give very low information.
    values = np.repeat(np.random.default_rng(21).normal(size=(4, 100)), 120, axis=1)
    report = CHECKER.compare_arrays(values, values.copy())
    assert not report['passed']
    assert not report['diagnostics_ok'] or any(not c['precision_ok'] for c in report['comparisons'])


@pytest.mark.parametrize('bad', [np.zeros((4, 100)), np.ones(12000),
                               np.full((4, 2000), np.nan)])
def test_missing_or_invalid_draws_raise(bad):
    with pytest.raises(ValueError):
        CHECKER.batch_statistics(bad)


def test_empty_and_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        CHECKER.compare_arrays(draws(21), draws(22)[None])


def test_fixed_case_inventory_and_standardization():
    cases = list(CHECKER.configurations())
    assert {c['case'] for c in cases} == {'gaussian', 'gaussian-cluster',
                                        'binomial', 'binomial-cluster'}
    for case in cases:
        z = np.asarray(case['Z'])
        assert np.allclose(z.mean(axis=0), 0)
        assert np.allclose(z.std(axis=0, ddof=1), 1)
        assert case['draws'] == 12000 and case['chains'] == 4
    assert all(c['smoke_only'] for c in CHECKER.configurations(True))


def test_unequal_saved_counts_are_valid_for_independent_comparisons():
    assert CHECKER.compare_arrays(draws(21), draws(22)[:, :11000])['passed']


def run_cli(*arguments):
    import os
    import subprocess
    import sys
    return subprocess.run([sys.executable, str(Path(CHECKER.__file__)), *map(str, arguments)],
                          capture_output=True, text=True, env=dict(os.environ), timeout=20)


def test_cli_missing_r_retains_config_and_fails(tmp_path):
    output = tmp_path/'missing-backend'
    result = run_cli('--rscript', tmp_path/'not-an-executable', '--r-library', tmp_path,
                     '--output', output, '--smoke', '--case', 'gaussian')
    assert result.returncode != 0
    assert (output/'gaussian/config.json').exists()
    assert not (output/'results.json').exists()


def test_cli_refuses_existing_output_without_overwriting(tmp_path):
    output = tmp_path/'existing'
    output.mkdir()
    marker = output/'retained.txt'
    marker.write_text('retain prior attempt')
    result = run_cli('--r-library', tmp_path, '--output', output)
    assert result.returncode != 0 and 'Output exists' in result.stderr
    assert marker.read_text() == 'retain prior attempt'


def test_cli_rejects_changed_frozen_inputs_before_calling_r(tmp_path):
    configs = tmp_path/'configs'
    configs.mkdir()
    (configs/'gaussian.json').write_text('{}')
    result = run_cli('--rscript', tmp_path/'not-an-executable', '--r-library', tmp_path,
                     '--output', tmp_path/'new-output', '--case', 'gaussian',
                     '--config-dir', configs)
    assert result.returncode != 0 and 'frozen configuration' in result.stderr
