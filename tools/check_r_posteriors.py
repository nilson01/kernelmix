"""Optional fitted-posterior comparisons with installed R bkmr 0.2.2.

R is used only by this developer check, never by package fitting or prediction.
Run --smoke before the four fixed main cases. Saved reports retain every result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path

for _thread in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_thread] = '1'

import numpy as np  # noqa: E402

from kernelmix import Prior, diagnose, fit  # noqa: E402


def configurations(smoke=False, r_draws=None, python_draws=None, r_segment_size=None):
    t = np.linspace(-1, 1, 12)
    z = np.column_stack((t, np.sin(5*t)))
    z = (z-z.mean(axis=0))/z.std(axis=0, ddof=1)
    x = np.column_stack((np.ones(12), np.cos(4*t)))
    y = .5*np.sin(2*t)+.2*x[:, 1]+np.array([.3, -.2, .4, -.4, .1, .2, -.3, .4, -.1, .2, -.4, .1])
    binary = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1])
    for family in ['gaussian', 'binomial']:
        for clustered in ([False] if smoke else [False, True]):
            yield {'case': family+('-cluster' if clustered else ''), 'family': family,
                   'clustered': clustered, 'cluster': np.repeat(np.arange(4), 3).tolist(),
                   'Z': z.tolist(), 'X': x.tolist(),
                   'y': (y if family == 'gaussian' else binary).tolist(),
                   'chains': 4, 'warmup': 100 if smoke else 3000,
                   'draws': 200 if smoke else 12000,
                   **({'r_draws': r_draws} if r_draws is not None else {}),
                   **({'python_draws': python_draws} if python_draws is not None else {}),
                   **({'r_segment_size': r_segment_size} if r_segment_size is not None else {}),
                   'r_seeds': [7301, 7302, 7303, 7304],
                   'python_seeds': [8401, 8402, 8403, 8404],
                   'smoke_only': smoke}


def batch_statistics(draws):
    draws = np.asarray(draws, dtype=float)
    if draws.ndim != 2 or not np.isfinite(draws).all():
        raise ValueError('Require finite (chains, draws) values')
    chains, n = draws.shape
    if chains < 2 or n < 2000:
        raise ValueError('At least two chains with 2000 saved draws are required')
    q = [.1, .5, .9]
    values = np.r_[draws.mean(), np.quantile(draws, q)]
    errors = []
    for batches in sorted({20, min(40, n//100)}):
        length = n//batches
        blocks = draws[:, :length*batches].reshape(chains*batches, length)
        estimates = np.column_stack((blocks.mean(axis=1), np.quantile(blocks, q, axis=1).T))
        errors.append(estimates.std(axis=0, ddof=1)/np.sqrt(len(blocks)))
    # Use the larger MCSE from two batch lengths as a conservative sensitivity check.
    mcse = np.max(errors, axis=0)
    return values, mcse


def compare_arrays(r, python):
    r = np.asarray(r)
    python = np.asarray(python)
    if r.ndim != 2 or python.ndim != 2:
        raise ValueError('Each implementation requires a (chains, draws) array')
    rv, re = batch_statistics(r)
    pv, pe = batch_statistics(python)
    rd, pd = diagnose(r), diagnose(python)
    diagnostics_ok = all(np.isfinite(d[k]) and d[k] <= 1.01 if k == 'rhat'
                         else np.isfinite(d[k]) and d[k] >= 400
                         for d in [rd, pd] for k in ['rhat', 'ess_bulk', 'ess_tail'])
    pooled_sd = np.sqrt((r.var(ddof=1)+python.var(ddof=1))/2)
    combined = np.hypot(re, pe)
    tolerance = 6*combined
    difference = pv-rv
    precision_ok = (re <= .02*pooled_sd) & (pe <= .02*pooled_sd)
    margin_ok = np.abs(difference)+3*combined <= .15*pooled_sd
    comparisons = [{'summary': label, 'r': float(a), 'python': float(b),
                    'r_mcse': float(c), 'python_mcse': float(d),
                    'difference': float(delta), 'tolerance': float(tol),
                    'within_tolerance': bool(abs(delta) <= tol),
                    'precision_ok': bool(precise), 'margin_ok': bool(margin),
                    'margin': float(.15*pooled_sd)}
                   for label, a, b, c, d, delta, tol, precise, margin in zip(
                       ['mean', 'q10', 'q50', 'q90'], rv, pv, re, pe, difference, tolerance, precision_ok, margin_ok)]
    return {'r_diagnostics': rd, 'python_diagnostics': pd,
            'diagnostics_ok': bool(diagnostics_ok), 'comparisons': comparisons,
            'passed': bool(diagnostics_ok and all(c['within_tolerance'] and c['precision_ok']
                                                and c['margin_ok'] for c in comparisons))}


def run_python(config):
    prior = Prior(r_distribution='gamma', r_mean=1, r_sd=np.sqrt(.5),
                  lambda_mean=1, lambda_sd=np.sqrt(.5), sigma2_shape=3, sigma2_rate=1,
                  cluster_lambda_mean=1, cluster_lambda_sd=np.sqrt(.5))
    z, x, y = [np.asarray(config[k]) for k in ['Z', 'X', 'y']]
    chain_rows = []
    messages = []
    start = time.monotonic()
    for ch, seed in enumerate(config['python_seeds'], 1):
        initial = {'beta': [0, 0], 'r': [.6+.2*ch]*2,
                   'lambda': .6+.2*ch, 'sigma2': .7 if config['family']=='gaussian' else 1}
        if config['clustered']:
            initial['lambda_cluster'] = .6+.2*ch
        with warnings.catch_warnings(record=True) as caught:
            result = fit(y, z, x, family=config['family'], standardize=False, intercept=False,
                         cluster=config['cluster'] if config['clustered'] else None,
                         prior=prior, initial=initial, draws=config.get('python_draws', config['draws']),
                         warmup=config['warmup'], chains=1, seed=seed,
                         proposal_scales={'r': .6, 'lambda': .6,
                                          **({'cluster_lambda': .6} if config['clustered'] else {})})
            h = result.predict_h(z[[0, 5, 11]], seed=seed+100).draws[0]
        messages.extend(str(w.message) for w in caught)
        s = result.samples
        columns = [s['beta'][0, :, 0], s['beta'][0, :, 1], s['r'][0, :, 0],
                   s['r'][0, :, 1], s['lambda'][0]]
        names = ['beta_0', 'beta_1', 'r_0', 'r_1', 'lambda']
        if config['clustered']:
            columns.append(s['lambda_cluster'][0])
            names.append('lambda_cluster')
        if config['family']=='gaussian':
            columns.append(s['sigma2'][0])
            names.append('sigma2')
        columns.extend([h[:, 0], h[:, 1], h[:, 2], h[:, 2]-h[:, 0]])
        names.extend(['h_0', 'h_5', 'h_11', 'h_contrast'])
        chain_rows.append(np.column_stack(columns))
    import scipy

    import kernelmix
    origin = Path(kernelmix.__file__).resolve().parent
    identity = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in origin.glob('*.py')}
    return np.stack(chain_rows), names, {'seconds': time.monotonic()-start, 'warnings': messages,
        'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
        'package_origin': str(origin), 'source_sha256': identity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rscript', default='Rscript')
    parser.add_argument('--r-library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--case', choices=['gaussian', 'gaussian-cluster', 'binomial', 'binomial-cluster'])
    parser.add_argument('--config-dir', type=Path,
                        help='Require equality with pre-frozen case JSON files.')
    parser.add_argument('--r-draws', type=int, help='Retained draws per R chain.')
    parser.add_argument('--python-draws', type=int, help='Retained draws per Python chain.')
    parser.add_argument('--r-segment-size', type=int, default=14999,
                        help='R transitions per installed kmbayes call, carrying state and RNG.')
    args = parser.parse_args()
    if args.r_segment_size < 1:
        raise ValueError('R segment size must be positive')
    for value in [args.r_draws, args.python_draws]:
        if value is not None and value < 2000:
            raise ValueError('Explicit draw counts must be at least 2000')
    if args.output.exists():
        raise SystemExit('Output exists; use a new directory to preserve earlier attempts')
    args.output.mkdir(parents=True)
    reports = []
    for cfg in configurations(args.smoke, args.r_draws, args.python_draws, args.r_segment_size):
        if args.case is not None and cfg['case'] != args.case:
            continue
        if args.config_dir is not None:
            frozen = json.loads((args.config_dir/f"{cfg['case']}.json").read_text())
            if frozen != cfg:
                raise ValueError('Current settings differ from the frozen configuration')
        dest = args.output/cfg['case']
        dest.mkdir()
        config = dest/'config.json'
        config.write_text(json.dumps(cfg, indent=2))
        env = dict(os.environ, R_LIBS=str(args.r_library.resolve()), LC_ALL='C')
        with (dest/'r.log').open('w') as log:
            subprocess.run([args.rscript, '--vanilla', str(Path(__file__).with_name('export_fitted_reference.R')),
                            str(config.resolve()), str(dest.resolve())], env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        metadata = json.loads((dest/'r-metadata.json').read_text())
        if (metadata['version'] != '0.2.2' or metadata['draws'] != cfg.get('r_draws', cfg['draws'])
                or Path(metadata['library']).resolve() != (args.r_library/'bkmr').resolve()):
            raise ValueError('Reference metadata does not match configuration')
        rchains = [np.genfromtxt(dest/f'r-chain-{ch}.csv.gz', delimiter=',', names=True)
                   for ch in range(1, cfg['chains']+1)]
        rnames = list(rchains[0].dtype.names)
        rdraws = np.stack([np.column_stack([chain[name] for name in rnames]) for chain in rchains])
        pdraws, names, pmetadata = run_python(cfg)
        if (names != rnames or rdraws.shape[0] != pdraws.shape[0]
                or rdraws.shape[2] != pdraws.shape[2]
                or rdraws.shape[1] != cfg.get('r_draws', cfg['draws'])
                or pdraws.shape[1] != cfg.get('python_draws', cfg['draws'])):
            raise ValueError('Reference columns or draw counts differ')
        np.savez_compressed(dest/'draws.npz', r=rdraws, python=pdraws, names=names)
        report = {'case': cfg['case'], 'smoke_only': args.smoke,
                  'r_metadata': metadata, 'python_metadata': pmetadata,
                  'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
                  'draws_sha256': hashlib.sha256((dest/'draws.npz').read_bytes()).hexdigest(),
                  'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'quantities': {} if args.smoke else {
                      name: compare_arrays(rdraws[:, :, j], pdraws[:, :, j])
                      for j, name in enumerate(names)}}
        report['passed'] = None if args.smoke else all(v['passed'] for v in report['quantities'].values())
        (dest/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False))
        reports.append(report)
        print(json.dumps({'case': cfg['case'], 'passed': report['passed'],
                          'r_seconds': metadata['seconds'], 'python_seconds': pmetadata['seconds']}), flush=True)
    if not reports:
        raise ValueError('No cases selected')
    (args.output/'results.json').write_text(json.dumps(reports, indent=2, allow_nan=False))
    if not args.smoke and not all(r['passed'] for r in reports):
        raise SystemExit('One or more comparisons failed or have inadequate diagnostics; inspect saved results')


if __name__ == '__main__':
    main()
