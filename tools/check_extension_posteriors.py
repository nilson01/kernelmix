"""Optional joint-posterior checks against independent low-dimensional integrals.

These fixed data checks test numerical sampling targets, not repeated-data coverage.
Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/check_extension_posteriors.py --output report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

try:
    import resource
except ImportError:  # Windows has no resource module.
    resource = None
import time

import numpy as np
import scipy
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import quad, simpson
from scipy.special import gammaln

import kernelmix as km

SEEDS = {"negative_binomial": 818, "survival": 819, "structured": 820}
NB_Y = np.array([0., 0., 1., 3.])
SURVIVAL_TIME = np.array([1., 1.5, 2., 3.])
SURVIVAL_EVENT = np.array([1, 0, 1, 0])
STRUCTURED_Y = np.array([-.5, .2, .6, .1, .5, 1.])
SLOPE = np.linspace(-1., 1., 6)
RANDOM_COVARIANCE = np.array([[.4, .1], [.1, .3]])


def h_prior(h):
    """Unnormalised marginal h density for tau~Gamma(4,4), h|tau~N(0,tau).

    Integrating tau gives the half-integer Bessel expression below. No
    production likelihood, kernel, sampler or prior helper constructs targets.
    """
    a = np.abs(h)
    c = math.sqrt(8.)
    return np.exp(-c*a) * (a**3 + 6*a*a/c + 15*a/c**2 + 15/c**3)


def nb_target(points=801, expanded=False):
    h = np.linspace(-16. if expanded else -12., 10. if expanded else 8., points)
    log_phi = np.linspace(-10. if expanded else -8., 10. if expanded else 8., points)
    phi = np.exp(log_phi)[None, :]
    eta = h[:, None]
    lp = np.zeros((points, points))
    for y in NB_Y:
        lp += (gammaln(y+phi)-gammaln(phi)-gammaln(y+1)
               -phi*np.logaddexp(0., eta-log_phi[None, :])
               -y*np.logaddexp(0., log_phi[None, :]-eta))
    # Gamma(shape2,rate.1) density and dphi/dlog_phi Jacobian.
    lp += 2*log_phi[None, :]-.1*phi
    lp += np.log(h_prior(h))[:, None]
    w = np.exp(lp-lp.max())

    def integrate(value):
        return simpson(simpson(w*value, x=log_phi, axis=1), x=h)

    norm = integrate(1.)
    return dict(h=float(integrate(eta)/norm), dispersion=float(integrate(phi)/norm),
                h_dispersion=float(integrate(eta*phi)/norm),
                h_below_zero=float(integrate((eta <= 0).astype(float))/norm))


def survival_target():
    events = SURVIVAL_EVENT.sum()
    duration = SURVIVAL_TIME.sum()
    shape, rate = 2.+events, 1.

    def weight(h):
        return float(h_prior(h))*math.exp(events*h-shape*np.logaddexp(0., math.log(duration)+h))

    functions = {"h": lambda h: h,
                 "baseline": lambda h: shape/(rate+duration*math.exp(h)),
                 "h_baseline": lambda h: h*shape/(rate+duration*math.exp(h))}
    norm, error = quad(weight, -25., 15., epsabs=1e-12, epsrel=1e-10, points=[0.])
    output = {}
    errors = [error]
    for name, fn in functions.items():
        value, err = quad(lambda h: weight(h)*fn(h), -25., 15., epsabs=1e-12, epsrel=1e-10, points=[0.])
        output[name] = value/norm
        errors.append(err/norm)
    return output, max(errors)


def structured_target(points=401, nodes=32, expanded=False):
    h = np.linspace(-16. if expanded else -12., 16. if expanded else 12., points)
    x, weights = hermgauss(nodes)
    white = np.stack(np.meshgrid(x*math.sqrt(2), x*math.sqrt(2), indexing='ij'), axis=-1).reshape(-1, 2)
    coeff = white @ np.linalg.cholesky(RANDOM_COVARIANCE).T
    gh_weights = (weights[:, None]*weights[None, :]).ravel()/math.pi
    residual_corr = .35 ** np.abs(np.arange(6)[:, None]-np.arange(6)[None, :])
    precision = np.linalg.inv(residual_corr)
    remainder = STRUCTURED_Y[None, :]-coeff[:, :1]-coeff[:, 1:]*SLOPE[None, :]
    # sigma2~IG(2,1) is integrated out. Conditional E[sigma2]=scale/(shape-1).
    shape = 2.+len(STRUCTURED_Y)/2
    rows = []
    for value in h:
        residual = remainder-value
        scale = 1.+.5*np.einsum('bi,ij,bj->b', residual, precision, residual)
        w = gh_weights*scale**(-shape)*h_prior(value)
        rows.append([w.sum(), (w*value).sum(), (w*coeff[:, 0]).sum(),
                     (w*coeff[:, 1]).sum(), (w*value*coeff[:, 0]).sum(),
                     (w*coeff[:, 0]*coeff[:, 1]).sum(), (w*scale/(shape-1)).sum()])
    integrated = simpson(np.asarray(rows), x=h, axis=0)
    names = ['h', 'random_intercept', 'random_slope', 'h_intercept', 'intercept_slope', 'sigma2']
    return dict(zip(names, map(float, integrated[1:]/integrated[0])))


def summarize(values, target):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("values must be finite, with at least2 chains and2 draws")
    if not np.isfinite(target) or np.var(values) == 0:
        raise ValueError("target must be finite and draws must vary")
    chains, draws = values.shape
    batch_size = max(1, draws//40)
    usable = draws//batch_size*batch_size
    batches = values[:, :usable].reshape(chains, -1, batch_size).mean(axis=-1).ravel()
    mcse = float(batches.std(ddof=1)/math.sqrt(len(batches)))
    half = draws//2
    split = np.concatenate([values[:, :half], values[:, -half:]])
    within = np.var(split, axis=1, ddof=1).mean()
    between = half*np.var(split.mean(axis=1), ddof=1)
    rhat = float(math.sqrt(((half-1)/half*within+between/half)/within))
    sd = float(values.std(ddof=1))
    error = abs(float(values.mean())-target)
    # Fixed, declared before running: six MCSEs AND <10% posterior SD,
    # MCSE below 2% posterior SD, raw split Rhat below1.05.
    passed = bool(error <= 6*mcse and error <= .10*sd and mcse <= .02*sd and rhat < 1.05)
    return dict(mean=float(values.mean()), target=target, error=error, mcse=mcse,
                sd=sd, split_rhat=rhat, passed=passed)


def summarize_covariance(left, right, left_mean, right_mean, cross_mean):
    target = cross_mean-left_mean*right_mean
    if target == 0:
        raise ValueError("The relative covariance criterion requires a nonzero target")
    result = summarize((left-left_mean)*(right-right_mean), target)
    result['covariance_precision_passed'] = bool(result['mcse'] <= abs(target)/6)
    result['covariance_magnitude_passed'] = bool(result['error'] <= .5*abs(target))
    result['passed'] = bool(result['passed'] and result['covariance_precision_passed']
                            and result['covariance_magnitude_passed'])
    return result


def _save_draws(directory, name, samples):
    if directory is None:
        return None
    path = directory/(name+'-draws.npz')
    np.savez_compressed(path, **samples)
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in samples.items()})


def source_identity():
    package = Path(km.__file__).resolve().parent
    repo = Path(__file__).resolve().parents[1]
    def git(*arguments):
        result = subprocess.run(['git', '-C', str(repo), *arguments], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return dict(kernelmix_file=str(package/'__init__.py'), commit=git('rev-parse', 'HEAD'),
                working_tree=git('status', '--porcelain'),
                sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(package.glob('*.py'))},
                dependencies=dict(numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version()),
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def run(draws=100000, warmup=1000, chains=2, artifact_dir=None):
    started = time.perf_counter()
    reports = {}
    identity = source_identity()
    coarse = nb_target(801)
    target = nb_target(1201, expanded=True)
    # h<=0 grid indicator has first-order boundary error: use smooth moments.
    del coarse['h_below_zero'], target['h_below_zero']
    quadrature_error = max(abs(target[k]-coarse[k]) for k in target)
    prior = km.Prior(lambda_mean=1., lambda_sd=.5, sigma2_shape=2., sigma2_rate=1.,
                     baseline_shape=2., baseline_rate=1.)
    options = dict(intercept=False, r_fixed=[0.], prior=prior, draws=draws,
                   warmup=warmup, chains=chains)
    case_start = time.perf_counter()
    fitted = km.fit(NB_Y, np.arange(4.)[:, None], family='negative_binomial',
                    proposal_scales={'dispersion': .8}, seed=SEEDS['negative_binomial'], **options)
    h, phi = fitted.samples['h'][..., 0], fitted.samples['dispersion']
    nb_covariance = summarize_covariance(h, phi, target['h'], target['dispersion'], target['h_dispersion'])
    reports['negative_binomial'] = dict(raw_draws=_save_draws(artifact_dir, 'negative_binomial', fitted.samples),
        covariances={'h_dispersion': nb_covariance}, quadrature_error=quadrature_error,
        summaries={name: summarize(value, target[name]) for name, value in
                   dict(h=h, dispersion=phi, h_dispersion=h*phi).items()},
        seconds=time.perf_counter()-case_start)
    target, quadrature_error = survival_target()
    case_start = time.perf_counter()
    fitted = km.fit(SURVIVAL_TIME, np.arange(4.)[:, None], family='survival',
                    event=SURVIVAL_EVENT, baseline_breaks=[], seed=SEEDS['survival'], **options)
    h, baseline = fitted.samples['h'][..., 0], fitted.samples['baseline_hazard'][..., 0]
    survival_covariance = summarize_covariance(h, baseline, target['h'], target['baseline'], target['h_baseline'])
    reports['survival'] = dict(raw_draws=_save_draws(artifact_dir, 'survival', fitted.samples),
        covariances={'h_baseline': survival_covariance}, quadrature_error=quadrature_error,
        summaries={name: summarize(value, target[name]) for name, value in
                   dict(h=h, baseline=baseline, h_baseline=h*baseline).items()},
        seconds=time.perf_counter()-case_start)
    coarse, target = structured_target(), structured_target(801, 48, True)
    quadrature_error = max(abs(target[k]-coarse[k]) for k in target)
    case_start = time.perf_counter()
    fitted = km.fit(STRUCTURED_Y, np.arange(6.)[:, None], family='gaussian',
                    approximation='rff', n_components=1, cluster=['a']*6,
                    random_slopes=SLOPE, random_covariance=RANDOM_COVARIANCE,
                    correlation='ar1', observation_time=np.arange(6.), correlation_parameter=.35,
                    seed=SEEDS['structured'], **options)
    # One frequency yields a cosine/sine pair (1,0) when r=0.
    # Its one nonzero feature gives common h; tau remains sampled.
    h = fitted.samples['gp_weight'][..., 0, 0]*np.sqrt(fitted.samples['lambda'][..., 0])
    coeff = fitted.samples['random_coeff'][..., 0, 0, :]
    b0, b1 = coeff[..., 0], coeff[..., 1]
    values = dict(h=h, random_intercept=b0, random_slope=b1, h_intercept=h*b0,
                  intercept_slope=b0*b1, sigma2=fitted.samples['sigma2'])
    covariances = {
        'h_intercept': summarize_covariance(h, b0, target['h'], target['random_intercept'], target['h_intercept']),
        'intercept_slope': summarize_covariance(b0, b1, target['random_intercept'], target['random_slope'], target['intercept_slope'])}
    reports['structured'] = dict(raw_draws=_save_draws(artifact_dir, 'structured', fitted.samples),
        covariances=covariances, quadrature_error=quadrature_error,
        summaries={name: summarize(value, target[name]) for name, value in values.items()},
        seconds=time.perf_counter()-case_start)
    for record in reports.values():
        record['passed'] = bool(record['quadrature_error'] < 1e-5 and
                                 all(s['passed'] for s in record['summaries'].values()) and
                                 all(s['passed'] for s in record['covariances'].values()))
    identity_unchanged = identity['sources'] == source_identity()['sources']
    return dict(schema_version=2, scope='fixed-data numerical joint-posterior targets; not coverage validation',
                settings=dict(draws=draws, warmup=warmup, chains=chains, seeds=SEEDS, prior=asdict(prior),
                              covariance_rule=dict(mcse_fraction=1/6, error_fraction=.5)),
                identity=identity, identity_unchanged=identity_unchanged,
                environment=dict(python=platform.python_version(), kernelmix_file=km.__file__),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                cases=reports, passed=identity_unchanged and all(r['passed'] for r in reports.values()),
                seconds=time.perf_counter()-started,
                peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*(1 if platform.system()=='Darwin' else 1024) if resource is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--draws', type=int, default=100000)
    args = parser.parse_args()
    if args.draws < 100:
        parser.error('--draws must be at least100')
    if args.output.exists():
        parser.error('output already exists; preserve prior attempts with a new output path')
    artifact_dir = args.output.parent/(args.output.stem+'-artifacts')
    artifact_dir.mkdir(parents=True, exist_ok=False)
    report = run(draws=200 if args.smoke else args.draws, warmup=100 if args.smoke else 1000, artifact_dir=artifact_dir)
    identity_path = artifact_dir/'identity.json'
    identity_path.write_text(json.dumps(report['identity'], indent=2)+'\n')
    report['identity_artifact'] = dict(path=str(identity_path.resolve()), sha256=hashlib.sha256(identity_path.read_bytes()).hexdigest())
    report['attempt'] = 'smoke' if args.smoke else 'fixed-case'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'], seconds=report['seconds'], output=str(args.output))))
    if not args.smoke and not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
