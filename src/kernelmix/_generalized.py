"""Noncentered BKMR for count and right-censored survival outcomes.

Standard-normal latent blocks are updated by elliptical slice sampling;
positive hyperparameters use log random-walk Metropolis updates. Survival
baselines use exact conditional Gamma draws. These likelihood extensions use
proper Gaussian priors for adjustment coefficients, including the intercept.
"""

import math
import time
import warnings
from dataclasses import asdict

import numpy as np

from ._input import prepare_data
from ._kernel import gaussian_kernel
from ._likelihoods import (
    baseline_posterior_parameters,
    negative_binomial_loglik,
    poisson_loglik,
    prepare_survival,
    survival_loglik,
)
from .priors import Prior, gamma_logpdf


def _count(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer at least {minimum}')
    return int(value)


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive scalar')
    return float(value)


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite vector of length {size}')
    return result.copy()


def _original_vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,):
        raise ValueError(f'{name} must have one entry per original observation')
    if np.isinf(result).any():
        raise ValueError(f'{name} cannot contain infinity')
    return result.copy()


def _prepare(y, Z, X, family, standardize, intercept, missing,
             exposure_names, covariate_names, groups, cluster, offset, event, baseline_breaks):
    outcome = np.asarray(y, dtype=float)
    if outcome.ndim != 1:
        raise ValueError('y must be a one-dimensional outcome vector')
    n = len(outcome)
    if np.isinf(outcome).any():
        raise ValueError('y cannot contain infinity')
    finite = np.isfinite(outcome)
    masked = outcome.copy()
    if family in ('poisson', 'negative_binomial'):
        if event is not None or baseline_breaks is not None:
            raise ValueError('event and baseline_breaks apply only to survival outcomes')
        if np.any(outcome[finite] < 0) or np.any(outcome[finite] != np.floor(outcome[finite])):
            raise ValueError('count outcomes must be nonnegative integers')
        offsets = np.zeros(n) if offset is None else _original_vector(offset, n, 'offset')
        masked[np.isnan(offsets)] = np.nan
    else:
        if offset is not None:
            raise ValueError('offset applies only to count outcomes')
        if event is None:
            raise ValueError('survival outcomes require event indicators')
        if np.any(outcome[finite] <= 0):
            raise ValueError('observed survival times must be positive')
        events = _original_vector(event, n, 'event')
        if not np.isin(events[np.isfinite(events)], [0, 1]).all():
            raise ValueError('event must contain only zero and one')
        masked[np.isnan(events)] = np.nan
    # Inject only additional missingness into a copied outcome. The common
    # helper preserves original row numbers, named columns and train transforms.
    data = prepare_data(masked, Z, X, family='gaussian', standardize=standardize,
                        intercept=intercept, missing=missing, exposure_names=exposure_names,
                        covariate_names=covariate_names, groups=groups, cluster=cluster)
    rows = data['rows']
    data['y'] = outcome[rows].copy()
    if family == 'survival':
        data.update(prepare_survival(data['y'], events[rows], baseline_breaks))
    else:
        data['offset'] = offsets[rows].copy()
    return data


def elliptical_slice(current, log_likelihood, rng, *, current_loglik=None, max_steps=10000):
    """One ESS step for a standard-normal prior, returning state, loglik, evaluations.

    The bracket is randomly placed around the current angle, following Murray,
    Adams and MacKay (2010), Figure 2. The bounded loop fails explicitly rather
    than returning a rejected state as if it were a successful ESS update.
    """
    current = np.asarray(current, dtype=float)
    if current.ndim != 1 or not np.isfinite(current).all():
        raise ValueError('ESS requires a finite one-dimensional current state')
    max_steps = _count(max_steps, 'max_steps', 1)
    evaluations = 0
    if current_loglik is None:
        current_loglik = float(log_likelihood(current))
        evaluations += 1
    if not np.isfinite(current_loglik):
        raise ValueError('ESS requires a finite initial log likelihood')
    direction = rng.normal(size=current.size)
    # 1-U is in (0,1], avoiding log(0) without clipping a slice level.
    threshold = current_loglik + math.log1p(-float(rng.random()))
    theta = float(rng.uniform(0, 2 * math.pi))
    lower, upper = theta - 2 * math.pi, theta
    for _ in range(max_steps):
        proposal = current * math.cos(theta) + direction * math.sin(theta)
        proposed_loglik = float(log_likelihood(proposal))
        evaluations += 1
        if np.isfinite(proposed_loglik) and proposed_loglik > threshold:
            return proposal, proposed_loglik, evaluations
        if theta < 0:
            lower = theta
        else:
            upper = theta
        if not lower < upper:
            break
        theta = float(rng.uniform(lower, upper))
    raise RuntimeError('elliptical slice bracket did not close; rescale predictors or inspect the likelihood')


def _initialize(initial, data, prior, family, selection, fixed, units, rng):
    n, m = data['Z'].shape
    p = data['X'].shape[1]
    g = len(data['cluster_levels'])
    allowed = {'beta', 'r', 'lambda', 'delta', 'pi', 'latent_white'}
    if g:
        allowed.update(('lambda_cluster', 'u_white'))
    if family == 'negative_binomial':
        allowed.add('dispersion')
    if family == 'survival':
        allowed.add('baseline_hazard')
    if initial is None:
        initial = {}
    if not isinstance(initial, dict) or set(initial) - allowed:
        raise ValueError('initial must be a dictionary containing only parameters applicable to this model')
    pi = _positive(initial.get('pi', 1. if selection == 'none' else .5), 'initial pi')
    if (selection == 'none' and pi != 1) or (selection != 'none' and pi >= 1):
        raise ValueError('initial pi must be one without selection and strictly between zero and one otherwise')
    delta = np.ones(m, dtype=bool)
    if selection != 'none':
        delta[:] = False
        for members in units:
            if rng.random() < pi:
                delta[rng.choice(members)] = True
    if 'delta' in initial:
        supplied = np.asarray(initial['delta'])
        if supplied.shape != (m,) or not np.isin(supplied, [0, 1]).all():
            raise ValueError('initial delta must contain one zero/one indicator per exposure')
        delta = supplied.astype(bool)
    if selection == 'none' and fixed is None and not delta.all():
        raise ValueError('without selection all exposure indicators must be active')
    if selection == 'group' and any(delta[members].sum() > 1 for members in units):
        raise ValueError('initial delta may select at most one exposure per group')
    if fixed is not None:
        if 'r' in initial and not np.array_equal(np.asarray(initial['r']), fixed):
            raise ValueError('initial r conflicts with r_fixed')
        if 'delta' in initial and not np.array_equal(delta, fixed > 0):
            raise ValueError('initial delta conflicts with r_fixed')
        r, delta = fixed.copy(), fixed > 0
    elif 'r' in initial:
        r = _vector(initial['r'], m, 'initial r')
        if np.any(r[~delta] != 0) or any(not np.isfinite(prior.log_r(value)) for value in r[delta]):
            raise ValueError('initial r must be zero for excluded exposures and inside its prior support otherwise')
    else:
        r = np.array([prior.sample_r(rng) if selected else 0. for selected in delta])
    beta = _vector(initial.get('beta', np.zeros(p)), p, 'initial beta')
    v = _vector(initial.get('latent_white', np.zeros(n)), n, 'initial latent_white')
    w = _vector(initial.get('u_white', np.zeros(g)), g, 'initial u_white')
    lambda_ = _positive(initial.get('lambda', prior.lambda_mean), 'initial lambda')
    lambda_cluster = _positive(initial.get('lambda_cluster', prior.cluster_lambda_mean), 'initial lambda_cluster') if g else None
    dispersion = _positive(initial.get('dispersion', prior.dispersion_shape / prior.dispersion_rate), 'initial dispersion') if family == 'negative_binomial' else None
    baseline = None
    if family == 'survival':
        j = data['durations'].shape[1]
        baseline = _vector(initial.get('baseline_hazard', np.full(j, prior.baseline_shape / prior.baseline_rate)), j, 'initial baseline_hazard')
        if np.any(baseline <= 0):
            raise ValueError('initial baseline_hazard must be strictly positive')
    white = np.concatenate((v, beta / prior.beta_sd, w))
    if not np.isfinite(white).all():
        raise ValueError('initial standardized coefficients overflowed; rescale beta or its prior')
    return dict(white=white, r=r, delta=delta, pi=pi, lambda_=lambda_,
                lambda_cluster=lambda_cluster, dispersion=dispersion, baseline=baseline)


def _gp_factor(Z, r):
    """Dense GP factor with the prediction path's numerical-null convention.

    Eigenvalues at most 100*n*eps*largest_eigenvalue are numerical null modes.
    Materially negative values fail; no diagonal jitter or low-rank model
    approximation is introduced.
    """
    kernel = gaussian_kernel(Z, r=r)
    eigenvalues, eigenvectors = np.linalg.eigh((kernel + kernel.T) / 2)
    tolerance = 100 * len(kernel) * np.finfo(float).eps * eigenvalues.max()
    if eigenvalues.min() < -tolerance:
        raise np.linalg.LinAlgError('GP kernel is materially indefinite')
    retained = np.where(eigenvalues > tolerance, eigenvalues, 0.)
    return eigenvectors * np.sqrt(retained)


def _log_gamma_shape_rate(value, shape, rate):
    # Normalizing constants cancel for a fixed Gamma prior in a scalar MH ratio.
    if not np.isfinite(value) or value <= 0:
        return -np.inf
    return (shape - 1) * math.log(value) - rate * value


def _log_proposal(value, sd, rng):
    log_value = math.log(value)
    log_candidate = log_value + float(rng.normal(scale=sd))
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        candidate = float(np.exp(log_candidate))
    return candidate, log_candidate - log_value


def fit_generalized(y, Z, X=None, *, family='poisson', draws=1000, warmup=1000,
                    chains=4, thin=1, seed=None, standardize=True, intercept=True,
                    selection='none', groups=None, prior=None, r_fixed=None, missing='raise',
                    exposure_names=None, covariate_names=None, initial=None, proposal_scales=None,
                    cluster=None, offset=None, event=None, baseline_breaks=None):
    """Fit Poisson, NB2 or piecewise-exponential PH regression with a GP term."""
    start = time.perf_counter()
    if family not in ('poisson', 'negative_binomial', 'survival'):
        raise ValueError('generalized family must be poisson, negative_binomial or survival')
    draws, warmup = _count(draws, 'draws', 2), _count(warmup, 'warmup', 0)
    chains, thin = _count(chains, 'chains', 1), _count(thin, 'thin', 1)
    if seed is not None:
        seed = _count(seed, 'seed', 0)
    if selection not in ('none', 'component', 'group'):
        raise ValueError('selection must be none, component or group')
    if (selection == 'group') != (groups is not None):
        raise ValueError('groups are required only for group selection')
    family_defaults = dict(lambda_mean=1., lambda_sd=.5, cluster_lambda_mean=1., cluster_lambda_sd=.5)
    if prior is None:
        prior = Prior(**family_defaults)
    elif isinstance(prior, dict):
        prior = Prior(**{**family_defaults, **prior})
    if not isinstance(prior, Prior):
        raise ValueError('prior must be Prior, a dictionary or None')
    scales = {'r': .3, 'lambda': .3, 'dispersion': .3, 'cluster_lambda': .3}
    if proposal_scales is not None:
        if not isinstance(proposal_scales, dict) or set(proposal_scales) - set(scales):
            raise ValueError('proposal_scales accepts r, lambda, dispersion and cluster_lambda')
        if 'dispersion' in proposal_scales and family != 'negative_binomial':
            raise ValueError('dispersion proposal scale applies only to negative_binomial')
        if 'cluster_lambda' in proposal_scales and cluster is None:
            raise ValueError('cluster_lambda proposal scale requires cluster labels')
        scales.update(proposal_scales)
    scales = {key: _positive(value, f'proposal_scales[{key}]') for key, value in scales.items()}
    data = _prepare(y, Z, X, family, standardize, intercept, missing,
                    exposure_names, covariate_names, groups, cluster, offset, event, baseline_breaks)
    n, m = data['Z'].shape
    p, g = data['X'].shape[1], len(data['cluster_levels'])
    messages = []
    amplitude_priors = [('lambda', prior.lambda_mean, prior.lambda_sd)]
    if g:
        amplitude_priors.append(('lambda_cluster', prior.cluster_lambda_mean, prior.cluster_lambda_sd))
    for name, mean, sd in amplitude_priors:
        rate = mean / sd**2
        if rate <= 2:
            moment = 'first and second moments' if rate <= .5 else 'second moment'
            messages.append(f'The {name} Gamma prior rate is {rate:g}; the prior-predictive exp-Gaussian {moment} may be infinite. Finite-chain response summaries do not establish finite posterior moments.')
    if family == 'negative_binomial' and prior.dispersion_shape <= 1:
        messages.append('The dispersion Gamma prior shape is at most one, so its reciprocal mean and the prior-predictive count second moment may be infinite. Finite-chain summaries do not establish finite posterior moments.')
    if family == 'survival':
        if not data['event'].any():
            messages.append('No observed events: survival estimates depend strongly on the priors; this fit does not provide evidence of estimated event risk.')
        if intercept:
            messages.append('The survival intercept and baseline hazard share an overall scale; their separate estimates depend on their proper priors.')
    for message in messages:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    j = data['durations'].shape[1] if family == 'survival' else 0
    fixed = None
    if r_fixed is not None:
        fixed = _vector(r_fixed, m, 'r_fixed')
        if np.any(fixed < 0) or selection != 'none':
            raise ValueError('r_fixed must be nonnegative and disallows variable selection')
    units = ([np.array([index]) for index in range(m)] if selection == 'component' else
             [np.flatnonzero(np.asarray(data['groups']) == label) for label in dict.fromkeys(data['groups'])]
             if selection == 'group' else [])
    if 8 * (10 * n * n + n * j) > 512 * 1024**2:
        raise ValueError('dense generalized working storage exceeds the 512 MiB safety budget')
    saved_bytes = chains * draws * (8 * (p + m + 2 * n + g + j + 3 + bool(g)
                                        + (family == 'negative_binomial')) + m)
    if saved_bytes > 1024**3:
        raise ValueError('saved posterior arrays exceed the 1 GiB safety budget')
    shape = (chains, draws)
    samples = {name: np.empty(shape + (size,)) for name, size in
               (('beta', p), ('r', m), ('h', n), ('latent_white', n))}
    samples.update({name: np.empty(shape) for name in ('lambda', 'pi')})
    samples['sigma2'] = np.ones(shape)
    samples['delta'] = np.empty(shape + (m,), dtype=bool)
    if g:
        samples['u'] = np.empty(shape + (g,))
        samples['lambda_cluster'] = np.empty(shape)
    if family == 'negative_binomial':
        samples['dispersion'] = np.empty(shape)
    if family == 'survival':
        samples['baseline_hazard'] = np.empty(shape + (j,))
    sequence = np.random.SeedSequence(seed)
    chain_info = []
    for chain_index, child in enumerate(sequence.spawn(chains)):
        rng = np.random.default_rng(child)
        state = _initialize(initial, data, prior, family, selection, fixed, units, rng)
        factor = _gp_factor(data['Z'], state['r'])
        evaluations = dict(warmup=0, sampling=0)
        keys = ['lambda', 'r', 'selection', 'ess']
        if g:
            keys.append('cluster_lambda')
        if family == 'negative_binomial':
            keys.append('dispersion')
        counts = {key: dict(accepted=0, proposed=0, warmup_accepted=0, warmup_proposed=0) for key in keys}

        def components(white, root, lam, cluster_lam):
            with np.errstate(over='ignore', invalid='ignore'):
                h = math.sqrt(lam) * (root @ white[:n])
                beta = prior.beta_sd * white[n:n+p]
                u = math.sqrt(cluster_lam) * white[n+p:] if g else np.empty(0)
                eta = h + data['X'] @ beta
                if g:
                    eta = eta + u[data['cluster_codes']]
            return h, beta, u, eta

        def likelihood(white, root=None, lam=None, cluster_lam=None, dispersion=None, baseline=None):
            root = factor if root is None else root
            lam = state['lambda_'] if lam is None else lam
            cluster_lam = state['lambda_cluster'] if cluster_lam is None else cluster_lam
            _, _, _, eta = components(white, root, lam, cluster_lam)
            if family == 'survival':
                return survival_loglik(data['event'], data['event_bin'], data['durations'], eta,
                                       state['baseline'] if baseline is None else baseline)
            with np.errstate(over='ignore', invalid='ignore'):
                eta = eta + data['offset']
            if family == 'poisson':
                return poisson_loglik(data['y'], eta)
            return negative_binomial_loglik(data['y'], eta,
                                           state['dispersion'] if dispersion is None else dispersion)

        current_ll = likelihood(state['white'])
        if not np.isfinite(current_ll):
            raise ValueError('initial generalized state has nonfinite likelihood; rescale inputs or provide finite initial parameters')
        saved = 0
        for iteration in range(warmup + draws * thin):
            warm = iteration < warmup
            phase = 'warmup' if warm else 'sampling'
            evaluations[phase] += 1 if iteration == 0 else 0
            accepted_key, proposed_key = ('warmup_accepted', 'warmup_proposed') if warm else ('accepted', 'proposed')
            state['white'], current_ll, used = elliptical_slice(state['white'], likelihood, rng, current_loglik=current_ll)
            evaluations[phase] += used
            counts['ess'][proposed_key] += 1
            counts['ess'][accepted_key] += 1

            def accept(candidate_ll, adjustment, kind):
                counts[kind][proposed_key] += 1
                ratio = candidate_ll - current_ll + adjustment
                if not np.isfinite(candidate_ll) or np.isnan(ratio):
                    return False
                if math.log1p(-float(rng.random())) < min(0., ratio):
                    counts[kind][accepted_key] += 1
                    return True
                return False

            # Under a log random walk, add log(candidate/current) to the MH ratio.
            for name, scale_key, mean, sd in (
                ('lambda_', 'lambda', prior.lambda_mean, prior.lambda_sd),
                *((('lambda_cluster', 'cluster_lambda', prior.cluster_lambda_mean, prior.cluster_lambda_sd),) if g else ()),
            ):
                old = state[name]
                candidate, jacobian = _log_proposal(old, scales[scale_key], rng)
                adjustment = gamma_logpdf(candidate, mean, sd) - gamma_logpdf(old, mean, sd) + jacobian
                if np.isfinite(candidate) and candidate > 0 and np.isfinite(adjustment):
                    kwargs = {'lam' if name == 'lambda_' else 'cluster_lam': candidate}
                    candidate_ll = likelihood(state['white'], **kwargs)
                    evaluations[phase] += 1
                else:
                    candidate_ll = -np.inf
                if accept(candidate_ll, adjustment, scale_key):
                    state[name], current_ll = candidate, candidate_ll

            if fixed is None:
                # Independence proposals from the complete conditional selection
                # prior cancel both indicator and active-relevance prior ratios.
                for members in units:
                    proposed_r, proposed_delta = state['r'].copy(), state['delta'].copy()
                    proposed_r[members], proposed_delta[members] = 0., False
                    if rng.random() < state['pi']:
                        member = int(rng.choice(members))
                        proposed_r[member] = prior.sample_r(rng)
                        proposed_delta[member] = True
                    try:
                        proposed_factor = _gp_factor(data['Z'], proposed_r)
                        candidate_ll = likelihood(state['white'], root=proposed_factor)
                        evaluations[phase] += 1
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                        candidate_ll = -np.inf
                    if accept(candidate_ll, 0., 'selection'):
                        state['r'], state['delta'], factor, current_ll = proposed_r, proposed_delta, proposed_factor, candidate_ll
                for index in np.flatnonzero(state['delta']):
                    old = state['r'][index]
                    candidate, jacobian = _log_proposal(old, scales['r'], rng)
                    adjustment = prior.log_r(candidate) - prior.log_r(old) + jacobian
                    candidate_ll = -np.inf
                    if np.isfinite(candidate) and candidate > 0 and np.isfinite(adjustment):
                        proposed_r = state['r'].copy()
                        proposed_r[index] = candidate
                        try:
                            proposed_factor = _gp_factor(data['Z'], proposed_r)
                            candidate_ll = likelihood(state['white'], root=proposed_factor)
                            evaluations[phase] += 1
                        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                            candidate_ll = -np.inf
                    if accept(candidate_ll, adjustment, 'r'):
                        state['r'], factor, current_ll = proposed_r, proposed_factor, candidate_ll
            if units:
                active = sum(bool(state['delta'][members].any()) for members in units)
                state['pi'] = float(rng.beta(prior.inclusion_alpha + active,
                                             prior.inclusion_beta + len(units) - active))
                if not 0 < state['pi'] < 1:
                    raise FloatingPointError('inclusion-probability update reached a floating-point boundary; rescale its prior')
            if family == 'negative_binomial':
                old = state['dispersion']
                candidate, jacobian = _log_proposal(old, scales['dispersion'], rng)
                adjustment = (_log_gamma_shape_rate(candidate, prior.dispersion_shape, prior.dispersion_rate)
                              - _log_gamma_shape_rate(old, prior.dispersion_shape, prior.dispersion_rate) + jacobian)
                candidate_ll = -np.inf
                if np.isfinite(candidate) and candidate > 0 and np.isfinite(adjustment):
                    candidate_ll = likelihood(state['white'], dispersion=candidate)
                    evaluations[phase] += 1
                if accept(candidate_ll, adjustment, 'dispersion'):
                    state['dispersion'], current_ll = candidate, candidate_ll
            if family == 'survival':
                _, _, _, eta = components(state['white'], factor, state['lambda_'], state['lambda_cluster'])
                shapes, rates = baseline_posterior_parameters(data['event'], data['event_bin'], data['durations'], eta,
                                                               prior.baseline_shape, prior.baseline_rate)
                baseline = rng.gamma(shapes, 1 / rates)
                if not np.isfinite(baseline).all() or np.any(baseline <= 0):
                    raise FloatingPointError('baseline Gamma update reached a floating-point boundary; rescale times or prior')
                state['baseline'] = baseline
                current_ll = likelihood(state['white'])
                evaluations[phase] += 1
                if not np.isfinite(current_ll):
                    raise FloatingPointError('baseline update produced nonfinite likelihood')
            if not warm and (iteration - warmup + 1) % thin == 0:
                h, beta, u, _ = components(state['white'], factor, state['lambda_'], state['lambda_cluster'])
                values = dict(beta=beta, r=state['r'], h=h, latent_white=state['white'][:n],
                              delta=state['delta'], pi=state['pi'], lambda_=state['lambda_'])
                values['lambda'] = values.pop('lambda_')
                if g:
                    values.update(u=u, lambda_cluster=state['lambda_cluster'])
                if family == 'negative_binomial':
                    values['dispersion'] = state['dispersion']
                if family == 'survival':
                    values['baseline_hazard'] = state['baseline']
                for name, value in values.items():
                    samples[name][chain_index, saved] = value
                saved += 1
        for record in counts.values():
            for phase_prefix in ('', 'warmup_'):
                denominator = record[phase_prefix + 'proposed']
                record[phase_prefix + 'rate'] = record[phase_prefix + 'accepted'] / denominator if denominator else None
        chain_info.append(dict(chain=chain_index, acceptance=counts, likelihood_evaluations=evaluations,
                               seed_spawn_key=list(child.spawn_key)))
    settings = dict(family=family, draws=draws, warmup=warmup, chains=chains, thin=thin,
                    seed=seed, standardize=bool(standardize), intercept=bool(intercept),
                    selection=selection, prior=asdict(prior), proposal_scales=scales,
                    r_fixed=None if fixed is None else fixed.tolist(), clustered=bool(g),
                    offset_used=offset is not None, beta_prior='normal',
                    baseline_convention='right_closed' if family == 'survival' else None)
    sampling_info = dict(runtime_seconds=time.perf_counter() - start, chains=chain_info,
                         algorithm='noncentered ESS/log-MH/Gibbs', proposal_adaptation=False,
                         seed_entropy=int(sequence.entropy), warnings=messages)
    from .result import BKMRFit
    return BKMRFit(data, samples, settings, sampling_info)
