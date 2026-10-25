"""Piecewise cause-specific survival likelihoods and conditional curves.

Likelihoods condition on survival to entry given the supplied latent predictors.
The caller defines predictor/random-effect priors for the observed entrant
cohort. These functions do not infer a population frailty distribution under
marginal left truncation. Predictor histories are supplied piecewise constants;
no interpolation or future predictor values are invented.
"""
import math

import numpy as np
from scipy.special import logsumexp

from ._clusters import encode_clusters

_LOG_MAX = math.log(np.finfo(float).max)
_LOG_EPS = math.log(np.finfo(float).eps)
_BUDGET = 512 * 1024 ** 2


def _vector(value, name, n=None, *, finite=True):
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must be a numeric vector') from exc
    if array.ndim != 1 or not len(array) or (n is not None and len(array) != n):
        raise ValueError(f'{name} must have one value per row and cannot be empty')
    if np.isnan(array).any() or (finite and not np.isfinite(array).all()):
        raise ValueError(f'{name} contains missing or invalid nonfinite values')
    return array.copy()


def _breaks(value):
    cuts = np.asarray(value, dtype=float)
    if (cuts.ndim != 1 or not np.isfinite(cuts).all() or (cuts <= 0).any()
            or (np.diff(cuts) <= 0).any()):
        raise ValueError('baseline_breaks must be finite positive strictly increasing internal cuts')
    return cuts.copy()


def _times(value):
    times = _vector(value, 'times')
    if (times < 0).any() or (np.diff(times) <= 0).any():
        raise ValueError('times must be nonnegative and strictly increasing')
    return times


def _duration(start, stop, cuts):
    left = np.r_[0., cuts]
    right = np.r_[cuts, np.inf]
    return np.maximum(np.minimum(stop[:, None], right) - np.maximum(start[:, None], left), 0.)


def _subjects(subject, n):
    if subject is None:
        return np.arange(n, dtype=int), list(range(n))
    return encode_clusters(subject, n)


def _history_rows(codes, starts, stops, event=None):
    """Validate each subject's input-order contiguous history; do not sort it."""
    order = np.argsort(codes, kind='stable')
    pointers = np.r_[0, np.cumsum(np.bincount(codes))].astype(int)
    terminal = np.zeros(len(codes), dtype=bool)
    terminal[order[pointers[1:] - 1]] = True
    earlier, later = order[:-1], order[1:]
    same_subject = codes[earlier] == codes[later]
    if not np.array_equal(starts[later[same_subject]], stops[earlier[same_subject]]):
        raise ValueError('subject histories must be chronological, contiguous and nonoverlapping')
    if event is not None and np.any(event[~terminal] != 0):
        raise ValueError('an event may occur only in the last row for each subject')
    return terminal, order, pointers


def prepare_extended_survival(time, event, *, entry=None, upper=None, subject=None,
                              n_causes=None, baseline_breaks=None):
    """Validate exact/right/left/interval observations and risk histories.

    ``event`` is zero for right censoring, 1..K for an observed cause, or -1
    for an unknown cause in a finite interval. ``time`` is the exact/censoring
    time or interval lower bound. ``upper`` is the finite interval upper bound,
    equal to time for exact observations, or infinity for right censoring.
    Returned right-censored upper bounds are canonicalized to infinity.

    Repeated subjects encode successive (start, stop] rows using entry/time.
    Earlier rows are risk contributions with event 0; only the last may have an
    event or finite censoring interval. Missing-row selection belongs to caller.
    """
    lower = _vector(time, 'time')
    n = len(lower)
    status = _vector(event, 'event', n)
    if (status != np.floor(status)).any() or (status < -1).any():
        raise ValueError('event must contain integer cause codes0..K or-1 for unknown interval cause')
    maximum = float(status.max())
    if maximum >= np.iinfo(np.int64).max:
        raise ValueError('event cause codes exceed supported integer range')
    if n_causes is None:
        if np.any(status == -1) and maximum < 1:
            raise ValueError('n_causes is required when observed interval causes are all unknown')
        causes = max(1, int(maximum))
    else:
        if (isinstance(n_causes, (bool, np.bool_))
                or not isinstance(n_causes, (int, np.integer)) or n_causes < 1):
            raise ValueError('n_causes must be a positive integer')
        causes = int(n_causes)
    if maximum > causes:
        raise ValueError('event contains a cause exceeding n_causes')
    status = status.astype(int)
    starts = np.zeros(n) if entry is None else _vector(entry, 'entry', n)
    if (starts < 0).any() or (lower < 0).any():
        raise ValueError('entry and time must be nonnegative')
    if upper is None:
        ends = np.where(status == 0, np.inf, lower)
    else:
        ends = _vector(upper, 'upper', n, finite=False)
        if np.isneginf(ends).any() or np.any(ends < lower):
            raise ValueError('upper must be at least time; only right censoring may use infinity')
    interval = np.isfinite(ends) & (ends > lower)
    if np.any((status == 0) & interval):
        raise ValueError('event0 means right censoring, not a finite censoring interval')
    if np.any(np.isinf(ends) & (status != 0)):
        raise ValueError('infinite upper bounds are allowed only for right censoring')
    if np.any((status == -1) & ~interval):
        raise ValueError('unknown causes require a finite nonzero censoring interval')
    if np.any(lower[~interval] <= starts[~interval]):
        raise ValueError('exact and right-censored times must be strictly after entry')
    if np.any(lower[interval] < starts[interval]):
        raise ValueError('interval lower bounds must be at least entry')
    ends[status == 0] = np.inf
    codes, levels = _subjects(subject, n)
    terminal, order, pointers = _history_rows(codes, starts, lower, status)
    if np.any(interval & ~terminal):
        raise ValueError('finite censoring intervals are allowed only in terminal subject rows')
    if baseline_breaks is None:
        exact_known = lower[(status > 0) & ~interval]
        if len(exact_known):
            maximum_followup = max(float(lower.max()), float(np.max(ends[np.isfinite(ends)])))
            cuts = np.unique(np.quantile(exact_known, np.arange(1, 10) / 10))
            cuts = cuts[(cuts > 0) & (cuts < maximum_followup)]
        else:
            cuts = np.empty(0)
    else:
        cuts = _breaks(baseline_breaks)
    if 8 * (2 * n * (len(cuts) + 1) + 8 * n) > _BUDGET:
        raise ValueError('survival duration arrays exceed the512MiB working-memory limit')
    durations = _duration(starts, lower, cuts)
    interval_durations = np.zeros_like(durations)
    if interval.any():
        interval_durations[interval] = _duration(lower[interval], ends[interval], cuts)
    return dict(time=lower, event=status, entry=starts, upper=ends,
                subject_codes=codes, subject_levels=levels,
                baseline_breaks=cuts, durations=durations,
                interval_durations=interval_durations,
                event_bin=np.searchsorted(cuts, lower, side='left'),
                is_interval=interval, terminal=terminal,
                subject_order=order, subject_ptr=pointers, n_causes=causes, K=causes)


def _log_integrated_hazard(log_rate, widths):
    selected = widths > 0
    if not np.any(selected):
        return -np.inf
    return float(logsumexp(log_rate[selected] + np.log(widths[selected])))


def _hazard_from_log(log_hazard):
    if log_hazard == -np.inf:
        return 0.
    if log_hazard > _LOG_MAX:
        return np.inf
    return math.exp(log_hazard)


def _log_failure(log_hazard):
    """log(1-exp(-H)) from log(H), preserving tiny interval probabilities."""
    if log_hazard < _LOG_EPS:
        # The relative correction to H is below machine precision; returning
        # log(H) also avoids underflow when H itself is unrepresentable.
        return log_hazard
    if log_hazard > _LOG_MAX:
        return 0.
    return math.log(-math.expm1(-math.exp(log_hazard)))


def extended_survival_loglik(eta, baseline, data):
    """Full conditional log likelihood with cause-specific constant hazards.

    Entry conditioning is rowwise given eta. Earlier subject rows accumulate
    risk; a terminal interval row adds an integrated event probability. This
    gives one entry-conditioned subject likelihood when its contiguous rows
    are multiplied. Candidate nonfinite/invalid predictors or hazards return
    minus infinity, never clipped likelihoods or rates.
    """
    predictors = np.asarray(eta, dtype=float)
    hazards = np.asarray(baseline, dtype=float)
    n, k, j = len(data['time']), int(data['n_causes']), len(data['baseline_breaks']) + 1
    if (predictors.shape != (n, k) or hazards.shape != (k, j)
            or not np.isfinite(predictors).all() or not np.isfinite(hazards).all()
            or (hazards <= 0).any()):
        return -np.inf
    log_baseline = np.log(hazards)
    total = 0.
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        for i in range(n):
            log_cause = predictors[i, :, None] + log_baseline
            if not np.isfinite(log_cause).all():
                return -np.inf
            # Center eta before adding baseline logs so cause fractions do
            # not lose the baseline contrast under a huge common predictor.
            reference = float(np.max(predictors[i]))
            relative_cause = (predictors[i] - reference)[:, None] + log_baseline
            relative_total = logsumexp(relative_cause, axis=0)
            log_rate = reference + relative_total
            log_fraction = relative_cause - relative_total
            before = _hazard_from_log(_log_integrated_hazard(log_rate, data['durations'][i]))
            if not np.isfinite(before):
                return -np.inf
            status = int(data['event'][i])
            if not data['is_interval'][i]:
                row = -before
                if status > 0:
                    row += log_cause[status - 1, data['event_bin'][i]]
            else:
                widths = data['interval_durations'][i]
                if status == -1:
                    row = -before + _log_failure(_log_integrated_hazard(log_rate, widths))
                else:
                    pieces = []
                    elapsed = 0.
                    for b in np.flatnonzero(widths > 0):
                        log_mass = log_rate[b] + math.log(widths[b])
                        pieces.append(-elapsed + log_fraction[status - 1, b]
                                      + _log_failure(log_mass))
                        elapsed += _hazard_from_log(log_mass)
                        if not np.isfinite(elapsed):
                            break
                    row = -before + float(logsumexp(pieces))
            total += row
            if not np.isfinite(total):
                return -np.inf
    return float(total)


def _prediction_inputs(eta, baseline, breaks, times):
    predictors = np.asarray(eta, dtype=float)
    cuts = _breaks(breaks)
    hazards = np.asarray(baseline, dtype=float)
    if predictors.ndim != 2 or not predictors.shape[0] or not predictors.shape[1] or not np.isfinite(predictors).all():
        raise ValueError('eta must be a nonempty finite rows-by-causes matrix')
    k = predictors.shape[1]
    if hazards.shape != (k, len(cuts) + 1) or not np.isfinite(hazards).all() or (hazards <= 0).any():
        raise ValueError('baseline must be positive finite causes-by-intervals hazards')
    return predictors, np.log(hazards), cuts, _times(times)


def _curve_budget(n, t, k):
    if 8 * (n * t * (1 + 2 * k) + 4 * n * k) > _BUDGET:
        raise ValueError('survival curves exceed the512MiB output-memory limit')


def _advance(survival, incidence, log_hazards, width):
    """Exact competing-risk mass increment on one constant-hazard segment."""
    log_total = float(logsumexp(log_hazards))
    integrated = _hazard_from_log(log_total + math.log(width))
    failure = -math.expm1(-integrated)
    incidence = incidence + survival * failure * np.exp(log_hazards - log_total)
    survival = survival * math.exp(-integrated)
    return survival, incidence


def _finite_hazard(log_hazards):
    if np.any(log_hazards > _LOG_MAX) or not np.isfinite(log_hazards).all():
        raise FloatingPointError('predicted hazard exceeds floating-point range; rescale predictors or priors')
    return np.exp(log_hazards)


def survival_curves(eta, baseline, breaks, times, *, entry=0):
    """Return (S, CIF, hazard) arrays conditional on survival to each entry time.

    S is survival probability; CIF is cumulative incidence by cause.
    Shapes are profiles-by-times and profiles-by-times-by-causes. Before entry,
    S=1,CIF=0 and hazards=0. At time zero the first baseline hazard is used.
    Other boundaries are right closed. Beyond the last baseline cut, the final
    hazard is extended; this is an explicit piecewise-baseline extrapolation.
    """
    predictors, log_baseline, cuts, requested = _prediction_inputs(eta, baseline, breaks, times)
    n, k = predictors.shape
    starts = np.asarray(entry, dtype=float)
    if starts.ndim == 0:
        starts = np.full(n, starts.item())
    if starts.shape != (n,) or not np.isfinite(starts).all() or (starts < 0).any():
        raise ValueError('entry must be a nonnegative finite scalar or one value per profile')
    _curve_budget(n, len(requested), k)
    survival = np.ones((n, len(requested)))
    cif = np.zeros((n, len(requested), k))
    hazard = np.zeros_like(cif)
    for i in range(n):
        current, s, ci = float(starts[i]), 1., np.zeros(k)
        for t, target in enumerate(requested):
            if target < starts[i]:
                continue
            split = np.r_[cuts[(cuts > current) & (cuts < target)], target]
            for endpoint in split:
                if endpoint > current:
                    b = np.searchsorted(cuts, current, side='right')
                    logs = predictors[i] + log_baseline[:, b]
                    s, ci = _advance(s, ci, logs, float(endpoint - current))
                    current = float(endpoint)
            survival[i, t], cif[i, t] = s, ci
            b = np.searchsorted(cuts, target, side='left')
            hazard[i, t] = _finite_hazard(predictors[i] + log_baseline[:, b])
    return survival, cif, hazard


def survival_path_curves(eta, baseline, breaks, times, *, start, stop, subject):
    """Conditional curves for supplied contiguous (start, stop] predictor paths.

    Return (subject_levels, S, CIF, hazard). Subject order follows first occurrence;
    rows within subjects must already be chronological. Each supplied row has
    constant eta; no interpolation is used. Predictions after the last supplied
    stop fail instead of extending the last predictor row. Before first start,
    S=1,CIF=0,hazard=0. The subject's first start is its conditioning entry time.
    """
    predictors, log_baseline, cuts, requested = _prediction_inputs(eta, baseline, breaks, times)
    n, k = predictors.shape
    starts, stops = _vector(start, 'start', n), _vector(stop, 'stop', n)
    if (starts < 0).any() or np.any(stops <= starts):
        raise ValueError('paths require finite0<=start<stop for every row')
    codes, levels = _subjects(subject, n)
    _, order, pointers = _history_rows(codes, starts, stops)
    _curve_budget(len(levels), len(requested), k)
    survival = np.ones((len(levels), len(requested)))
    cif = np.zeros((len(levels), len(requested), k))
    hazard = np.zeros_like(cif)
    for i in range(len(levels)):
        rows = order[pointers[i]:pointers[i + 1]]
        if requested[-1] > stops[rows[-1]]:
            raise ValueError('requested times extend beyond a supplied subject path; supply future rows explicitly')
        current, s, ci = float(starts[rows[0]]), 1., np.zeros(k)
        boundaries = np.unique(np.r_[cuts, stops[rows]])
        for t, target in enumerate(requested):
            if target < starts[rows[0]]:
                continue
            split = np.r_[boundaries[(boundaries > current) & (boundaries < target)], target]
            for endpoint in split:
                if endpoint > current:
                    # Immediately to the right of a path/baseline boundary,
                    # integration uses the next interval. Reporting at the
                    # boundary itself uses the preceding right-closed row.
                    row = rows[np.searchsorted(stops[rows], current, side='right')]
                    b = np.searchsorted(cuts, current, side='right')
                    s, ci = _advance(s, ci, predictors[row] + log_baseline[:, b], float(endpoint - current))
                    current = float(endpoint)
            survival[i, t], cif[i, t] = s, ci
            row = rows[np.searchsorted(stops[rows], target, side='left')]
            b = np.searchsorted(cuts, target, side='left')
            hazard[i, t] = _finite_hazard(predictors[row] + log_baseline[:, b])
    return levels, survival, cif, hazard
