"""Exact count and piecewise-exponential survival log likelihoods.

Count predictors are log means (including any offset). Survival predictors
are log relative hazards. All likelihoods retain parameter-dependent
normalization terms; nonfinite candidate predictors return minus infinity.
"""

import numpy as np
from scipy.special import betaln, gammaln


def _vector(value, name, n=None):
    a = np.asarray(value, dtype=float)
    if a.ndim != 1 or (n is not None and len(a) != n):
        raise ValueError(f'{name} must be a vector with one value per observation')
    return a


def _counts(y):
    values = _vector(y, 'y')
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError('counts must be finite nonnegative integers')
    return values


def _finite_sum(value):
    total = float(np.sum(value))
    return total if np.isfinite(total) else -np.inf


def poisson_loglik(y, eta):
    """Sum log Poisson PMFs for a log mean, including log-factorials."""
    y = _counts(y)
    eta = _vector(eta, 'eta', len(y))
    if not np.isfinite(eta).all():
        return -np.inf
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        return _finite_sum(y * eta - np.exp(eta) - gammaln(y + 1))


def negative_binomial_loglik(y, eta, dispersion):
    """NB2 log PMF with mean exp(eta) and variance mu + mu**2/dispersion.

    The beta-function identity avoids subtracting two large log gamma terms.
    Log-add-exp expressions avoid exponentiating the count mean.
    """
    y = _counts(y)
    eta = _vector(eta, 'eta', len(y))
    if not np.isscalar(dispersion) or not np.isfinite(dispersion) or dispersion <= 0:
        return -np.inf
    if not np.isfinite(eta).all():
        return -np.inf
    log_d = np.log(dispersion)
    with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
        normalizer = -betaln(dispersion, y + 1) - np.logaddexp(log_d, np.log(y))
        terms = (normalizer - dispersion * np.logaddexp(0, eta - log_d)
                 - y * np.logaddexp(0, log_d - eta))
        return _finite_sum(terms)


def _breaks(value):
    cuts = np.asarray(value, dtype=float)
    if cuts.ndim != 1 or not np.isfinite(cuts).all() or np.any(cuts <= 0) or np.any(np.diff(cuts) <= 0):
        raise ValueError('baseline_breaks must be finite positive strictly increasing cutpoints')
    return cuts.copy()


def interval_durations(times, breaks):
    """Time spent in intervals (0,b1], ..., (bJ,infinity).

    Zero times are allowed for prediction. Times are never sorted or clipped.
    """
    times = _vector(times, 'times')
    if not np.isfinite(times).all() or np.any(times < 0):
        raise ValueError('times must be finite and nonnegative')
    cuts = _breaks(breaks)
    if 8 * len(times) * (len(cuts) + 1) > 512 * 1024**2:
        raise ValueError('interval durations exceed the 512 MiB allocation budget')
    left = np.concatenate(([0.], cuts))
    right = np.concatenate((cuts, [np.inf]))
    return np.maximum(np.minimum(times[:, None], right) - left, 0.)


def prepare_survival(time, event, baseline_breaks=None):
    """Create interval data; boundary events belong to the left interval.

    Default cutpoints are unique event-time deciles below the largest retained
    observed time. No observed events gives one exponential interval.
    """
    time = _vector(time, 'time')
    event = _vector(event, 'event', len(time))
    if not len(time) or not np.isfinite(time).all() or np.any(time <= 0):
        raise ValueError('survival times must be positive and finite')
    if not np.isfinite(event).all() or not np.isin(event, [0, 1]).all():
        raise ValueError('event must contain only zero and one')
    if baseline_breaks is None:
        events = time[event == 1]
        cuts = np.unique(np.quantile(events, np.arange(1, 10) / 10)) if len(events) else np.empty(0)
        cuts = cuts[(cuts > 0) & (cuts < time.max())]
    else:
        cuts = _breaks(baseline_breaks)
    return dict(event=event.astype(int), baseline_breaks=cuts,
                durations=interval_durations(time, cuts),
                event_bin=np.searchsorted(cuts, time, side='left'))


def _survival_arrays(event, event_bin, durations, eta, baseline_hazard=None):
    event = _vector(event, 'event')
    eta = _vector(eta, 'eta', len(event))
    bins = np.asarray(event_bin)
    durations = np.asarray(durations, dtype=float)
    if durations.ndim != 2 or durations.shape[0] != len(event) or durations.shape[1] < 1:
        raise ValueError('durations must have one row per observation and one column per interval')
    if not np.isfinite(durations).all() or np.any(durations < 0):
        raise ValueError('durations must be finite and nonnegative')
    if not np.isfinite(event).all() or not np.isin(event, [0, 1]).all():
        raise ValueError('event must contain only zero and one')
    if (bins.shape != event.shape or not np.issubdtype(bins.dtype, np.integer)
            or np.any(bins < 0) or np.any(bins >= durations.shape[1])):
        raise ValueError('event_bin must index one valid interval per observation')
    if baseline_hazard is not None:
        hazard = _vector(baseline_hazard, 'baseline_hazard', durations.shape[1])
        if not np.isfinite(hazard).all() or np.any(hazard <= 0):
            raise ValueError('baseline_hazard must be positive and finite')
    return event, bins, durations, eta


def survival_loglik(event, event_bin, durations, eta, baseline_hazard):
    """Full right-censored piecewise PH log likelihood, not partial likelihood."""
    event, bins, durations, eta = _survival_arrays(event, event_bin, durations, eta, baseline_hazard)
    if not np.isfinite(eta).all():
        return -np.inf
    hazard = np.asarray(baseline_hazard, dtype=float)
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        terms = event * (np.log(hazard[bins]) + eta) - np.exp(eta) * (durations @ hazard)
        return _finite_sum(terms)


def baseline_posterior_parameters(event, event_bin, durations, eta, shape, rate):
    """Independent conditional Gamma shapes/rates for baseline hazard Gibbs draws."""
    event, bins, durations, eta = _survival_arrays(event, event_bin, durations, eta)
    if not np.isfinite(shape) or not np.isfinite(rate) or min(shape, rate) <= 0:
        raise ValueError('baseline Gamma shape and rate must be positive and finite')
    if not np.isfinite(eta).all():
        raise FloatingPointError('nonfinite survival predictor in baseline update')
    counts = np.bincount(bins, weights=event, minlength=durations.shape[1])
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        rates = rate + durations.T @ np.exp(eta)
    if not np.isfinite(rates).all() or np.any(rates <= 0):
        raise FloatingPointError('baseline Gamma rates overflowed; rescale times or predictors')
    return shape + counts, rates
