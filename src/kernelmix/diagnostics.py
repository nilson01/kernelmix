"""Rank-normalized split R-hat and autocorrelation-based effective sample sizes."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def _split(values):
    half = values.shape[1] // 2
    return np.concatenate((values[:, :half], values[:, -half:]), axis=0)


def _rank_normalize(values):
    ranks = rankdata(values.ravel(), method="average").reshape(values.shape)
    return ndtri((ranks - 0.375) / (values.size + 0.25))


def _rhat(values):
    n = values.shape[1]
    within = np.mean(np.var(values, axis=1, ddof=1))
    between = n * np.var(np.mean(values, axis=1), ddof=1)
    if within == 0:
        return np.nan if between == 0 else np.inf
    return float(np.sqrt(((n - 1) * within / n + between / n) / within))


def _ess(values):
    """Geyer initial positive, monotone pair estimate, capped at sample count."""
    m, n = values.shape
    centered = values - values.mean(axis=1, keepdims=True)
    size = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(centered, n=size, axis=1)
    acov = np.fft.irfft(transformed * np.conjugate(transformed), n=size, axis=1)
    acov = acov[:, :n] / n
    within = np.mean(np.var(values, axis=1, ddof=1))
    between = n * np.var(values.mean(axis=1), ddof=1)
    variance = (n - 1) * within / n + between / n
    if not np.isfinite(variance) or variance <= 0 or within == 0:
        return np.nan
    rho = 1 - (within - acov.mean(axis=0)) / variance
    rho[0] = 1.0
    pairs = []
    for lag in range(0, n - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair <= 0:
            break
        pairs.append(min(pair, pairs[-1]) if pairs else pair)
    if not pairs:
        return float(m * n)
    tau = max(1.0, -1 + 2 * sum(pairs))
    return float(min(m * n, m * n / tau))


def diagnose(values):
    """Check sampling diagnostics for one parameter shaped (chains, saved draws).

    R-hat compares variation within and between split chains, using both ranks
    and ranks of distances from the median. Effective sample size (ESS) estimates how many independent
    draws would give similar precision. Bulk ESS uses ranks; tail ESS uses
    indicators at the 5th and 95th percentiles. These checks do not prove
    convergence or establish that the statistical model fits the data.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("diagnostics require a finite (chains, draws) array")
    if values.shape[1] < 4:
        return {"rhat": np.nan, "ess_bulk": np.nan, "ess_tail": np.nan}
    split = _split(values)
    ranks = _rank_normalize(split)
    folded = _rank_normalize(np.abs(split - np.median(split)))
    rhat = max(_rhat(ranks), _rhat(folded)) if values.shape[0] >= 2 else np.nan
    low, high = np.quantile(split, [0.05, 0.95])
    tails = [_ess((split <= low).astype(float)), _ess((split >= high).astype(float))]
    tail = min(tails) if np.isfinite(tails).all() else np.nan
    return {"rhat": float(rhat), "ess_bulk": _ess(ranks), "ess_tail": float(tail)}
