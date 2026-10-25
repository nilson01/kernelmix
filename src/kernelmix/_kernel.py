"""Dense Gaussian kernels and conditional latent-function moments."""
import numpy as np
from scipy.linalg import cho_solve


def _matrix(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or not np.isfinite(a).all() or not a.shape[0] or not a.shape[1]:
        raise ValueError(f'{name} must be a nonempty finite matrix')
    return a


def gaussian_kernel(Z1, Z2=None, r=None):
    """Return exp(-sum_j r[j]*(Z1[:,j]-Z2[:,j])**2).

    Relevance values are nonnegative; all-zero relevance gives a constant kernel.
    """
    a = _matrix(Z1, 'Z1')
    b = a if Z2 is None else _matrix(Z2, 'Z2')
    if a.shape[1] != b.shape[1]:
        raise ValueError('kernel inputs must have equal column counts')
    weights = np.ones(a.shape[1]) if r is None else np.asarray(r, dtype=float)
    if weights.shape != (a.shape[1],) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('r must be a finite nonnegative vector, one value per exposure')
    # Accumulation avoids allocating an n1*n2*M tensor; direct differences
    # avoid cancellation from the squared-norm identity for near-duplicate rows.
    distance = np.zeros((len(a), len(b)))
    with np.errstate(over='ignore', invalid='ignore'):
        for j in np.flatnonzero(weights):
            distance += weights[j] * np.square(a[:, j, None] - b[None, :, j])
        kernel = np.exp(-distance)
    if not np.isfinite(kernel).all():
        raise ValueError('kernel calculation overflowed; rescale inputs')
    return kernel


def covariance_components(Z, r, lambda_):
    """Return (kernel, I + lambda*kernel, lower Cholesky), without jitter."""
    if not np.isscalar(lambda_) or not np.isfinite(lambda_) or lambda_ <= 0:
        raise ValueError('lambda must be finite and positive')
    kernel = gaussian_kernel(Z, r=r)
    covariance = lambda_ * kernel
    covariance.flat[::len(kernel) + 1] += 1.0
    if not np.isfinite(covariance).all():
        raise ValueError('covariance overflowed')
    try:
        lower = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError('I + lambda*K lost positive definiteness at floating-point precision; rescale or reduce lambda') from exc
    return kernel, covariance, lower


def psd_factor(covariance):
    """Eigenfactor for PSD draws; clip roundoff only, reject real indefiniteness."""
    value = np.asarray(covariance, dtype=float)
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.isfinite(value).all():
        raise ValueError('covariance must be a finite square matrix')
    value = (value + value.T) / 2
    eig, vectors = np.linalg.eigh(value)
    tolerance = 100 * np.finfo(float).eps * max(1, len(value)) * max(1.0, np.max(np.abs(eig), initial=0))
    if np.min(eig, initial=0) < -tolerance:
        raise np.linalg.LinAlgError('conditional covariance is materially indefinite')
    return vectors * np.sqrt(np.maximum(eig, 0))


def conditional_h(Z, Znew, y, X, beta, r, lambda_, sigma2):
    """Return joint conditional latent GP mean/covariance at standardized Znew."""
    z = _matrix(Z, 'Z')
    new = _matrix(Znew, 'Znew')
    outcome, design, coefficients = np.asarray(y, dtype=float), np.asarray(X, dtype=float), np.asarray(beta, dtype=float)
    if outcome.shape != (len(z),) or design.ndim != 2 or design.shape[0] != len(z) or coefficients.shape != (design.shape[1],):
        raise ValueError('y, X, and beta have incompatible shapes')
    if any(not np.isfinite(v).all() for v in (outcome, design, coefficients)):
        raise ValueError('y, X, and beta must be finite')
    if not np.isscalar(sigma2) or not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError('sigma2 must be finite and positive')
    _, _, lower = covariance_components(z, r, lambda_)
    cross = gaussian_kernel(new, z, r)
    solved = cho_solve((lower, True), cross.T, check_finite=False)
    mean = lambda_ * cross @ cho_solve((lower, True), outcome - design @ coefficients, check_finite=False)
    covariance = sigma2 * lambda_ * (gaussian_kernel(new, r=r) - lambda_ * cross @ solved)
    covariance = (covariance + covariance.T) / 2
    # Check here as well as at sampling, so moment-only prediction cannot hide
    # a materially indefinite result. Retain eigen-projected roundoff repair.
    factor = psd_factor(covariance)
    covariance = factor @ factor.T
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise FloatingPointError('conditional moments overflowed')
    return mean, covariance
