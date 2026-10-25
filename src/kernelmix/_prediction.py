"""Conditional GP calculations for explicitly saved latent functions."""
import numpy as np

from ._kernel import gaussian_kernel, psd_factor


def _check_support(h, basis):
    residual = h - basis @ (basis.T @ h)
    if np.max(np.abs(residual), initial=0) > 1e-8 * max(1., np.max(np.abs(h), initial=0)):
        raise ValueError('saved h lies outside the numerical training-kernel support')


def conditional_saved_h(training, new, h, r, amplitude):
    """Condition a Gaussian process on saved training h, including PSD kernels.

    Eigenvalues below 100*n*machine-epsilon*largest-eigenvalue define numerical
    null directions. No diagonal jitter is added. Observed training profiles
    retain their saved h exactly. This is floating-point rank handling, not a
    low-rank approximation selected for computation speed.
    """
    kernel = gaussian_kernel(training, r=r)
    eig, vectors = np.linalg.eigh(kernel)
    tolerance = 100 * len(kernel) * np.finfo(float).eps * max(1., eig.max())
    if eig.min() < -tolerance:
        raise np.linalg.LinAlgError('training kernel is materially indefinite')
    keep = eig > tolerance
    basis = vectors[:, keep]
    _check_support(h, basis)
    cross = gaussian_kernel(new, training, r=r)
    projected = cross @ basis
    mean = (projected / eig[keep]) @ (basis.T @ h)
    covariance = amplitude * (gaussian_kernel(new, r=r) - (projected/eig[keep]) @ projected.T)
    covariance = (covariance + covariance.T)/2
    active = np.asarray(r) > 0
    for j, profile in enumerate(new):
        matches = np.flatnonzero(np.all(training[:, active] == profile[active], axis=1))
        if len(matches):
            mean[j] = h[matches[0]]
            covariance[j, :] = 0
            covariance[:, j] = 0
    factor = psd_factor(covariance)
    return mean, factor @ factor.T
