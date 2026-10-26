"""Scalar priors for Bayesian kernel machine regression (BKMR).

The default Gaussian/probit sampler follows bkmr 0.2.2 prior conventions.
Generalized and structured samplers use separate amplitude defaults.
Bounds for ``invunif`` apply to 1/r.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Prior:
    """Prior settings for exposure relevance, model variances and outcome models.

    A prior describes a parameter before the outcome data are used. Relevance r_m
    controls how exposure m changes the kernel. Paired ``*_mean``/``*_sd``
    fields give gamma means and standard deviations. ``beta_sd`` instead gives
    the normal coefficient-prior standard deviation. ``*_shape``/``*_rate``
    fields give gamma shape and rate; residual variance uses an inverse-gamma.

    ``invunif`` means 1/r_m ~ Uniform(r_lower, r_upper); ``unif`` means
    r_m ~ Uniform(r_lower, r_upper). Excluded exposures have r_m=0.
    For the default Gaussian/probit sampler, ``lambda`` is tau/sigma2;
    for count/survival and structured samplers, it is the GP variance tau directly.
    Here tau is the variance multiplying the kernel and sigma2 is residual variance.
    """
    r_distribution: str = 'invunif'
    r_mean: float = 5.0
    r_sd: float = 5.0
    r_lower: float = 0.0
    r_upper: float = 100.0
    lambda_mean: float = 10.0
    lambda_sd: float = 10.0
    sigma2_shape: float = 0.001
    sigma2_rate: float = 0.001
    inclusion_alpha: float = 1.0
    inclusion_beta: float = 1.0
    cluster_lambda_mean: float = 10.0
    cluster_lambda_sd: float = 10.0
    beta_sd: float = 5.0
    dispersion_shape: float = 2.0
    dispersion_rate: float = 0.1
    baseline_shape: float = 0.1
    baseline_rate: float = 0.1

    def __post_init__(self):
        if self.r_distribution not in ('gamma', 'unif', 'invunif'):
            raise ValueError("r_distribution must be 'gamma', 'unif', or 'invunif'")
        for name in ('r_mean', 'r_sd', 'lambda_mean', 'lambda_sd',
                     'sigma2_shape', 'sigma2_rate', 'inclusion_alpha', 'inclusion_beta',
                     'cluster_lambda_mean', 'cluster_lambda_sd', 'beta_sd',
                     'dispersion_shape', 'dispersion_rate', 'baseline_shape', 'baseline_rate'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a finite positive scalar')
        for name in ('r_lower', 'r_upper'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
                raise ValueError(f'{name} must be finite')
        if not 0 <= self.r_lower < self.r_upper:
            raise ValueError('r bounds must satisfy 0 <= r_lower < r_upper')
        # Reject gamma parameters that overflow/underflow before any fit starts.
        for mean, sd in ((self.r_mean, self.r_sd), (self.lambda_mean, self.lambda_sd),
                         (self.cluster_lambda_mean, self.cluster_lambda_sd)):
            shape, rate = (mean / sd) ** 2, mean / sd ** 2
            if not np.isfinite(shape) or not np.isfinite(rate) or min(shape, rate) <= 0:
                raise ValueError('gamma mean and sd imply unusable shape/rate')

    def sample_r(self, rng):
        if self.r_distribution == 'gamma':
            value = rng.gamma((self.r_mean / self.r_sd) ** 2, self.r_sd ** 2 / self.r_mean)
        else:
            value = rng.uniform(self.r_lower, self.r_upper)
            if self.r_distribution == 'invunif':
                value = np.inf if value == 0 else 1.0 / value
        if not np.isfinite(value) or value <= 0:
            raise FloatingPointError('prior draw is outside finite positive range; rescale prior')
        return float(value)

    def log_r(self, value):
        if not np.isfinite(value) or value <= 0:
            return -np.inf
        if self.r_distribution == 'gamma':
            return gamma_logpdf(value, self.r_mean, self.r_sd)
        transformed = 1.0 / value if self.r_distribution == 'invunif' else value
        if not self.r_lower <= transformed <= self.r_upper:
            return -np.inf
        return -math.log(self.r_upper - self.r_lower) - (2 * math.log(value) if self.r_distribution == 'invunif' else 0)


def gamma_logpdf(value, mean, sd):
    if not np.isfinite(value) or value <= 0:
        return -np.inf
    shape, rate = (mean / sd) ** 2, mean / sd ** 2
    return shape * math.log(rate) - math.lgamma(shape) + (shape - 1) * math.log(value) - rate * value


PriorConfig = Prior
