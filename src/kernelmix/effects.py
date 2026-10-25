"""Posterior exposure associations in original exposure units.

These are conditional association summaries, not causal effects. Response-scale
summaries hold raw adjustment covariates at ``X``, or their retained training
means by default. Intervals are equal-tailed pointwise posterior intervals.
Calls are limited to 4096 result locations because joint prediction and
contrast matrices are dense; smaller grids usually give clearer summaries.
"""
from dataclasses import dataclass, field
from itertools import combinations
from numbers import Integral

import numpy as np

__all__ = ['EffectResult', 'CurveResult', 'SurfaceResult', 'overall_effect',
           'univariate_response', 'bivariate_response', 'single_exposure_effects',
           'interaction_effects']

# Contrasts and joint GP prediction use dense matrices. Bound allocations.
_MAX_LOCATIONS = 4096


def _bounded(count):
    if count > _MAX_LOCATIONS:
        raise ValueError(f'at most {_MAX_LOCATIONS} effect locations are supported per call; use smaller grids or fewer exposures')



@dataclass
class EffectResult:
    """Joint draws (chain, draw, location) and pointwise posterior summaries.

    ``locations`` contains one dictionary per location, in the same order as
    the last draw dimension. ``profiles`` records the original-unit exposure
    rows used in prediction. ``coefficients`` maps their joint draws to each
    reported association. SD uses the sample denominator when possible.
    """
    draws: np.ndarray
    mean: np.ndarray
    sd: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    locations: list
    profiles: np.ndarray
    coefficients: np.ndarray
    scale: str
    probability: float
    kind: str
    metadata: dict = field(default_factory=dict)

    def to_records(self):
        """Return plain dictionaries; pandas is not needed."""
        return [dict(location, mean=float(self.mean[i]), sd=float(self.sd[i]),
                     lower=float(self.lower[i]), upper=float(self.upper[i]),
                     scale=self.scale, probability=self.probability)
                for i, location in enumerate(self.locations)]


@dataclass
class CurveResult(EffectResult):
    """One or more curves, with variable and exposure value in locations."""


@dataclass
class SurfaceResult(EffectResult):
    """A flattened surface, ordered with the second grid changing fastest."""
    variables: tuple = ()
    grids: tuple = ()

    @property
    def shape(self):
        return tuple(len(grid) for grid in self.grids)


def _training(fit):
    z = np.asarray(fit.data['Z_original'], dtype=float)
    if z.ndim != 2 or not z.size or not np.isfinite(z).all():
        raise ValueError('fit must contain finite retained original exposures')
    return z


def _quantiles(value, name, *, pair=False):
    q = np.asarray(value, dtype=float)
    if q.ndim != 1 or not q.size or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError(f'{name} must be a nonempty sequence in [0, 1]')
    if pair and (q.size != 2 or q[0] > q[1]):
        raise ValueError(f'{name} must contain two ordered quantiles (low, high)')
    return q


def _q(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim != 0 or not np.isfinite(a) or not 0 <= float(a) <= 1:
        raise ValueError(f'{name} must be a finite number in [0, 1]')
    return float(a)


def _variable(fit, variable):
    names = list(fit.exposure_names)
    if isinstance(variable, str):
        if variable not in names:
            raise ValueError(f'unknown exposure {variable!r}')
        return names.index(variable)
    if isinstance(variable, (bool, np.bool_)) or not isinstance(variable, Integral):
        raise ValueError('variables must be exposure names or integer column indices')
    if not 0 <= variable < len(names):
        raise ValueError('exposure index is out of range')
    return int(variable)


def _grid(z, variable, grid, points):
    if grid is None:
        if isinstance(points, (bool, np.bool_)) or not isinstance(points, Integral) or points < 2:
            raise ValueError('points must be an integer of at least two')
        _bounded(points)
        return np.linspace(z[:, variable].min(), z[:, variable].max(), points)
    g = np.asarray(grid, dtype=float)
    if g.ndim != 1 or not g.size or not np.isfinite(g).all():
        raise ValueError('grid must be a nonempty finite one-dimensional sequence')
    _bounded(g.size)
    if np.any(np.diff(g) <= 0):
        raise ValueError('grid must be strictly increasing')
    return g.copy()


def _evaluate(fit, profiles, coefficients, locations, *, scale, X, probability,
              max_draws, seed, kind, cls=EffectResult, metadata=None, **extra):
    if scale not in ('latent', 'response'):
        raise ValueError("scale must be 'latent' or 'response'")
    p = np.asarray(probability, dtype=float)
    if p.ndim != 0 or not np.isfinite(p) or not 0 < float(p) < 1:
        raise ValueError('probability must lie strictly between zero and one')
    profiles = np.asarray(profiles, dtype=float)
    coefficients = np.asarray(coefficients, dtype=float)
    # Predict once, jointly. Repeated profiles share the exact same random draw.
    unique, inverse = np.unique(profiles, axis=0, return_inverse=True)
    weights = np.zeros((len(locations), len(unique)))
    for column, target in enumerate(inverse):
        weights[:, target] += coefficients[:, column]
    options = dict(probability=float(p), max_draws=max_draws, seed=seed)
    covariates = None
    if scale == 'latent':
        if X is not None:
            raise ValueError('X only applies to response-scale summaries')
        prediction = fit.predict_h(unique, **options)
    else:
        raw = np.asarray(fit.data['X_original'], dtype=float)
        covariates = raw.mean(axis=0) if X is None else np.asarray(X, dtype=float)
        if covariates.ndim != 1 or covariates.size != raw.shape[1] or not np.isfinite(covariates).all():
            raise ValueError('X must be a finite vector of raw adjustment covariates')
        if fit.family in ('poisson', 'negative_binomial'):
            options['offset'] = np.zeros(len(unique))
        prediction = fit.predict(unique, Xnew=np.tile(covariates, (len(unique), 1)),
                                 kind='mean', scale='response', **options)
    d = np.asarray(prediction.draws)
    if d.ndim != 3 or d.shape[-1] != len(unique) or not np.isfinite(d).all():
        raise ValueError('prediction must provide finite joint draws (chain, draw, row)')
    values = np.einsum('cdp,lp->cdl', d, weights)
    flat = values.reshape(-1, len(locations))
    tail = (1 - float(p)) / 2
    meta = {} if metadata is None else dict(metadata)
    meta['covariates'] = None if covariates is None else covariates.copy()
    meta['family'] = fit.family
    if scale == 'response' and fit.family in ('poisson', 'negative_binomial', 'survival'):
        meta['random_intercept'] = 'zero'
        meta['response_quantity'] = 'relative_hazard' if fit.family == 'survival' else 'unit-exposure count mean'
    return cls(draws=values, mean=flat.mean(axis=0),
               sd=flat.std(axis=0, ddof=1 if len(flat) > 1 else 0),
               lower=np.quantile(flat, tail, axis=0),
               upper=np.quantile(flat, 1 - tail, axis=0), locations=locations,
               profiles=unique.copy(), coefficients=weights, scale=scale,
               probability=float(p), kind=kind, metadata=meta, **extra)


def overall_effect(fit, quantiles=(.25, .5, .75), reference=.5, scale='latent',
                   X=None, probability=.95, max_draws=None, seed=None):
    """Compare all-exposure quantile profiles with one reference profile."""
    z = _training(fit)
    qs = _quantiles(quantiles, 'quantiles')
    _bounded(len(qs))
    ref = _q(reference, 'reference')
    profiles = np.vstack([np.quantile(z, qs, axis=0), np.quantile(z, ref, axis=0)])
    weights = np.column_stack([np.eye(len(qs)), -np.ones(len(qs))])
    locations = [dict(quantile=float(q), reference=ref) for q in qs]
    return _evaluate(fit, profiles, weights, locations, scale=scale, X=X,
                     probability=probability, max_draws=max_draws, seed=seed,
                     kind='overall', metadata={'reference': ref})


def univariate_response(fit, variable=None, grid=None, points=50,
                        fixed_quantile=.5, center=True, scale='latent', X=None,
                        probability=.95, max_draws=None, seed=None):
    """Vary exposures on original-unit grids; others stay at fixed_quantile.

    ``variable=None`` returns all curves jointly, using a common grid when one
    is supplied. Centering subtracts the common all-median profile draw by draw,
    including when fixed_quantile differs from .5. This differs from R's
    subtraction of the average fitted grid value.
    """
    z = _training(fit)
    q = _q(fixed_quantile, 'fixed_quantile')
    if not isinstance(center, (bool, np.bool_)):
        raise ValueError('center must be boolean')
    variables = range(z.shape[1]) if variable is None else [_variable(fit, variable)]
    profiles, locations = [], []
    background = np.quantile(z, q, axis=0)
    for j in variables:
        current_grid = _grid(z, j, grid, points)
        _bounded(len(locations) + len(current_grid))
        for value in current_grid:
            profile = background.copy()
            profile[j] = value
            profiles.append(profile)
            locations.append(dict(variable=fit.exposure_names[j], variable_index=j, value=float(value)))
    n = len(profiles)
    weights = np.eye(n)
    if center:
        profiles.append(np.quantile(z, .5, axis=0))
        weights = np.column_stack([weights, -np.ones(n)])
    return _evaluate(fit, profiles, weights, locations, scale=scale, X=X,
                     probability=probability, max_draws=max_draws, seed=seed,
                     kind='univariate', cls=CurveResult,
                     metadata={'center': bool(center), 'fixed_quantile': q, 'reference_quantile': .5 if center else None})


def bivariate_response(fit, variables=(0, 1), grids=None, points=20,
                       fixed_quantile=.5, center=True, scale='latent', X=None,
                       probability=.95, max_draws=None, seed=None):
    """Vary a pair on a Cartesian grid; center at the all-median profile."""
    z = _training(fit)
    if len(variables) != 2:
        raise ValueError('variables must contain exactly two distinct exposures')
    j, k = [_variable(fit, v) for v in variables]
    if j == k:
        raise ValueError('variables must be distinct')
    if not isinstance(center, (bool, np.bool_)):
        raise ValueError('center must be boolean')
    q = _q(fixed_quantile, 'fixed_quantile')
    if grids is None:
        grids = (None, None)
    if len(grids) != 2:
        raise ValueError('grids must contain two grid sequences')
    g1, g2 = (_grid(z, j, grids[0], points), _grid(z, k, grids[1], points))
    _bounded(len(g1) * len(g2))
    profiles, locations = [], []
    background = np.quantile(z, q, axis=0)
    for a in g1:
        for b in g2:
            profile = background.copy()
            profile[j], profile[k] = a, b
            profiles.append(profile)
            locations.append(dict(variable1=fit.exposure_names[j], variable2=fit.exposure_names[k], value1=float(a), value2=float(b)))
    n = len(profiles)
    weights = np.eye(n)
    if center:
        profiles.append(np.quantile(z, .5, axis=0))
        weights = np.column_stack([weights, -np.ones(n)])
    return _evaluate(fit, profiles, weights, locations, scale=scale, X=X,
                     probability=probability, max_draws=max_draws, seed=seed,
                     kind='bivariate', cls=SurfaceResult,
                     metadata={'center': bool(center), 'fixed_quantile': q, 'reference_quantile': .5 if center else None},
                     variables=(fit.exposure_names[j], fit.exposure_names[k]), grids=(g1, g2))


def single_exposure_effects(fit, quantiles=(.25, .75),
                            fixed_quantiles=(.25, .5, .75), scale='latent', X=None,
                            probability=.95, max_draws=None, seed=None):
    """High-minus-low exposure contrasts at each background quantile."""
    z = _training(fit)
    low, high = _quantiles(quantiles, 'quantiles', pair=True)
    fixed = _quantiles(fixed_quantiles, 'fixed_quantiles')
    _bounded(len(fixed) * z.shape[1])
    profiles, locations = [], []
    limits = np.quantile(z, [low, high], axis=0)
    for q in fixed:
        for j, name in enumerate(fit.exposure_names):
            pair = np.tile(np.quantile(z, q, axis=0), (2, 1))
            pair[:, j] = limits[:, j]
            profiles.extend(pair)
            locations.append(dict(variable=name, variable_index=j, fixed_quantile=float(q), low_quantile=float(low), high_quantile=float(high), low=float(limits[0, j]), high=float(limits[1, j])))
    weights = np.kron(np.eye(len(locations)), [[-1., 1.]])
    return _evaluate(fit, profiles, weights, locations, scale=scale, X=X,
                     probability=probability, max_draws=max_draws, seed=seed, kind='single_exposure')


def interaction_effects(fit, variables=None, quantiles=(.25, .75),
                        fixed_quantile=.5, scale='latent', X=None,
                        probability=.95, max_draws=None, seed=None):
    """Pairwise differences of differences: (HH-LH) - (HL-LL).

    ``variables`` selects exposures by index or name. All unordered pairs in
    that selection are returned. Other exposures stay at fixed_quantile.
    This pairwise quantity differs from R's all-background interaction summary.
    """
    z = _training(fit)
    low, high = _quantiles(quantiles, 'quantiles', pair=True)
    q = _q(fixed_quantile, 'fixed_quantile')
    selected = list(range(z.shape[1])) if variables is None else [_variable(fit, v) for v in variables]
    if len(selected) < 2 or len(set(selected)) != len(selected):
        raise ValueError('variables must select at least two distinct exposures')
    _bounded(len(selected) * (len(selected) - 1) // 2)
    limits = np.quantile(z, [low, high], axis=0)
    profiles, locations = [], []
    for j, k in combinations(selected, 2):
        for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
            profile = np.quantile(z, q, axis=0)
            profile[j], profile[k] = limits[a, j], limits[b, k]
            profiles.append(profile)
        locations.append(dict(variable1=fit.exposure_names[j], variable2=fit.exposure_names[k], fixed_quantile=q, low_quantile=float(low), high_quantile=float(high), low1=float(limits[0, j]), high1=float(limits[1, j]), low2=float(limits[0, k]), high2=float(limits[1, k])))
    weights = np.kron(np.eye(len(locations)), [[1., -1., -1., 1.]])
    return _evaluate(fit, profiles, weights, locations, scale=scale, X=X,
                     probability=probability, max_draws=max_draws, seed=seed, kind='interaction')
