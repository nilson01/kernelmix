"""Gaussian block correlations, log-variance scaling and residual prediction.

Correlation operates only within observation clusters. The covariance is
sigma2 * D R D, where D has diagonal exp(log_variance/2). No diagonal jitter
is added. Conditional prediction returns a residual mean and covariance;
the caller controls random draws.
"""

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

_LIMIT = 512 * 1024**2


def _budget(elements):
    if 8 * int(elements) > _LIMIT:
        raise ValueError('Gaussian residual working arrays exceed the 512 MiB budget')


def _vector(value, name, n=None):
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or (n is not None and len(result) != n) or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite vector with one entry per observation')
    return result


def _codes(value, name, n=None):
    codes = np.asarray(value)
    if (codes.ndim != 1 or (n is not None and len(codes) != n)
            or not np.issubdtype(codes.dtype, np.integer) or np.any(codes < 0)):
        raise ValueError(f'{name} must be a nonnegative integer cluster-code vector')
    return codes


def _kind(kind, matrix):
    if isinstance(kind, np.ndarray):
        if matrix is not None:
            raise ValueError('supply a correlation matrix only once')
        matrix, kind = kind, 'fixed'
    if matrix is not None:
        if kind not in (None, 'fixed', 'matrix'):
            raise ValueError('matrix cannot be combined with a parametric correlation')
        kind = 'fixed'
    if kind in (None, 'independent'):
        return None, matrix
    if kind not in ('ar1', 'exponential', 'fixed', 'matrix'):
        raise ValueError('correlation kind must be ar1, exponential, fixed or None')
    return 'fixed' if kind == 'matrix' else kind, matrix


def _parameter(kind, parameter):
    if kind not in ('ar1', 'exponential'):
        if parameter is not None:
            raise ValueError('correlation parameter applies only to ar1 or exponential')
        return None
    if (isinstance(parameter, (bool, np.bool_)) or not np.isscalar(parameter)
            or not np.isfinite(parameter)):
        raise ValueError('correlation parameter must be finite')
    if kind == 'ar1' and not -1 < parameter < 1:
        raise ValueError('AR1 correlation must lie strictly between -1 and one')
    if kind == 'exponential' and parameter <= 0:
        raise ValueError('exponential correlation range must be positive')
    return float(parameter)


def _times(times, n, kind, name):
    times = _vector(times, name, n)
    if kind == 'ar1' and np.any(times != np.floor(times)):
        raise ValueError('AR1 observation times must be integer valued')
    return times


def _parametric(a, b, kind, parameter):
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        distance = np.abs(a[:, None] - b[None, :])
        if kind == 'ar1':
            result = np.power(parameter, distance)
        else:
            result = np.exp(-distance / parameter)
    if not np.isfinite(result).all():
        raise FloatingPointError('residual correlation calculation overflowed')
    return result


def _correlation_matrix(value, n, codes, name, *, positive_definite=True):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (n, n) or not np.isfinite(matrix).all():
        raise ValueError(f'{name} must be a finite square matrix matching its observations')
    _budget(4*n*n)
    if not np.allclose(matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError(f'{name} must be symmetric')
    if not np.allclose(np.diag(matrix), 1., rtol=0, atol=1e-12):
        raise ValueError(f'{name} must have unit diagonal')
    if np.any(np.abs(matrix) > 1+1e-12):
        raise ValueError(f'{name} entries must lie between minus one and one')
    # Check block separation without allocating a second n-by-n identity mask.
    for group in np.unique(codes):
        rows = np.flatnonzero(codes == group)
        other = np.flatnonzero(codes != group)
        if len(other) and np.any(np.abs(matrix[np.ix_(rows, other)]) > 1e-12):
            raise ValueError(f'{name} must be zero between clusters')
        block = (matrix[np.ix_(rows, rows)] + matrix[np.ix_(rows, rows)].T) / 2
        if positive_definite:
            try:
                np.linalg.cholesky(block)
            except np.linalg.LinAlgError as exc:
                raise ValueError(f'{name} must be positive definite within each cluster') from exc
        else:
            _psd(block, name)
    return matrix


def _psd(covariance, name):
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(covariance).all():
        raise FloatingPointError(f'{name} is nonfinite')
    if not len(covariance):
        return covariance
    values, vectors = np.linalg.eigh(covariance)
    scale = max(1., float(np.max(np.abs(values), initial=0)))
    tolerance = 100 * len(covariance) * np.finfo(float).eps * scale
    if values.min(initial=0) < -tolerance:
        raise ValueError(f'{name} is materially indefinite')
    # Remove negative roundoff only; retain legitimate small positive variance.
    return (vectors * np.maximum(values, 0.)) @ vectors.T


def correlation_blocks(codes, times=None, kind=None, parameter=None, matrix=None):
    """Return (indices, R, lower-Cholesky) for each correlated cluster.

    Independent errors use an empty fast path. AR1 requires distinct integer
    times within each training cluster; exponential correlation requires
    distinct finite times. A fixed matrix must be positive definite and zero
    between clusters. Inputs refer to already retained rows.
    """
    kind, matrix = _kind(kind, matrix)
    parameter = _parameter(kind, parameter)
    if kind is None:
        return []
    if codes is None:
        raise ValueError('residual correlation requires observation clusters')
    codes = _codes(codes, 'codes')
    if not len(codes):
        raise ValueError('correlation requires at least one observation')
    sizes = [np.count_nonzero(codes == group) for group in np.unique(codes)]
    _budget(5*sum(int(size)**2 for size in sizes))
    if kind == 'fixed':
        if matrix is None:
            raise ValueError('fixed correlation requires a matrix')
        matrix = _correlation_matrix(matrix, len(codes), codes, 'matrix')
    else:
        times = _times(times, len(codes), kind, 'times')
    result = []
    for group in np.unique(codes):
        indices = np.flatnonzero(codes == group)
        if kind == 'fixed':
            block = matrix[np.ix_(indices, indices)].copy()
        else:
            local = times[indices]
            if len(np.unique(local)) != len(local):
                raise ValueError('duplicate within-cluster times give singular residual correlation')
            block = _parametric(local, local, kind, parameter)
        try:
            lower = np.linalg.cholesky((block + block.T) / 2)
        except np.linalg.LinAlgError as exc:
            raise ValueError('within-cluster residual correlation is not positive definite') from exc
        result.append((indices, block, lower))
    return result


def _standardized_residual(y, eta, sigma2, log_variance):
    with np.errstate(over='ignore', invalid='ignore', divide='ignore', under='ignore'):
        residual = y - eta
        exponent = np.full_like(residual, -np.inf)
        np.log(np.abs(residual), out=exponent, where=residual != 0)
        standardized = np.sign(residual) * np.exp(exponent - .5*(np.log(sigma2)+log_variance))
    return standardized


def gaussian_loglik(y, eta, sigma2, log_variance, blocks):
    """Normalized Gaussian log density with block correlation and log variance.

    Invalid candidate scales/predictors return minus infinity. Covariances are
    factored as D R D, so large variance ratios do not require a scaled dense
    covariance factorization. Zero residuals remain zero even at tiny scales.
    """
    y = _vector(y, 'y')
    eta = np.asarray(eta, dtype=float)
    lv = np.asarray(log_variance, dtype=float)
    if eta.shape != y.shape or lv.shape != y.shape:
        raise ValueError('eta and log_variance must have one entry per observation')
    if (isinstance(sigma2, (bool, np.bool_)) or not np.isscalar(sigma2)
            or not np.isfinite(sigma2) or sigma2 <= 0
            or not np.isfinite(eta).all() or not np.isfinite(lv).all()):
        return -np.inf
    standardized = _standardized_residual(y, eta, sigma2, lv)
    if not np.isfinite(standardized).all():
        return -np.inf
    assigned = np.zeros(len(y), dtype=bool)
    quadratic, logdet = 0., 0.
    with np.errstate(over='ignore', invalid='ignore'):
        for indices, correlation, lower in blocks:
            indices = np.asarray(indices)
            if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
                    or np.any(indices < 0) or np.any(indices >= len(y))
                    or len(np.unique(indices)) != len(indices) or assigned[indices].any()):
                raise ValueError('correlation blocks must contain disjoint valid indices')
            if np.shape(correlation) != (len(indices), len(indices)) or np.shape(lower) != np.shape(correlation):
                raise ValueError('correlation block factors have incompatible shapes')
            assigned[indices] = True
            local = solve_triangular(lower, standardized[indices], lower=True, check_finite=False)
            quadratic += float(local @ local)
            logdet += 2*float(np.log(np.diag(lower)).sum())
        quadratic += float(standardized[~assigned] @ standardized[~assigned])
        total = -.5*(len(y)*np.log(2*np.pi) + len(y)*np.log(sigma2) + lv.sum() + logdet + quadratic)
    return float(total) if np.isfinite(total) else -np.inf


def residual_prediction(y, eta_train, sigma2, logvar_train, logvar_new,
                        train_codes, new_codes, train_time, new_time, kind, parameter, rng,
                        *, matrix=None, cross=None, new_matrix=None, observation=True):
    """Return joint conditional residual (mean, covariance), without drawing.

    ``cross`` is a new-by-training correlation matrix; ``new_matrix`` is a
    new-by-new correlation matrix. For fixed training correlations, omitted
    cross correlations mean independent new residuals. Equal known cluster/time
    points share the conditioned residual process. ``observation=False`` keeps
    the conditional mean and returns zero covariance. ``rng`` is accepted for
    interface consistency and is not consumed.
    """
    y = _vector(y, 'y')
    eta = _vector(eta_train, 'eta_train', len(y))
    lv_train = _vector(logvar_train, 'logvar_train', len(y))
    lv_new = _vector(logvar_new, 'logvar_new')
    if not isinstance(observation, (bool, np.bool_)):
        raise ValueError('observation must be boolean')
    if isinstance(sigma2, (bool, np.bool_)) or not np.isscalar(sigma2) or not np.isfinite(sigma2) or sigma2 <= 0:
        raise ValueError('sigma2 must be finite and positive')
    n, count = len(y), len(lv_new)
    _budget(3*count*count + 5*(n+count))
    kind, matrix = _kind(kind, matrix)
    parameter = _parameter(kind, parameter)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        scale_new = np.exp(.5*(np.log(sigma2)+lv_new))
    if not np.isfinite(scale_new).all() or np.any(scale_new <= 0):
        raise FloatingPointError('new residual scales exceed floating-point range')
    standardized = _standardized_residual(y, eta, sigma2, lv_train)
    if not np.isfinite(standardized).all():
        raise FloatingPointError('training standardized residuals exceed floating-point range')
    mean = np.zeros(count)
    covariance = np.zeros((count, count))
    if kind is None:
        if cross is not None or new_matrix is not None:
            raise ValueError('custom residual correlation inputs require fixed correlation')
        if observation:
            np.fill_diagonal(covariance, scale_new**2)
            if not np.isfinite(covariance).all():
                raise FloatingPointError('new residual covariance overflowed')
        return mean, covariance
    if train_codes is None or new_codes is None:
        raise ValueError('correlated prediction requires training and new cluster codes')
    train_codes = _codes(train_codes, 'train_codes', n)
    new_codes = _codes(new_codes, 'new_codes', count)
    training = correlation_blocks(train_codes, train_time, kind, parameter, matrix=matrix)
    factors = {int(train_codes[indices[0]]): (indices, block, lower) for indices, block, lower in training}
    stored_elements = 2*sum(len(indices)**2 for indices, _, _ in training)
    if kind != 'fixed':
        if cross is not None or new_matrix is not None:
            raise ValueError('custom correlation matrices require fixed training correlation')
        train_time = _times(train_time, n, kind, 'train_time')
        new_time = _times(new_time, count, kind, 'new_time')
    else:
        if cross is None:
            cross = None
        else:
            cross = np.asarray(cross, dtype=float)
            if cross.shape != (count, n) or not np.isfinite(cross).all():
                raise ValueError('cross must be a finite new-by-training correlation matrix')
            _budget(stored_elements + 3*count*count + 2*count*n)
            for code in np.unique(new_codes):
                rows = np.flatnonzero(new_codes == code)
                other = np.flatnonzero(train_codes != code)
                if len(other) and np.any(np.abs(cross[np.ix_(rows, other)]) > 1e-12):
                    raise ValueError('cross must be zero between different clusters')
        if new_matrix is not None:
            new_matrix = _correlation_matrix(new_matrix, count, new_codes, 'new_matrix', positive_definite=False)
    for code in np.unique(new_codes):
        rows = np.flatnonzero(new_codes == code)
        size = len(rows)
        if int(code) in factors:
            columns, _, lower = factors[int(code)]
        else:
            columns, lower = np.empty(0, dtype=int), None
        _budget(stored_elements + 3*count*count + 10*(len(columns)+size)**2)
        if kind == 'fixed':
            prior = np.eye(size) if new_matrix is None else new_matrix[np.ix_(rows, rows)]
            shared = np.zeros((size, len(columns))) if cross is None else cross[np.ix_(rows, columns)]
        else:
            prior = _parametric(new_time[rows], new_time[rows], kind, parameter)
            shared = _parametric(new_time[rows], train_time[columns], kind, parameter)
        conditional = prior.copy()
        if len(columns):
            whitened = solve_triangular(lower, shared.T, lower=True, check_finite=False)
            conditional -= whitened.T @ whitened
            local_mean = shared @ cho_solve((lower, True), standardized[columns], check_finite=False)
            mean[rows] = scale_new[rows] * local_mean
        conditional = _psd(conditional, 'joint residual correlation Schur complement')
        if len(columns):
            for local_row, row in enumerate(rows):
                # Correlation exactly one identifies the same standardized residual.
                # Preserve that identity instead of sampling Schur roundoff.
                matches = (np.flatnonzero(shared[local_row] == 1.) if kind == 'fixed'
                           else np.flatnonzero(new_time[row] == train_time[columns]))
                if len(matches):
                    mean[row] = scale_new[row] * standardized[columns[matches[0]]]
                    conditional[local_row, :] = 0.
                    conditional[:, local_row] = 0.
        if observation:
            with np.errstate(over='ignore', invalid='ignore'):
                local_covariance = scale_new[rows, None] * conditional * scale_new[None, rows]
            if not np.isfinite(local_covariance).all():
                raise FloatingPointError('conditional residual covariance overflowed')
            covariance[np.ix_(rows, rows)] = local_covariance
    if not np.isfinite(mean).all():
        raise FloatingPointError('conditional residual mean overflowed')
    return mean, covariance
