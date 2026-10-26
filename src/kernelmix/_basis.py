"""Dense and finite-feature Gaussian process (GP) representations.

Predictive-process (PP) and random Fourier feature (RFF) paths use fixed basis
randomness and form only row-by-feature and feature-by-feature arrays.
PP omits the parent GP's residual variance;
RFF and PP are model approximations, not exact dense-GP computations.
"""

from numbers import Integral

import numpy as np

_LIMIT = 512 * 1024**2


def _budget(elements):
    if 8 * int(elements) > _LIMIT:
        raise ValueError('basis working arrays exceed the 512 MiB memory budget')


def _matrix(value, name, columns=None):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or not a.shape[0] or not a.shape[1] or not np.isfinite(a).all():
        raise ValueError(f'{name} must be a nonempty finite numeric matrix')
    if columns is not None and a.shape[1] != columns:
        raise ValueError(f'{name} must have {columns} columns')
    return a


def _settings(kernel, alpha):
    if kernel not in ('gaussian', 'matern32', 'rational_quadratic'):
        raise ValueError('kernel must be gaussian, matern32 or rational_quadratic')
    if (isinstance(alpha, (bool, np.bool_)) or not np.isscalar(alpha)
            or not np.isfinite(alpha) or alpha <= 0):
        raise ValueError('alpha must be finite and positive')
    return float(alpha)


def _relevance(r, columns):
    r = np.ones(columns) if r is None else np.asarray(r, dtype=float)
    if r.shape != (columns,) or not np.isfinite(r).all() or np.any(r < 0):
        raise ValueError('r must be a finite nonnegative vector, one value per exposure')
    return r


def kernel_matrix(Z1, Z2=None, r=None, *, kernel='gaussian', alpha=1):
    """Return Gaussian, Matérn 3/2 or rational-quadratic kernel values.

    A kernel assigns similarity to pairs of exposure profiles. Here each profile
    has similarity one with itself. Let q be sum_m r_m * (z_m - z'_m)**2.
    The three kernels are exp(-q), (1+sqrt(6q))*exp(-sqrt(6q)), and
    (1+q/alpha)**(-alpha). Relevance r_m is nonnegative; alpha is positive.
    Inputs are used directly, without training standardization.
    """
    a = _matrix(Z1, 'Z1')
    b = a if Z2 is None else _matrix(Z2, 'Z2', a.shape[1])
    alpha = _settings(kernel, alpha)
    weights = _relevance(r, a.shape[1])
    _budget(4 * len(a) * len(b))
    distance = np.zeros((len(a), len(b)))
    with np.errstate(over='ignore', invalid='ignore', under='ignore', divide='ignore'):
        for j in np.flatnonzero(weights):
            distance += weights[j] * np.square(a[:, j, None] - b[None, :, j])
        if kernel == 'gaussian':
            result = np.exp(-distance)
        elif kernel == 'matern32':
            root = np.sqrt(distance) * np.sqrt(6.)
            # exp(log1p(root)-root) avoids inf*0 at very large distances.
            result = np.exp(np.log1p(root) - root)
            result[np.isinf(root)] = 0.
        else:
            # Log-add-exp also handles q/alpha overflowing at tiny alpha.
            log_q = np.full_like(distance, -np.inf)
            np.log(distance, out=log_q, where=distance > 0)
            exponent = alpha * np.logaddexp(0., log_q - np.log(alpha))
            result = np.exp(-exponent)
    if not np.isfinite(result).all():
        raise FloatingPointError('kernel calculation overflowed; rescale exposures or relevance')
    return result


def _eigen(kernel):
    n = len(kernel)
    _budget(10 * n * n)
    values, vectors = np.linalg.eigh((kernel + kernel.T) / 2)
    largest = max(0., float(values.max(initial=0)))
    tolerance = 100 * n * np.finfo(float).eps * largest
    if values.min(initial=0) < -tolerance:
        raise np.linalg.LinAlgError('kernel is materially indefinite')
    values = np.where(values > tolerance, values, 0.)
    return values, vectors


def _inverse_root(kernel):
    values, vectors = _eigen(kernel)
    inverse = np.zeros_like(values)
    np.divide(1., np.sqrt(values), out=inverse, where=values > 0)
    return (vectors * inverse) @ vectors.T


def _maxmin(Z, count):
    """Deterministic max-min knots, using length-n distances rather than n²."""
    chosen = []
    nearest = np.full(len(Z), np.inf)
    current = 0
    for _ in range(min(count, len(Z))):
        chosen.append(current)
        distance = np.zeros(len(Z))
        with np.errstate(over='ignore', invalid='ignore'):
            for column in range(Z.shape[1]):
                distance += np.square(Z[:, column] - Z[current, column])
        if not np.isfinite(distance).all():
            raise FloatingPointError('knot distance overflowed; rescale exposures')
        nearest = np.minimum(nearest, distance)
        nearest[chosen] = 0.
        current = int(np.argmax(nearest))
        if nearest[current] == 0:
            break
    return Z[chosen].copy(), np.asarray(chosen, dtype=int)


def prepare_basis(Z, *, approximation='dense', kernel='gaussian', alpha=1,
                  n_components=32, knots=None, rng=None):
    """Prepare serializable arrays/scalars for a fixed GP representation.

    RFF components count frequency pairs, hence there are twice as many
    features. Its base frequencies are frozen and do not depend on relevance.
    PP automatically chooses at most n distinct max-min knots when omitted.
    Explicit knots are already in the caller's transformed exposure units.
    """
    z = _matrix(Z, 'Z')
    alpha = _settings(kernel, alpha)
    if approximation not in ('dense', 'pp', 'rff'):
        raise ValueError('approximation must be dense, pp or rff')
    if isinstance(n_components, (bool, np.bool_)) or not isinstance(n_components, Integral) or n_components < 1:
        raise ValueError('n_components must be a positive integer')
    n_components = int(n_components)
    if knots is not None and approximation != 'pp':
        raise ValueError('knots apply only to the predictive-process approximation')
    result = dict(approximation=approximation, kernel=kernel, alpha=alpha,
                  n_exposures=z.shape[1], n_training=len(z),
                  numerical_rank_rule='eigenvalue > 100*n*eps*largest',
                  version=1)
    if approximation == 'dense':
        _budget(10 * len(z)**2 + z.size)
        result.update(train_Z=z.copy(), n_features=len(z))
    elif approximation == 'pp':
        if knots is None:
            count = min(n_components, len(z))
            _budget(5*len(z)*count + 10*count*count + z.size)
            chosen, indices = _maxmin(z, count)
        else:
            chosen = _matrix(knots, 'knots', z.shape[1]).copy()
            count = len(chosen)
            _budget(5*len(z)*count + 10*count*count + chosen.size)
            indices = np.empty(0, dtype=int)
        result.update(knots=chosen, knot_indices=indices, n_features=len(chosen),
                      residual_correction=False, knot_selection='maxmin' if knots is None else 'explicit')
    else:
        _budget(5*len(z)*n_components + 3*n_components*z.shape[1])
        if rng is None:
            rng = np.random.default_rng()
        if not isinstance(rng, np.random.Generator):
            raise ValueError('rng must be a NumPy Generator or None')
        base = rng.normal(size=(n_components, z.shape[1]))
        if kernel == 'matern32':
            scales = np.sqrt(3. / rng.chisquare(3., size=n_components))
        elif kernel == 'rational_quadratic':
            scales = np.sqrt(rng.gamma(alpha, 1/alpha, size=n_components))
        else:
            scales = np.ones(n_components)
        if not np.isfinite(base).all() or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise FloatingPointError('spectral frequency draw left its finite positive range')
        result.update(base_frequencies=base, frequency_scales=scales,
                      n_components=n_components, n_features=2*n_components,
                      bit_generator=type(rng.bit_generator).__name__)
    return result


def basis_matrix(Z, r, basis):
    """Train/new features; PP and RFF do not construct a training n-by-n matrix."""
    z = _matrix(Z, 'Z', int(basis['n_exposures']))
    weights = _relevance(r, z.shape[1])
    kernel, alpha = basis['kernel'], basis['alpha']
    _settings(kernel, alpha)
    mode = basis['approximation']
    if mode == 'rff':
        base = _matrix(basis['base_frequencies'], 'base_frequencies', z.shape[1])
        scales = np.asarray(basis['frequency_scales'], dtype=float)
        if scales.shape != (len(base),) or not np.isfinite(scales).all() or np.any(scales <= 0):
            raise ValueError('invalid frozen frequency scales')
        _budget(5*len(z)*len(base) + 2*base.size)
        with np.errstate(over='ignore', invalid='ignore'):
            frequency = base * scales[:, None] * (np.sqrt(2.)*np.sqrt(weights))[None, :]
            phase = z @ frequency.T
            result = np.concatenate((np.cos(phase), np.sin(phase)), axis=1) / np.sqrt(len(base))
        if not np.isfinite(result).all():
            raise FloatingPointError('RFF features overflowed; rescale exposures or relevance')
        return result
    if mode == 'pp':
        knots = _matrix(basis['knots'], 'knots', z.shape[1])
        _budget(5*len(z)*len(knots) + 10*len(knots)**2)
        root_inverse = _inverse_root(kernel_matrix(knots, r=weights, kernel=kernel, alpha=alpha))
        return kernel_matrix(z, knots, weights, kernel=kernel, alpha=alpha) @ root_inverse
    if mode != 'dense':
        raise ValueError('unknown approximation in saved basis')
    train = _matrix(basis['train_Z'], 'train_Z', z.shape[1])
    _budget(5*len(z)*len(train) + 10*len(train)**2)
    values, vectors = _eigen(kernel_matrix(train, r=weights, kernel=kernel, alpha=alpha))
    inverse = np.zeros_like(values)
    np.divide(1., np.sqrt(values), out=inverse, where=values > 0)
    if z.shape == train.shape and np.array_equal(z, train):
        return vectors * np.sqrt(values)
    return kernel_matrix(z, train, weights, kernel=kernel, alpha=alpha) @ (vectors * inverse)


def _supported_values(values, eigenvalues, vectors, name):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(eigenvalues),) or not np.isfinite(values).all():
        raise ValueError(f'{name} must contain one finite value per anchor')
    retained = eigenvalues > 0
    projection = vectors[:, retained] @ (vectors[:, retained].T @ values)
    tolerance = 1000 * len(values) * np.finfo(float).eps * max(1., float(np.linalg.norm(values)))
    if np.linalg.norm(values-projection) > tolerance:
        raise ValueError(f'{name} is outside the saved kernel numerical support')
    return values


def basis_predict(Znew, r, basis, weights, lambda_, rng, training_h=None):
    """Joint h draws given one state, using saved anchor values when available.

    For dense models ``training_h`` means saved training h; for PP it means
    saved knot h, including the amplitude. These anchors avoid eigensystem
    coordinate changes across serialization/platforms. PP/RFF predictions are
    deterministic given their finite-basis state; no parent-GP residual is added.
    """
    if isinstance(lambda_, (bool, np.bool_)) or not np.isscalar(lambda_) or not np.isfinite(lambda_) or lambda_ <= 0:
        raise ValueError('lambda_ must be finite and positive')
    z = _matrix(Znew, 'Znew', int(basis['n_exposures']))
    r = _relevance(r, z.shape[1])
    w = np.asarray(weights, dtype=float)
    if w.shape != (int(basis['n_features']),) or not np.isfinite(w).all():
        raise ValueError('weights must be finite with one value per basis feature')
    mode, kernel, alpha = basis['approximation'], basis['kernel'], basis['alpha']
    if mode == 'rff':
        with np.errstate(over='ignore', invalid='ignore'):
            result = np.sqrt(lambda_) * (basis_matrix(z, r, basis) @ w)
        if not np.isfinite(result).all():
            raise FloatingPointError('RFF prediction overflowed')
        return result
    if mode not in ('pp', 'dense'):
        raise ValueError('unknown approximation in saved basis')
    anchors = _matrix(basis['knots'] if mode == 'pp' else basis['train_Z'], 'anchors', z.shape[1])
    _budget(10*len(anchors)**2 + 6*len(z)*len(anchors) + (10*len(z)**2 if mode == 'dense' else 0))
    values, vectors = _eigen(kernel_matrix(anchors, r=r, kernel=kernel, alpha=alpha))
    if training_h is None:
        root = ((vectors * np.sqrt(values)) @ vectors.T if mode == 'pp'
                else vectors * np.sqrt(values))
        h = np.sqrt(lambda_) * (root @ w)
    else:
        h = _supported_values(training_h, values, vectors, 'training_h')
    if not np.isfinite(h).all():
        raise FloatingPointError('anchor GP values overflowed')
    inverse = np.zeros_like(values)
    np.divide(1., values, out=inverse, where=values > 0)
    pinv = (vectors * inverse) @ vectors.T
    cross = kernel_matrix(z, anchors, r, kernel=kernel, alpha=alpha)
    mean = cross @ (pinv @ h)
    if not np.isfinite(mean).all():
        raise FloatingPointError('conditional GP mean overflowed')
    if mode == 'pp':
        return mean
    if mode != 'dense':
        raise ValueError('unknown approximation in saved basis')
    if not isinstance(rng, np.random.Generator):
        raise ValueError('dense prediction requires a NumPy Generator')
    covariance = lambda_ * (kernel_matrix(z, r=r, kernel=kernel, alpha=alpha) - cross @ pinv @ cross.T)
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(covariance).all():
        raise FloatingPointError('conditional GP covariance overflowed')
    eig, evec = np.linalg.eigh(covariance)
    # The parent covariance sets the scale even when the conditional variance
    # vanishes. This clips cancellation roundoff, never material negative modes.
    tolerance = 100 * (len(z)+len(anchors)) * np.finfo(float).eps * lambda_
    if eig.min(initial=0) < -tolerance:
        raise np.linalg.LinAlgError('dense conditional GP covariance is materially indefinite')
    eig = np.where(eig > tolerance, eig, 0.)
    result = mean + (evec * np.sqrt(eig)) @ rng.normal(size=len(z))
    # Already conditioned anchors and identical active profiles are the same
    # process value; preserve those identities exactly after numerical solves.
    active = r > 0
    for row in range(len(z)):
        matches = np.flatnonzero(np.all(z[row, active] == anchors[:, active], axis=1))
        if len(matches):
            result[row] = h[matches[0]]
    _, first, inverse_rows = np.unique(z[:, active], axis=0, return_index=True, return_inverse=True)
    if not np.isfinite(result).all():
        raise FloatingPointError('joint GP prediction overflowed')
    return result[first][inverse_rows]
