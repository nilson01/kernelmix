"""Array input checks and stored training transformations (no pandas dependency)."""
import numpy as np

from ._clusters import cluster_mask, encode_clusters


def matrix(value, name, *, columns=None):
    """Convert a numeric 2D input; enforce exact named column order if given."""
    if columns is not None and hasattr(value, 'columns'):
        actual = [str(x) for x in value.columns]
        if actual != list(columns):
            raise ValueError(f'{name} columns must match training names and order: {list(columns)}')
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name} must contain numeric values') from exc
    if array.ndim != 2:
        raise ValueError(f'{name} must be a two-dimensional matrix')
    return array.copy()


def _names(value, explicit, count, prefix):
    supplied = list(value.columns) if hasattr(value, 'columns') else None
    names = explicit if explicit is not None else supplied
    if names is None:
        names = [f'{prefix}{j + 1}' for j in range(count)]
    if isinstance(names, str):
        raise ValueError(f'{prefix} names must be a sequence')
    names = [str(name) for name in names]
    if len(names) != count or len(set(names)) != count or any(not x for x in names):
        raise ValueError(f'{prefix} names must be unique nonempty names, one per column')
    return names


def normalize_groups(groups, m):
    if groups is None:
        return None
    values = np.asarray(groups, dtype=object)
    if values.shape != (m,):
        raise ValueError('groups must have one label per exposure')
    result = []
    for label in values:
        if label is None or (isinstance(label, (float, np.floating)) and not np.isfinite(label)):
            raise ValueError('groups cannot contain missing/nonfinite labels')
        if not isinstance(label, (str, int, float, np.integer, np.floating)):
            raise ValueError('group labels must be strings or finite numbers')
        result.append(str(label))
    return result


def prepare_data(y, Z, X=None, *, family='gaussian', standardize=True,
                 intercept=True, missing='raise', exposure_names=None,
                 covariate_names=None, groups=None, cluster=None):
    """Return copied model data, with joint complete-case rows and ddof=1 scaling."""
    if family not in ('gaussian', 'binomial'):
        raise ValueError("family must be 'gaussian' or 'binomial'")
    if missing not in ('raise', 'drop'):
        raise ValueError("missing must be 'raise' or 'drop'")
    if not isinstance(standardize, (bool, np.bool_)) or not isinstance(intercept, (bool, np.bool_)):
        raise ValueError('standardize and intercept must be boolean')
    z = matrix(Z, 'Z')
    if z.shape[1] == 0:
        raise ValueError('Z must contain at least one exposure')
    outcome = np.asarray(y, dtype=float)
    if outcome.ndim != 1 or len(outcome) != len(z):
        raise ValueError('y must be a vector with one value per row of Z')
    x = np.empty((len(z), 0)) if X is None else matrix(X, 'X')
    if len(x) != len(z):
        raise ValueError('X and Z must have the same number of rows')
    znames = _names(Z, exposure_names, z.shape[1], 'Z')
    xnames = _names(X, covariate_names, x.shape[1], 'X')
    if intercept and 'Intercept' in xnames:
        raise ValueError('covariate name Intercept is reserved when intercept=True')
    original_n = len(z)
    # NaN denotes missingness. Infinity is invalid even with complete-case drop.
    if any(np.isinf(a).any() for a in (z, outcome, x)):
        raise ValueError('inputs cannot contain infinity')
    keep = np.isfinite(outcome) & np.isfinite(z).all(axis=1) & np.isfinite(x).all(axis=1)
    keep &= cluster_mask(cluster, original_n)
    if missing == 'raise' and not keep.all():
        raise ValueError("missing input values; use missing='drop' for joint complete cases")
    rows = np.flatnonzero(keep)
    z, outcome, x = z[keep], outcome[keep], x[keep]
    if len(z) < 2:
        raise ValueError('at least two complete observations are required')
    if family == 'binomial' and not np.isin(outcome, [0, 1]).all():
        raise ValueError('binomial y must contain only 0 and 1')
    center = z.mean(axis=0) if standardize else np.zeros(z.shape[1])
    scale = z.std(axis=0, ddof=1) if standardize else np.ones(z.shape[1])
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('standardization requires finite, nonconstant exposures')
    design = np.column_stack((np.ones(len(x)), x)) if intercept else x.copy()
    if design.shape[1]:
        # Diagnose rank after column scaling so a change of measurement units
        # cannot make an otherwise independent covariate look rank deficient.
        # Keep the actual model design in the caller's original units.
        magnitudes = np.max(np.abs(design), axis=0)
        if (magnitudes == 0).any() or len(design) <= design.shape[1]:
            raise ValueError('X design must have full column rank and fewer columns than observations; do not supply a duplicate intercept')
        diagnostic_design = design / magnitudes
        if np.linalg.matrix_rank(diagnostic_design) < design.shape[1]:
            raise ValueError('X design must have full column rank and fewer columns than observations; do not supply a duplicate intercept')
    transformed = (z - center) / scale
    if not np.isfinite(transformed).all():
        raise ValueError('exposure transformation overflowed; rescale inputs')
    cluster_codes, cluster_levels = encode_clusters(cluster, original_n, rows)
    return dict(y=outcome.copy(), Z=transformed, Z_original=z.copy(), X=design,
                X_original=x.copy(), z_center=center, z_scale=scale, rows=rows,
                original_n=original_n, exposure_names=znames,
                covariate_names=(['Intercept'] + xnames if intercept else xnames),
                groups=normalize_groups(groups, z.shape[1]),
                cluster_codes=cluster_codes, cluster_levels=cluster_levels)


def prediction_matrix(value, names, *, name='Znew'):
    """Check prediction matrix, including exact DataFrame column order."""
    array = matrix(value, name, columns=names)
    if array.shape[1] != len(names) or not np.isfinite(array).all() or array.shape[0] == 0:
        raise ValueError(f'{name} must have nonempty finite rows and {len(names)} columns')
    return array
