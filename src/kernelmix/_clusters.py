"""Typed observation-cluster labels and retained-row encoding.

Exposure-selection groups are handled separately from these observation labels.
"""
import numpy as np


def canonical_label(value):
    """Return a hashable typed identity; integer1 differs from string'1'.

    Accept only integers (excluding booleans) and nonempty strings. Missing
    values are handled by ``cluster_mask``, never treated as an actual level.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('cluster labels must be integers or nonempty strings, not booleans')
    if isinstance(value, (int, np.integer)):
        return ('int', int(value))
    if isinstance(value, (str, np.str_)) and len(value) > 0:
        return ('str', str(value))
    raise ValueError('cluster labels must be integers or nonempty strings')


def _labels(values, n):
    labels = np.asarray(values, dtype=object)
    if labels.shape != (n,):
        raise ValueError('cluster must have one label per original observation')
    return labels


def cluster_mask(values, n):
    """Return nonmissing mask; reject unsupported labels even on dropped rows.

    None or floating NaN entries denote missingness. An absent cluster vector
    (``values is None``) returns an all-true mask. Finite floating labels,
    infinities, booleans, empty strings and unhashable labels are invalid.
    """
    if values is None:
        return np.ones(n, dtype=bool)
    labels = _labels(values, n)
    valid = np.ones(n, dtype=bool)
    for i, value in enumerate(labels):
        if value is None or (isinstance(value, (float, np.floating)) and np.isnan(value)):
            valid[i] = False
        else:
            canonical_label(value)
    return valid


def encode_clusters(values, n, rows=None):
    """Return retained integer codes and original typed levels in first-use order.

    ``n`` is the original row count; ``rows`` optionally selects retained rows.
    All unsupported original labels fail. Missing labels are allowed only
    outside retained rows. For absent clustering return ``(None, [])``.
    """
    if values is None:
        return None, []
    labels = _labels(values, n)
    valid = cluster_mask(labels, n)
    if rows is None:
        retained = np.arange(n)
    else:
        retained = np.asarray(rows)
        if retained.ndim != 1 or not np.issubdtype(retained.dtype, np.integer):
            raise ValueError('cluster retained rows must be an integer vector')
        if (retained < 0).any() or (retained >= n).any():
            raise ValueError('cluster retained row index is out of bounds')
    if not valid[retained].all():
        raise ValueError('cluster labels contain missing values in retained rows')
    mapping, levels = {}, []
    codes = np.empty(len(retained), dtype=int)
    for i, row in enumerate(retained):
        identity = canonical_label(labels[row])
        if identity not in mapping:
            mapping[identity] = len(levels)
            levels.append(identity[1])
        codes[i] = mapping[identity]
    return codes, levels
