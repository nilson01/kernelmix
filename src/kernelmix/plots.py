"""Optional Matplotlib plots. All functions return axes and leave display to you.

Matplotlib is imported only when a plot is requested. No function saves files,
shows figures, or changes global styles. Posterior intervals are pointwise;
they are not simultaneous uncertainty bands. Values remain available through
result ``to_records()`` methods for captions and accessible data tables.
"""
from numbers import Integral

import numpy as np

__all__ = ['plot_trace', 'plot_pips', 'plot_response', 'plot_overall', 'plot_surface', 'plot_survival']

_COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#000000')
_STYLES = ('-', '--', '-.', ':')
_MARKERS = ('o', 's', '^', 'D', 'v')


def _axis(ax):
    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(layout='constrained')
    return ax


def _interval_label(result):
    return f'{100 * result.probability:g}% pointwise posterior interval'


def _response_label(result):
    centered = result.metadata.get('center', False)
    if result.scale == 'latent':
        return 'Change in h' if centered or result.kind not in ('univariate', 'bivariate') else 'h'
    probability = result.metadata.get('family') == 'binomial'
    label = ('Relative hazard' if result.metadata.get('family') == 'survival' else
             'Predicted probability' if probability else 'Predicted count mean' if
             result.metadata.get('family') in ('poisson', 'negative_binomial') else 'Predicted outcome mean')
    return ('Change in ' + label.lower()) if centered or result.kind not in ('univariate', 'bivariate') else label


def plot_trace(fit, parameter='lambda', indices=None, ax=None):
    """Plot one parameter's saved draws for each chain.

    ``indices`` selects one component of an array-valued parameter. Use one
    integer per parameter axis, for example ``indices=0`` for default-sampler
    beta or ``indices=(0, 0)`` for structured beta. Only saved draws are shown;
    warmup draws are excluded. A trace does not establish convergence.
    """
    values = np.asarray(fit.posterior_samples(parameter))
    if values.ndim < 2:
        raise ValueError('parameter must have chain and draw dimensions')
    if values.ndim > 2:
        if indices is None:
            raise ValueError('indices are required for a nonscalar parameter')
        trailing = (indices,) if isinstance(indices, Integral) else tuple(indices)
        if len(trailing) != values.ndim - 2 or any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral) for i in trailing):
            raise ValueError('indices must specify one integer per parameter dimension')
        if any(i < 0 or i >= size for i, size in zip(trailing, values.shape[2:])):
            raise ValueError('parameter index is out of range')
        values = values[(slice(None), slice(None)) + trailing]
    elif indices is not None:
        raise ValueError('indices do not apply to a scalar parameter')
    ax = _axis(ax)
    x = np.arange(1, values.shape[1] + 1)
    for chain, row in enumerate(values):
        ax.plot(x, row, color=_COLORS[chain % len(_COLORS)],
                linestyle=_STYLES[chain % len(_STYLES)], linewidth=1,
                label=f'Chain {chain + 1}')
    suffix = '' if indices is None else f'[{indices}]'
    ax.set(xlabel='Saved draw', ylabel=f'{parameter}{suffix}', title='Posterior trace (warmup excluded)')
    ax.legend()
    return ax


def plot_pips(fit, ax=None):
    """Plot each exposure's posterior inclusion probability (PIP) from zero to one.

    A PIP is the proportion of saved states in which an exposure is selected.
    Group selection also shows group PIPs and exposure PIPs among states where
    the group is selected. Marker shapes distinguish these quantities; they are
    probabilities, not uncertainty intervals. A selection fit is required.
    """
    rows = fit.inclusion_probabilities()
    if not rows:
        raise ValueError('fit has no inclusion probabilities to plot')
    ax = _axis(ax)
    y = np.arange(len(rows))
    ax.scatter([row['pip'] for row in rows], y, marker='o', color=_COLORS[0], label='Exposure PIP')
    if 'group_pip' in rows[0]:
        ax.scatter([row['group_pip'] for row in rows], y + .15, marker='s',
                   facecolors='none', edgecolors=_COLORS[1], label='Group PIP')
        ax.scatter([row['conditional_pip'] for row in rows], y - .15, marker='^',
                   color=_COLORS[2], label='Within-group conditional PIP')
    ax.set_yticks(y, [row['variable'] for row in rows])
    ax.set(xlim=(0, 1), xlabel='Posterior inclusion probability', ylabel='Exposure')
    ax.legend()
    return ax


def plot_response(curve, ax=None):
    """Plot exposure curves with a separate credible interval at each grid point.

    Intervals have equal posterior probability below and above their limits.
    Multiple exposures share one axis; use separate calls when their units
    differ. Colors, line styles, markers and labels identify each curve.
    """
    if curve.kind != 'univariate':
        raise ValueError('plot_response requires a univariate_response result')
    ax = _axis(ax)
    names = list(dict.fromkeys(location['variable'] for location in curve.locations))
    for index, name in enumerate(names):
        selected = np.array([i for i, row in enumerate(curve.locations) if row['variable'] == name])
        x = np.array([curve.locations[i]['value'] for i in selected])
        order = np.argsort(x)
        selected, x = selected[order], x[order]
        color = _COLORS[index % len(_COLORS)]
        ax.plot(x, curve.mean[selected], label=name, color=color,
                linestyle=_STYLES[index % len(_STYLES)],
                marker=_MARKERS[index % len(_MARKERS)], markevery=max(1, len(x) // 8), markersize=3)
        ax.fill_between(x, curve.lower[selected], curve.upper[selected], color=color, alpha=.16,
                        label=_interval_label(curve) if index == 0 else None)
    if curve.metadata.get('center', False):
        ax.axhline(0, color='#555555', linewidth=.8, linestyle=':')
    ax.set(xlabel=f'{names[0]} (original units)' if len(names) == 1 else 'Exposure value (original units)',
           ylabel=_response_label(curve))
    ax.legend()
    return ax


def plot_overall(effect, ax=None):
    """Plot profile differences from overall_effect, with intervals at each quantile."""
    if effect.kind != 'overall':
        raise ValueError('plot_overall requires an overall_effect result')
    ax = _axis(ax)
    x = np.array([row['quantile'] for row in effect.locations])
    order = np.argsort(x)
    ax.vlines(x, effect.lower, effect.upper, color=_COLORS[0], label=_interval_label(effect))
    ax.plot(x[order], effect.mean[order], color=_COLORS[0], marker='o', label='Posterior mean')
    ax.axhline(0, color='#555555', linewidth=.8, linestyle=':')
    ax.set(xlabel='Quantile of every exposure', ylabel=_response_label(effect),
           title=f"Reference: all exposures at quantile {effect.metadata['reference']:g}")
    ax.legend()
    return ax


def plot_surface(surface, ax=None):
    """Plot a posterior mean surface; uncertainty remains in the result.

    The color map shows values over the two exposure grids. Centered surfaces
    use different colors for negative and positive values, with zero at the
    middle of the scale. Uncentered surfaces use one ordered color scale.
    The colorbar labels the posterior mean. Use result.lower/upper or
    to_records() for uncertainty.
    """
    if surface.kind != 'bivariate':
        raise ValueError('plot_surface requires a bivariate_response result')
    if min(surface.shape) < 2:
        raise ValueError('surface plotting requires at least two values per grid')
    ax = _axis(ax)
    values = surface.mean.reshape(surface.shape).T
    options = {}
    if surface.metadata.get('center', False):
        from matplotlib.colors import Normalize
        limit = max(float(np.max(np.abs(values))), np.finfo(float).eps)
        options.update(cmap='RdBu_r', norm=Normalize(vmin=-limit, vmax=limit))
    else:
        options['cmap'] = 'cividis'
    artist = ax.pcolormesh(surface.grids[0], surface.grids[1], values,
                           shading='nearest', **options)
    ax.figure.colorbar(artist, ax=ax, label=f'Posterior mean: {_response_label(surface).lower()}')
    ax.set(xlabel=f'{surface.variables[0]} (original units)',
           ylabel=f'{surface.variables[1]} (original units)',
           title='Posterior mean surface (uncertainty not shown)')
    return ax



def plot_survival(prediction, row=0, ax=None):
    """Plot survival, cumulative incidence or hazards with pointwise intervals.

    Survival is the probability of no event by a time. Cumulative incidence is
    the probability of a specified cause by a time, accounting for other causes.
    Hazards are event rates among those still at risk and are plotted only at
    evaluated times. Intervals apply to each time separately.
    """
    if prediction.kind not in ('survival', 'hazard', 'cumulative_incidence') or prediction.mean.ndim != 2:
        raise ValueError('plot_survival requires a SurvivalPrediction')
    if isinstance(row, (bool, np.bool_)) or not isinstance(row, Integral) or not 0 <= row < prediction.mean.shape[0]:
        raise ValueError('row must index a prediction profile')
    ax = _axis(ax)
    if prediction.kind in ('survival', 'cumulative_incidence'):
        ax.plot(prediction.times, prediction.mean[row], color=_COLORS[0], label='Posterior mean')
        ax.fill_between(prediction.times, prediction.lower[row], prediction.upper[row],
                        color=_COLORS[0], alpha=.16, label=_interval_label(prediction))
        ax.set(xlabel='Time', ylabel=r'$S(t)$' if prediction.kind=='survival' else r'$F_k(t)$', ylim=(0, 1))
    else:
        # A sparse time grid need not include every baseline cut. Show only
        # evaluated hazards, rather than implying a false step location.
        ax.vlines(prediction.times, prediction.lower[row], prediction.upper[row],
                  color=_COLORS[0], label=_interval_label(prediction))
        ax.plot(prediction.times, prediction.mean[row], color=_COLORS[0],
                marker='o', linestyle='none', label='Posterior mean')
        ax.set(xlabel='Time', ylabel='Hazard (per time unit)')
    ax.legend()
    return ax
