"""Plot the saved ozone summaries without refitting the model."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Output stem without extension')
    args = parser.parse_args()
    with (args.results / 'curves.csv').open() as stream:
        curves = list(csv.DictReader(stream))
    with (args.results / 'observations.csv').open() as stream:
        observations = list(csv.DictReader(stream))
    palette = ['#0072B2', '#B24A00', '#006B55']
    names = ['Solar.R', 'Wind', 'Temp']
    labels = ['Solar radiation (Langleys)', 'Wind (mph)', 'Temperature (°F)']
    with plt.rc_context({'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.9), layout='constrained')
        for j, name in enumerate(names):
            selected = [r for r in curves if r['variable'] == name]
            x, mean, lower, upper = [np.array([float(r[k]) for r in selected]) for k in
                                    ['value', 'mean', 'lower', 'upper']]
            ax = axes[0, j]
            ax.axhline(0, color='.5', lw=.7, ls=':')
            ax.fill_between(x, lower, upper, color=palette[j], alpha=.18,
                            label='95% pointwise posterior interval')
            ax.plot(x, mean, color=palette[j], lw=1.5, label='Posterior mean')
            ax.set(xlabel=labels[j], ylabel=r'$h(\mathbf{z})-h(\mathbf{z}_{0.5})$' if j == 0 else '')
            ax.text(.02, .97, chr(97+j), transform=ax.transAxes, va='top', fontweight='bold')
        actual = np.array([float(r['log_ozone']) for r in observations])
        fitted = np.array([float(r['fitted_log_ozone']) for r in observations])
        residual = np.array([float(r['residual']) for r in observations])
        ax = axes[1, 0]
        ax.scatter(fitted, actual, s=14, color='.25', alpha=.65)
        limit = [min(actual.min(), fitted.min()), max(actual.max(), fitted.max())]
        ax.plot(limit, limit, color='.5', ls='--', lw=.8)
        ax.set(xlabel='Fitted log ozone', ylabel='Observed log ozone')
        ax.text(.02, .97, 'd', transform=ax.transAxes, va='top', fontweight='bold')
        # Keep missing source dates as gaps: use points, not a connecting line.
        dates = np.array([np.datetime64(r['date']) for r in observations])
        day = (dates - np.datetime64('1973-05-01')).astype('timedelta64[D]').astype(int)
        ax = axes[1, 1]
        ax.scatter(day, residual, s=14, color='.25', alpha=.65)
        ax.axhline(0, color='.5', lw=.8)
        ax.set(xlabel='Days since May 1', ylabel='Log-ozone residual')
        ax.text(.02, .97, 'e', transform=ax.transAxes, va='top', fontweight='bold')
        adjacent = np.diff(dates).astype('timedelta64[D]').astype(int) == 1
        ax = axes[1, 2]
        ax.scatter(residual[:-1][adjacent], residual[1:][adjacent], s=14, color='.25', alpha=.65)
        ax.axhline(0, color='.8', lw=.7)
        ax.axvline(0, color='.8', lw=.7)
        ax.set(xlabel=r'Residual at $t$', ylabel=r'Residual at $t+1$')
        ax.text(.02, .97, 'f', transform=ax.transAxes, va='top', fontweight='bold')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for extension in ['pdf', 'png']:
            fig.savefig(args.output.with_suffix('.'+extension), dpi=300, facecolor='white')
        plt.close(fig)
    manifest = {'inputs': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [args.results/'curves.csv', args.results/'observations.csv']},
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'size_inches': [10.2, 5.9], 'png_dpi': 300,
                'uncertainty': '95% equal-tailed pointwise credible intervals under independent Gaussian working errors; not serial-dependence robust',
                'curve_definition': '5th–95th marginal quantiles; other variables at sample medians; reference is all-median profile',
                'missing_dates': 'unobserved dates not connected or imputed'}
    args.output.with_suffix('.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
