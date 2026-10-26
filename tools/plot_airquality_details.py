"""Plot additional ozone views from saved public tables, without fitting a model.

From a source checkout with Python, NumPy, and Matplotlib installed::

    python tools/plot_airquality_details.py

Outputs are written to docs/figures by default. Existing ozone-detail exports
are replaced on rerun; source tables and earlier figures are never written.
"""
from __future__ import annotations

# ruff: noqa: E402
# Thread limits must precede numerical imports; select Agg before pyplot.
import argparse
import csv
import hashlib
import json
import os
import platform
from datetime import date
from pathlib import Path

# Keep library thread pools bounded before importing numerical libraries.
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['Solar.R', 'Wind', 'Temp']
LABELS = ['Solar radiation (Langleys)', 'Wind (mph)', 'Temperature (°F)']
PALETTE = ['#0072B2', '#B24A00', '#006B55']
MONTHS = ['May', 'June', 'July', 'August', 'September']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def clean_axes(axes):
    for ax in np.asarray(axes).ravel():
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(False)
        ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs/figures')
    parser.add_argument('--only', choices=['ozone-availability', 'ozone-observed',
                        'ozone-parameters', 'ozone-diagnostics', 'ozone-support'],
                        help='Export only the selected figure.')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = ROOT / 'docs/results/airquality'
    sources = [ROOT / 'docs/data/airquality.csv', ROOT / 'docs/data/airquality-provenance.json',
               *[results / name for name in ['parameters.csv', 'diagnostics.csv',
                                            'curves.csv', 'observations.csv', 'results.json']]]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    originals = [ROOT / 'docs/figures' / f'{stem}.{ext}' for stem in
                 ['airquality', 'count-survival', 'response'] for ext in ['pdf', 'png']]
    original_hashes = {str(p.relative_to(ROOT)): sha(p) for p in originals}
    rows = read_csv(sources[0])
    provenance = json.loads(sources[1].read_text())
    summary = json.loads((results / 'results.json').read_text())
    parameters = read_csv(results / 'parameters.csv')
    diagnostics = read_csv(results / 'diagnostics.csv')
    curves = read_csv(results / 'curves.csv')
    observations = read_csv(results / 'observations.csv')
    fields = ['Ozone', *NAMES]
    values = np.array([[np.nan if row[k] == 'NA' else float(row[k]) for k in fields] for row in rows])
    keep = np.isfinite(values).all(axis=1)
    retained = np.flatnonzero(keep) + 1
    missing = {k: sum(row[k] == 'NA' for row in rows) for k in rows[0]}
    counts = [{'month': month, 'recorded': sum(int(r['Month']) == month for r in rows),
               'complete': sum(int(r['Month']) == month and bool(k) for r, k in zip(rows, keep))}
              for month in range(5, 10)]
    assert len(rows) == 153 and int(keep.sum()) == 111
    assert source_hashes['docs/data/airquality.csv'] == provenance['sha256'] == summary['data_sha256']
    assert retained.tolist() == provenance['retained_rows']
    assert (np.flatnonzero(~keep) + 1).tolist() == provenance['excluded_rows'] == summary['profile']['excluded_rows']
    assert missing == provenance['missing'] == summary['profile']['missing_by_field']
    assert counts == summary['profile']['month_counts']
    assert [int(r['source_row']) for r in observations] == retained.tolist()
    assert all(float(r['ozone_ppb']) == values[int(r['source_row']) - 1, 0] for r in observations)
    assert len(parameters) == len(diagnostics) == 10
    assert [r['parameter'] for r in parameters] == [r['parameter'] for r in diagnostics]
    dates = [date(1973, int(r['Month']), int(r['Day'])) for r in rows]
    assert all((b - a).days == 1 for a, b in zip(dates, dates[1:]))
    checks = {'recorded': len(rows), 'complete': int(keep.sum()), 'missing_by_field': missing,
              'month_counts': counts, 'source_hash_and_retained_rows_match': True,
              'observed_ozone_and_source_rows_match': True, 'parameters': len(parameters),
              'diagnostics': len(diagnostics), 'calendar_days_contiguous': True}
    exported = []

    def export(fig, stem, caption, alt, transform, inputs, data_rows=None, extra=None):
        if args.only is not None and stem != args.only:
            plt.close(fig)
            return
        dest = args.output_dir / stem
        if data_rows is not None:
            write_csv(dest.with_suffix('.csv'), data_rows)
        fig.canvas.draw()
        # Check every text artist against the fixed-size canvas, without changing export bounds.
        width, height = fig.canvas.get_width_height()
        renderer = fig.canvas.get_renderer()
        clipped = []
        for ax in fig.axes:
            box = ax.get_tightbbox(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
                clipped.append(ax.get_title())
        if clipped:
            raise ValueError(f'Text outside canvas: {clipped}')
        for ext in ['pdf', 'png']:
            fig.savefig(dest.with_suffix('.' + ext), dpi=450, facecolor='white', transparent=False)
        png_data = dest.with_suffix('.png').read_bytes()
        png_pixels = [int.from_bytes(png_data[16:20], 'big'),
                      int.from_bytes(png_data[20:24], 'big')]
        manifest = {'description': caption, 'alt_text': alt, 'transformations': transform,
                    'inputs': {p: source_hashes[p] for p in inputs},
                    'script_sha256': sha(Path(__file__)),
                    'size_inches': list(fig.get_size_inches()), 'png_dpi': 450,
                    'png_pixels': png_pixels,
                    'background': 'opaque white', 'pdf_fonttype': 42,
                    'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                                 'matplotlib': matplotlib.__version__},
                    'source_checks': checks, 'text_within_canvas': True,
                    'original_figures_sha256': original_hashes,
                    'outputs_sha256': {ext: sha(dest.with_suffix('.' + ext)) for ext in
                                       (['pdf', 'png', 'csv'] if data_rows is not None else ['pdf', 'png'])},
                    'export_style': {'width_mm': 182.88, 'font_family': 'Arial',
                                     'font_size_pt': 8.2, 'panel_font_size_pt': 8.5,
                                     'axes_width_pt': .5, 'background_grids': False,
                                     'bar_hatching': False}}
        if extra:
            manifest.update(extra)
        dest.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        exported.append(stem)
        plt.close(fig)

    def panel(ax, letter):
        ax.text(0, 1.035, letter, transform=ax.transAxes, ha='left', va='bottom',
                fontsize=8.5, fontweight='bold')

    with plt.rc_context({'font.size': 8.2, 'axes.titlesize': 8.2, 'axes.labelsize': 8.2,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
                         'axes.linewidth': .5, 'axes.spines.top': False, 'axes.spines.right': False,
                         'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                         'xtick.major.width': .5, 'ytick.major.width': .5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'xtick.direction': 'out', 'ytick.direction': 'out',
                         'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42}):
        # Keep calendar order and exact daily missingness; separate the four rows.
        fig = plt.figure(figsize=(7.2, 3.7), facecolor='white')
        ax = fig.add_axes([.16, .625, .70, .225])
        bars = fig.add_axes([.16, .135, .70, .31])
        ink, recorded, absent, complete = '#222222', '#82949F', '#A54532', '#176B86'
        available = np.isfinite(values).T
        assert int((~available).sum()) == 44
        assert np.array_equal(available.all(axis=0), keep)
        for j in range(4):
            for i in range(153):
                present = bool(available[j, i])
                ax.add_patch(plt.Rectangle((i - .43, j - .29), .86, .58,
                                          facecolor=recorded if present else 'white',
                                          edgecolor='none'))
            absent_days = np.flatnonzero(~available[j])
            ax.scatter(absent_days, np.full(len(absent_days), j), marker='x',
                       s=6, color=absent, linewidths=.65, zorder=3)
            ax.text(1.12, j, f'{int(available[j].sum())}/153',
                    transform=ax.get_yaxis_transform(), va='center', ha='right',
                    fontsize=8, color=ink)
        starts = [next(i for i, r in enumerate(rows) if int(r['Month']) == month)
                  for month in range(5, 10)]
        ends = starts[1:] + [153]
        centers = [(start + end - 1) / 2 for start, end in zip(starts, ends)]
        for start in starts[1:]:
            ax.axvline(start - .5, ymin=.04, ymax=.96, color='white', lw=2)
        ax.set(xlim=(-.8, 152.8), ylim=(3.65, -.65), yticks=range(4),
               yticklabels=['Ozone', 'Solar radiation', 'Wind', 'Temperature'],
               xticks=centers, xticklabels=MONTHS)
        ax.spines[:].set_visible(False)
        ax.tick_params(axis='both', length=0, pad=5)
        fig.text(.105, .91, 'a', fontsize=9, weight='bold', color=ink)
        fig.text(.16, .91, 'Daily availability', fontsize=8.5, color=ink)
        fig.text(.94, .865, 'Recorded', ha='right', fontsize=7.5, color=ink)
        ax.legend(handles=[Patch(facecolor=recorded, label='Recorded'),
                           plt.Line2D([], [], color=absent, marker='x', ls='',
                                      markersize=4, label='Missing')],
                  loc='lower right', bbox_to_anchor=(1, 1.11), borderaxespad=0,
                  ncols=2, frameon=False, handlelength=1.1, columnspacing=1.5,
                  handletextpad=.5, fontsize=7.5)

        # Bar outlines show all recorded days; blue fill shows complete rows.
        y = np.arange(5)
        total = np.array([r['recorded'] for r in counts])
        n_complete = np.array([r['complete'] for r in counts])
        bars.barh(y, total, height=.52, facecolor='white', edgecolor='#737D82',
                  linewidth=.7, zorder=1)
        bars.barh(y, n_complete, height=.52, facecolor=complete,
                  edgecolor=complete, linewidth=.7, zorder=2)
        for j, row in enumerate(counts):
            bars.text(1.12, j, f"{row['complete']}/{row['recorded']}",
                      transform=bars.get_yaxis_transform(), ha='right',
                      va='center', fontsize=8, color=ink)
        bars.set(yticks=y, yticklabels=MONTHS, ylim=(4.7, -.7),
                 xlim=(0, 33), xticks=[0, 10, 20, 30], xlabel='Days')
        bars.spines[['top', 'right', 'left']].set_visible(False)
        bars.spines['bottom'].set_color('#737D82')
        bars.tick_params(axis='y', length=0, pad=5)
        bars.tick_params(axis='x', colors=ink, length=3)
        bars.set_axisbelow(True)
        fig.text(.105, .51, 'b', fontsize=9, weight='bold', color=ink)
        fig.text(.16, .51, 'Complete days by month', fontsize=8.5, color=ink)
        fig.text(.94, .465, 'Complete / recorded', ha='right', fontsize=7.5, color=ink)
        bars.legend(handles=[Patch(facecolor=complete, edgecolor=complete, label='Complete'),
                             Patch(facecolor='white', edgecolor='#737D82', label='Recorded')],
                    loc='lower right', bbox_to_anchor=(1, 1.14), borderaxespad=0,
                    ncols=2, frameon=False, handlelength=1.1, columnspacing=1.5,
                    handletextpad=.5, fontsize=7.5)
        export(fig, 'ozone-availability',
               'Daily measurement availability and monthly complete-case counts for all 153 days from May through September 1973. Crosses mark missing measurements; blue bar lengths show complete days within outlines for all recorded days. Fractions give exact counts. No values are imputed.',
               'Four calendar-ordered availability rows show 37 missing ozone measurements, seven missing solar-radiation measurements, and no missing wind or temperature. Recorded counts are 116, 146, 153 and 153 of 153 days. Monthly complete/recorded counts are 24/31, 9/30, 26/31, 23/31 and 29/30; 111 days are complete in total.',
               ['Classify the four analysis fields as recorded or NA for each original calendar day.',
                'Preserve all 153 calendar positions and all 44 missing cells, including two days missing both ozone and solar radiation.',
                'Count complete and recorded days by month; no imputation or smoothing.'],
               ['docs/data/airquality.csv', 'docs/data/airquality-provenance.json', 'docs/results/airquality/results.json'], counts,
               extra={'export_style': {'width_mm': 182.88, 'height_mm': 93.98,
                                       'font_family': 'Arial', 'font_size_pt': 8.2,
                                       'panel_font_size_pt': 9, 'axes_width_pt': .5,
                                       'background_grids': False, 'bar_hatching': False},
                      'encoding': {'recorded_cell': recorded, 'missing_cross': absent,
                                   'complete_bar': complete, 'recorded_bar': 'white outlined',
                                   'calendar_order': True, 'month_separators': 'white'},
                      'recorded_per_variable': available.sum(axis=1).tolist(),
                      'missing_cells': int((~available).sum())})

        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), sharey=True, layout='constrained')
        for j, ax in enumerate(axes):
            ax.scatter(values[keep, j + 1], values[keep, 0], s=12, color=PALETTE[j], marker=['o', 's', '^'][j], alpha=.8,
                       edgecolors='white', linewidths=.25)
            ax.set(xlabel=LABELS[j], title=['Solar radiation', 'Wind', 'Temperature'][j], ylim=(0, 180))
        axes[0].set_ylabel('Ozone (ppb)')
        for j, ax in enumerate(axes):
            panel(ax, chr(97 + j))
        clean_axes(axes)
        observed_rows = [{'source_row': int(i + 1), 'date': dates[i].isoformat(),
                          **{name: rows[i][name] for name in fields}} for i in np.flatnonzero(keep)]
        export(fig, 'ozone-observed',
               'Observed ozone against solar radiation, wind, and temperature for the same 111 complete days used in the saved analysis. Each point is one day. Values are on their original scales, with no smoothing or adjustment for month.',
               'Three scatter plots show ozone in parts per billion against solar radiation in Langleys, wind in miles per hour, and temperature in degrees Fahrenheit, for 111 complete days.',
               ['Retain rows with nonmissing Ozone, Solar.R, Wind, and Temp.',
                'Plot raw pairs on original scales; no jitter, aggregation, smoothing, or model adjustment.'],
               ['docs/data/airquality.csv', 'docs/data/airquality-provenance.json', 'docs/results/airquality/observations.csv'], observed_rows)

        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), layout='constrained',
                                  gridspec_kw={'width_ratios': [1, 1.15]})
        groups = [[r for r in parameters if r['parameter'].startswith('r[')],
                  [r for r in parameters if r['parameter'].startswith('beta[') and r['parameter'] != 'beta[Intercept]']]
        for j, (ax, selected) in enumerate(zip(axes, groups)):
            means, lower, upper = [np.array([float(r[k]) for r in selected]) for k in ['mean', 'lower', 'upper']]
            assert np.all(lower <= means) and np.all(means <= upper)
            y = np.arange(len(selected))
            ax.errorbar(means, y, xerr=[means - lower, upper - means], fmt='o', color=PALETTE[j], capsize=2.5, lw=.9, ms=4)
            ax.set(yticks=y, yticklabels=[r['parameter'].split('[')[1][:-1] for r in selected],
                   ylim=(len(selected) - .5, -.5), xlabel=[r'Kernel relevance $r_m$', r'Month coefficient $\beta$ (log ozone)'][j],
                   title=['Kernel relevance', 'Month adjustment vs May'][j])
            ax.axvline(0, color='.6', ls=':', lw=.5)
            panel(ax, chr(97 + j))
        axes[0].set_xlim(left=0)
        clean_axes(axes)
        export(fig, 'ozone-parameters',
               'Saved posterior means (points) and 95% equal-tailed posterior intervals (lines) for exposure kernel relevance parameters r_m and month coefficients beta. Month coefficients are relative to May; the intercept is omitted. The panels use different scales. Relevance parameters describe the kernel and are not causal importance measures. Intervals use independent Gaussian working errors.',
               'Two interval plots separate three exposure relevance parameters from four month coefficients relative to May. Points are saved posterior means and lines are 95 percent equal-tailed posterior intervals.',
               ['Select three r parameters and four month beta coefficients from parameters.csv.',
                'Omit lambda, sigma2, and beta[Intercept]; plot saved means and interval endpoints unchanged.'],
               ['docs/results/airquality/parameters.csv'],
               extra={'displayed_parameters': [r['parameter'] for group in groups for r in group],
                      'uncertainty': '95% equal-tailed posterior intervals under independent Gaussian working errors'})

        fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.8), sharey=True, layout='constrained',
                                  gridspec_kw={'width_ratios': [1, 1.2, 1.2]})
        diag_labels = [r'$\lambda$', r'$\sigma^2$', r'$r_{\mathrm{Solar.R}}$', r'$r_{\mathrm{Wind}}$',
                       r'$r_{\mathrm{Temp}}$', r'$\beta_{\mathrm{Intercept}}$', r'$\beta_{\mathrm{June}}$',
                       r'$\beta_{\mathrm{July}}$', r'$\beta_{\mathrm{August}}$', r'$\beta_{\mathrm{September}}$']
        y = np.arange(10)
        for j, (ax, key) in enumerate(zip(axes, ['rhat', 'ess_bulk', 'ess_tail'])):
            nums = [float(r[key]) for r in diagnostics]
            ax.scatter(nums, y, s=18, color=PALETTE[j], marker=['o', 's', '^'][j])
            ax.set(yticks=y, yticklabels=diag_labels, ylim=(9.6, -.6),
                   title=['Split R-hat', 'Bulk ESS', 'Tail ESS'][j],
                   xlabel=[r'$\widehat{R}$', 'Effective sample size', 'Effective sample size'][j])
            if j:
                ax.set_xlim(0, 72000)
                ax.set_xticks([0, 20000, 40000, 60000], ['0', '20,000', '40,000', '60,000'])
                ax.tick_params(axis='x', labelsize=8)
            else:
                ax.set_xlim(.9998, 1.006)
                ax.set_xticks([1, 1.002, 1.004, 1.006])
                ax.tick_params(axis='x', labelsize=8)
                ax.axvline(1, color='.6', ls=':', lw=.5)
            panel(ax, chr(97 + j))
        clean_axes(axes)
        export(fig, 'ozone-diagnostics',
               'Saved sampling diagnostics for all ten parameters. R-hat compares within-chain and between-chain variation using rank-normalized and folded split chains. Bulk effective sample size (ESS) summarizes sampling efficiency for the main distribution; tail ESS uses the smaller 5% and 95% indicator-tail estimate. Here lambda = tau/sigma^2, where tau is the kernel variance amplitude and sigma^2 is the error variance; r denotes exposure kernel relevance and beta denotes regression coefficients. These diagnostics do not establish convergence or validate model assumptions.',
               'Aligned dot plots show rank-normalized split R-hat, bulk effective sample size, and tail effective sample size for all ten saved parameters. Bulk and tail panels share the same scale.',
               ['Display all ten diagnostics.csv rows, preserving source order and values.',
                'Use separate R-hat, bulk ESS, and tail ESS axes; show 1 as a reference for R-hat, without a pass/fail threshold.'],
               ['docs/results/airquality/diagnostics.csv'],
               extra={'diagnostic_ranges': {k: [min(float(r[k]) for r in diagnostics), max(float(r[k]) for r in diagnostics)] for k in ['rhat', 'ess_bulk', 'ess_tail']}})

        fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True, layout='constrained')
        support_ranges = {}
        for j, (ax, name) in enumerate(zip(axes, NAMES)):
            selected = [r for r in curves if r['variable'] == name]
            x = np.array([float(r['value']) for r in selected])
            distance = np.array([float(r['nearest_standardized_distance']) for r in selected])
            assert len(selected) == {'Solar.R': 21, 'Wind': 17, 'Temp': 21}[name]
            assert np.all(np.diff(x) > 0) and np.all(distance >= 0)
            ax.scatter(x, distance, s=16, color=PALETTE[j], marker=['o', 's', '^'][j])
            ax.set(xlabel=LABELS[j], title=['Solar radiation', 'Wind', 'Temperature'][j], ylim=(0, .9))
            panel(ax, chr(97 + j))
            support_ranges[name] = {'rows': len(selected), 'exposure_range': [float(x.min()), float(x.max())],
                                    'distance_range': [float(distance.min()), float(distance.max())]}
        axes[0].set_ylabel('Nearest-profile distance\n(standardized exposures)')
        clean_axes(axes)
        export(fig, 'ozone-support',
               'Nearest observed-profile distances for the saved marginal-curve locations: 21 each for solar radiation and temperature, and 17 distinct wind values. At each location the other exposures are held at their complete-case medians. Distance is the Euclidean distance to the nearest complete observed exposure profile after dividing each coordinate difference by its complete-case sample standard deviation. The saved locations span the 5th–95th marginal percentiles. Greater distance means the profile is farther from the observed data; no distance threshold or validity guarantee is implied.',
               'Three point plots show the saved standardized nearest-profile distance along each exposure grid, with a common vertical scale and exposures on their original scales.',
               ['Read value and nearest_standardized_distance from all 59 curves.csv rows unchanged; wind percentile ties yield 17 distinct saved locations.',
                'No distance recomputation, smoothing, thresholding, model fit, or posterior evaluation.'],
               ['docs/results/airquality/curves.csv'], extra={'displayed_ranges': support_ranges})

    assert all(sha(ROOT / name) == digest for name, digest in source_hashes.items())
    assert all(sha(ROOT / name) == digest for name, digest in original_hashes.items())
    print(json.dumps({'exported': exported, 'source_tables_unchanged': True,
                      'original_figures_unchanged': True, 'checks': checks}, indent=2))


if __name__ == '__main__':
    main()
