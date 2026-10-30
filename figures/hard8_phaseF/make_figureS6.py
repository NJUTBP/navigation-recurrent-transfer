"""Supplementary Figure S6: optimization and split diagnostics for the extended 8 x 8 task.

Reads figure_source_data/FigureA5 and hard8x8_phaseF_followup.
"""
from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from .nature_style import (
        apply_publication_style, PALETTE, LINEWIDTH, LINEWIDTH_SECONDARY,
        BAND_ALPHA, ERROR_LINEWIDTH, CAPSIZE, deterministic_jitter,
        mean_ci95, mean_sem, reference_line, style_axis, place_panel_labels, legend, categorical_xlim,
        save_publication_figure,
    )
except ImportError:  # allow direct execution from hard8_phaseF/
    from nature_style import (
        apply_publication_style, PALETTE, LINEWIDTH, LINEWIDTH_SECONDARY,
        BAND_ALPHA, ERROR_LINEWIDTH, CAPSIZE, deterministic_jitter,
        mean_ci95, mean_sem, reference_line, style_axis, place_panel_labels, legend, categorical_xlim,
        save_publication_figure,
    )

# Data interface unchanged.
try:
    from . import figure_paths
except ImportError:  # direct execution from hard8_phaseF/
    import figure_paths
AGGREGATE = figure_paths.AGGREGATE
DATA = figure_paths.PHASEF_FIGURE_DATA
FOLLOW = figure_paths.PHASEF_TABLES
OUT = figure_paths.OUT

NAV = PALETTE['navigation']
# Same colours as Figure 6 (role registry).
MATCHED = PALETTE['matched']
FRESH = PALETTE['fresh']
BASELINE = PALETTE['relation']

try:
    from .nature_style import role, line_kw, light_line_kw
except ImportError:
    from nature_style import role, line_kw, light_line_kw

PRIMARY_NAV = 'navigation_trainable_scalar_gain'
PRIMARY_MATCH = 'matched_random_trainable_scalar_gain'
PRIMARY_FRESH = 'fresh_random_trainable_scalar_gain'

SPLITS = ['id_seen_short', 'heldout_short', 'seen_long', 'heldout_long']
SLABEL = ['Seen\nshort', 'Held-out\nshort', 'Seen\nlong', 'Held-out\nlong']

# Keep the original CSV paths and column interface unchanged.
metrics = pd.read_csv(DATA / 'source_level_metrics.csv')
baselines = pd.read_csv(DATA / 'source_level_baselines.csv')
primary = pd.read_csv(DATA / 'primary_generalization_contrasts.csv')
secondary = pd.read_csv(DATA / 'secondary_id_contrasts.csv')
perlen = pd.read_csv(DATA / 'all_per_length.csv')
interv = pd.read_csv(DATA / 'source_level_relation_interventions.csv')
condmat = pd.read_csv(DATA / 'condition_performance_matrix.csv')
updates = pd.read_csv(DATA / 'all_update_curves.csv')
diag = pd.read_csv(DATA / 'source_level_condition_diagnostics.csv')
audit = pd.read_csv(FOLLOW / 'all_dataset_audit.csv')


def _mean_ci_errorbar(ax, x, values, *, color, marker='o', zorder=5, horizontal=False):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    m, lo, hi = mean_ci95(values)
    if horizontal:
        ax.errorbar(
            m, x, xerr=[[m - lo], [hi - m]], fmt=marker, ms=4.4,
            color=color, mfc=color, mec='white', mew=.4, capsize=CAPSIZE,
            elinewidth=ERROR_LINEWIDTH, zorder=zorder,
        )
    else:
        ax.errorbar(
            x, m, yerr=[[m - lo], [hi - m]], fmt=marker, ms=4.4,
            color=color, mfc=color, mec='white', mew=.4, capsize=CAPSIZE,
            elinewidth=ERROR_LINEWIDTH, zorder=zorder,
        )
    return m, lo, hi


def curve_by_update(condition):
    z = updates[updates.condition == condition].copy()
    s = z.groupby(['source_seed', 'update'], as_index=False).validation_accuracy.mean()
    rows = []
    for update, q in s.groupby('update'):
        a = q.validation_accuracy.to_numpy(float)
        m, se = mean_sem(a)
        rows.append((update, m, se))
    return pd.DataFrame(rows, columns=['update', 'mean', 'sem']).sort_values('update')


def panel_learning(ax):
    # Training curves as in Figures 3A/5A -- navigation full colour,
    # comparison conditions light solid lines; role markers carry identity.
    for cond, label, rname in [
        (PRIMARY_NAV, 'Navigation', 'navigation'),
        (PRIMARY_MATCH, 'Matched control', 'matched'),
        (PRIMARY_FRESH, 'Fresh-random control', 'fresh'),
    ]:
        spec = role(rname)
        color = spec['color']
        z = curve_by_update(cond)
        x = z['update'].to_numpy()
        y = z['mean'].to_numpy()
        se = z['sem'].to_numpy()
        ax.fill_between(x, y - se, y + se, color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
        hero = rname == 'navigation'
        ax.plot(x, y, **(line_kw if hero else light_line_kw)(spec, ms=4.0, markevery=4, label=label,
                                                            zorder=4 if hero else 3))
    ax.set_xlim(0, 8100)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 2500, 5000, 7500])
    ax.set_xlabel('Gradient updates')
    ax.set_ylabel('Validation accuracy')
    legend(ax, loc='center right')
    style_axis(ax, max_y_ticks=5, max_x_ticks=4)


def panel_bestupdate(ax):
    conds = [
        PRIMARY_NAV, PRIMARY_MATCH, PRIMARY_FRESH,
        'navigation_trainable_input_adapter', 'matched_random_trainable_input_adapter',
        'fresh_random_trainable_input_adapter',
        'navigation_full_core_trainable', 'matched_random_full_core_trainable',
        'fresh_random_full_core_trainable',
    ]
    labels = [
        r'Navigation, learned $g$', r'Matched, learned $g$', r'Fresh-random, learned $g$',
        r'Navigation, learned $\mathbf{A}$', r'Matched, learned $\mathbf{A}$', r'Fresh-random, learned $\mathbf{A}$',
        'Navigation, full fine-tuning', 'Matched, full fine-tuning', 'Fresh-random, full fine-tuning',
    ]
    colors = [NAV, MATCHED, FRESH] * 3
    markers = ['o', 's', '^'] * 3
    y = np.arange(len(conds), dtype=float)

    for i, (condition, color, marker) in enumerate(zip(conds, colors, markers)):
        vals = pd.to_numeric(diag[diag.condition == condition].best_update, errors='coerce').to_numpy(float)
        vals = vals[np.isfinite(vals)]
        yy = np.full(len(vals), y[i]) + deterministic_jitter(len(vals), width=.07)
        ax.scatter(vals, yy, s=12, color=color, marker=marker, alpha=.28, lw=0, zorder=2)
        _mean_ci_errorbar(ax, y[i], vals, color=color, marker=marker, horizontal=True)

    ax.axvline(7600, color=PALETTE['neutral_mid'], ls=(0, (3, 2)), lw=.85, zorder=0)
    for sep in [2.5, 5.5]:
        ax.axhline(sep, color=PALETTE['neutral_lighter'], lw=.9, zorder=0)

    # Requested: remove the descriptive text beside the dashed vertical line.
    ax.set_yticks(y, labels, fontsize=6.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 8200)
    ax.set_xticks([0, 2500, 5000, 7500])
    ax.set_xlabel('Selected checkpoint (update)')
    style_axis(ax, max_y_ticks=9, max_x_ticks=4)


def panel_baseline(ax):
    x = np.arange(4)
    means, los, his = [], [], []
    for split in SPLITS:
        vals = baselines[(baselines.split == split) & (baselines.baseline == 'start_plus_length')].accuracy.to_numpy(float)
        m, lo, hi = mean_ci95(vals)
        means.append(m)
        los.append(lo)
        his.append(hi)
    means = np.array(means)
    los = np.array(los)
    his = np.array(his)
    ax.errorbar(
        x, means, yerr=[means - los, his - means], fmt='o-', color=BASELINE,
        ms=4.2, lw=LINEWIDTH, capsize=CAPSIZE, elinewidth=ERROR_LINEWIDTH,
        label='Start + path length',
    )
    reference_line(ax, 1 / 64)

    # Requested: remove the descriptive text beside the dashed chance line.
    categorical_xlim(ax, x, pad=.35)
    ax.set_xticks(x, SLABEL)
    ax.set_ylim(0, .20)
    ax.set_ylabel('Test accuracy')
    style_axis(ax, max_y_ticks=5, max_x_ticks=4)


def parse_counts(s):
    return {int(k): int(v) for k, v in json.loads(s).items()}


def panel_lengths(ax):
    uniq = audit.drop_duplicates(subset=['split', 'length_counts'])

    def distribution(split):
        row = uniq[uniq.split == split].iloc[0]
        d = parse_counts(row.length_counts)
        xs = np.array(sorted(d))
        ys = np.array([d[x] for x in xs], float)
        ys /= ys.sum()
        return xs, ys

    # Split identity is independent of the matched-control condition colour.
    specs = [
        ('train_8000', 'Train', NAV, 'o', '-'),
        ('id_seen_short', 'Seen short', '#80858A', 's', '-'),
        ('heldout_short', 'Held-out short', BASELINE, 'D', '--'),
    ]
    xs_seen, ys_seen = distribution('seen_long')
    xs_held, ys_held = distribution('heldout_long')
    combine_long = np.array_equal(xs_seen, xs_held) and np.allclose(ys_seen, ys_held, rtol=0, atol=1e-12)

    for split, label, color, marker, ls in specs:
        xs, ys = distribution(split)
        ax.plot(xs, ys, marker=marker, ms=3.4, lw=LINEWIDTH_SECONDARY, ls=ls, color=color, label=label)

    if combine_long:
        ax.plot(
            xs_seen, ys_seen, marker='^', ms=3.4, lw=LINEWIDTH_SECONDARY,
            ls=(0, (2, 1.5)), color=FRESH, label='Long, seen & held-out',
        )
    else:
        ax.plot(
            xs_seen, ys_seen, marker='^', ms=3.4, lw=LINEWIDTH_SECONDARY,
            ls='-', color=FRESH, label='Seen long',
        )
        xs, ys = distribution('heldout_long')
        ax.plot(
            xs, ys, marker='v', ms=3.4, lw=LINEWIDTH_SECONDARY, ls=':',
            color=PALETTE.get('relation_dark', BASELINE), label='Held-out long',
        )

    # Requested: remove the broad pale long-horizon background; keep the boundary line.
    ax.axvline(5.5, color=PALETTE['neutral_mid'], lw=.8, ls=(0, (3, 2)), zorder=0)
    ax.set_xticks(range(1, 13))
    ax.set_xlim(.7, 12.3)
    ax.set_ylim(0, .48)
    ax.set_xlabel('Path length $L_{\\mathrm{p}}$')
    ax.set_ylabel('Proportion of split')
    legend(ax, loc='upper right')
    style_axis(ax, max_y_ticks=5, max_x_ticks=12)


def make_s6():
    apply_publication_style()
    fig = plt.figure(figsize=(7.25, 5.55))
    gs = fig.add_gridspec(
        2, 2, left=.09, right=.99, top=.94, bottom=.12,
        wspace=.46, hspace=.52, width_ratios=[1, 1.04],
    )
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
    ]
    panel_learning(axs[0])
    panel_bestupdate(axs[1])
    panel_baseline(axs[2])
    panel_lengths(axs[3])
    place_panel_labels(fig, list(zip(axs, 'ABCD')))
    save_publication_figure(fig, OUT / 'SupplementaryFigureS6')


if __name__ == '__main__':
    make_s6()
