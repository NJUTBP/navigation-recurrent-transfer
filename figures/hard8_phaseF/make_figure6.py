from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from .nature_style import (
        apply_publication_style, PALETTE, LINEWIDTH, LINEWIDTH_SECONDARY,
        BAND_ALPHA, ERROR_LINEWIDTH, CAPSIZE, deterministic_jitter,
        mean_ci95, mean_sem, reference_line, style_axis, place_panel_labels,
        legend, annotate, categorical_xlim, save_publication_figure,
        FIG_WIDTH, FS_ANNOT, FS_INPANEL, FS_MIN, role, line_kw, legend_handle,
        accuracy_axis, integer_axis, closed_frame, swarm, summary_errorbar,
        pair_lines, fmt_p, num, difference_spec, lighten,
    )
except ImportError:  # allow direct execution from hard8_phaseF/
    from nature_style import (
        apply_publication_style, PALETTE, LINEWIDTH, LINEWIDTH_SECONDARY,
        BAND_ALPHA, ERROR_LINEWIDTH, CAPSIZE, deterministic_jitter,
        mean_ci95, mean_sem, reference_line, style_axis, place_panel_labels,
        legend, annotate, categorical_xlim, save_publication_figure,
        FIG_WIDTH, FS_ANNOT, FS_INPANEL, FS_MIN, role, line_kw, legend_handle,
        accuracy_axis, integer_axis, closed_frame, swarm, summary_errorbar,
        pair_lines, fmt_p, num, difference_spec, lighten,
    )

# Data interface unchanged.
# v2.4 presentation: closed axes (x and y meet in one corner) in all four panels;
# A's y axis ends at a labelled tick above the largest source difference.
# v2.5 presentation (notation follows the main text): A numbers the ten
# navigation sources s = 1..10 (seeds 6101..6110 in order; Delta_s, Section 2.8)
# and colours the Nav. - matched difference with the mix of the two colours
# (matched marker), as every paired difference in Figures 3 and 5; B has no
# "trained"/"extrapolation" labels (the dashed line marks the training range,
# see caption); accuracy axes read "Test accuracy" / "Short-path test accuracy"
# / "Long-path test accuracy"; path length L and fixed gain g carry their symbols.
AGGREGATE = Path(os.environ.get('CTP_AGGREGATE_DIR', Path.cwd() / 'aggregate_full')).resolve()
DATA = Path(os.environ.get('CTP_HARD8_FIGURE_DATA', AGGREGATE / 'figure_source_data' / 'FigureA5')).resolve()
FOLLOW = Path(os.environ.get('CTP_HARD8_FOLLOWUP', AGGREGATE / 'hard8x8_phaseF_followup')).resolve()
OUT = Path(os.environ.get('CTP_FIGURE_OUT', AGGREGATE / 'figures')).resolve()
OUT.mkdir(parents=True, exist_ok=True)

NAV = PALETTE['navigation']
MATCHED = PALETTE['matched']      # v2: L* 55 (was neutral_dark, L* 38 = same grey as NAV)
FRESH = PALETTE['fresh']          # v2: L* 75
BASELINE = PALETTE['relation']
ROLE_OF = {'Navigation': 'navigation', 'Matched control': 'matched', 'Fresh-random control': 'fresh'}

PRIMARY_NAV = 'navigation_trainable_scalar_gain'
PRIMARY_MATCH = 'matched_random_trainable_scalar_gain'
PRIMARY_FRESH = 'fresh_random_trainable_scalar_gain'

FIXED = {
    'navigation': {
        1: 'navigation_identity_frozen',
        3: 'navigation_fixed_gain3_frozen',
        10: 'navigation_fixed_gain10_frozen',
    },
    'matched': {
        1: 'matched_random_identity_frozen',
        3: 'matched_random_fixed_gain3_frozen',
        10: 'matched_random_fixed_gain10_frozen',
    },
}
FULL = {
    'Navigation': 'navigation_full_core_trainable',
    'Matched control': 'matched_random_full_core_trainable',
    'Fresh-random control': 'fresh_random_full_core_trainable',
}
SCALAR = {
    'Navigation': PRIMARY_NAV,
    'Matched control': PRIMARY_MATCH,
    'Fresh-random control': PRIMARY_FRESH,
}
COL = {
    'Navigation': NAV,
    'Matched control': MATCHED,
    'Fresh-random control': FRESH,
}

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


def source_vals(condition, split, col='accuracy'):
    z = metrics[(metrics.condition == condition) & (metrics.split == split)]
    a = pd.to_numeric(z[col], errors='coerce').to_numpy(float)
    return a[np.isfinite(a)]


def source_frame(condition, split, col='accuracy'):
    z = metrics[(metrics.condition == condition) & (metrics.split == split)][['source_seed', col]].copy()
    z[col] = pd.to_numeric(z[col], errors='coerce')
    return z[np.isfinite(z[col])].sort_values('source_seed').reset_index(drop=True)


def paired_source_frame(left_condition, right_condition, split, col='accuracy'):
    left = source_frame(left_condition, split, col).rename(columns={col: 'left'})
    right = source_frame(right_condition, split, col).rename(columns={col: 'right'})
    return left.merge(right, on='source_seed', how='inner').sort_values('source_seed').reset_index(drop=True)


def _mean_ci_errorbar(ax, x, values, *, color, marker='o', face=None, zorder=5):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    m, lo, hi = mean_ci95(values)
    face = color if face is None else face
    ax.errorbar(
        x, m, yerr=[[m - lo], [hi - m]], fmt=marker, ms=4.4,
        color=color, mfc=face, mec=color if face == 'white' else 'white',
        mew=.75 if face == 'white' else .4, capsize=CAPSIZE,
        elinewidth=ERROR_LINEWIDTH, zorder=zorder,
    )
    return m, lo, hi


def source_length(condition):
    z = perlen[perlen.condition == condition].copy()
    length = pd.to_numeric(z.path_length, errors='coerce')
    z = z[((z.split == 'id_seen_short') & (length <= 5)) |
          ((z.split == 'seen_long') & (length >= 6))]
    s = z.groupby(['source_seed', 'path_length'], as_index=False).accuracy.mean()
    rows = []
    for length_value, q in s.groupby('path_length'):
        a = q.accuracy.to_numpy(float)
        m, se = mean_sem(a)
        rows.append((int(length_value), m, se))
    return pd.DataFrame(rows, columns=['length', 'mean', 'sem']).sort_values('length')


def panel_replication(ax):
    """A: per-source Navigation - matched difference on short, seen paths."""
    row = primary[primary.split == 'id_seen_short'].iloc[0]
    d = json.loads(row.per_source_differences)
    seeds = sorted(d, key=lambda x: int(x))
    vals = np.array([d[x] for x in seeds], float)
    x = np.arange(len(seeds))
    # Paired difference Nav. - matched: mixed colour, matched-control marker.
    spec = difference_spec(role('navigation'), role('matched'))
    col = spec['color']
    reference_line(ax, 0)
    ax.plot([-0.4, len(seeds) - 0.6], [row.mean_difference] * 2, color=col, lw=LINEWIDTH_SECONDARY,
            ls=(0, (4, 2)), zorder=2)
    ax.scatter(x, vals, zorder=4, s=34 * spec['ms_scale'] ** 2, marker=spec['marker'], facecolor=col,
               edgecolor='white', linewidth=0.6)
    # Mean and its 95 % CI (frozen ci95_low / ci95_high) beyond the last source.
    xm = len(seeds) + 0.35
    summary_errorbar(ax, xm, spec=spec, mean=float(row.mean_difference),
                     lo=float(row.ci95_low), hi=float(row.ci95_high))
    # Sources are numbered s = 1..10 in seed order (6101 -> 1, ..., 6110 -> 10).
    ax.set_xticks(list(x) + [xm], [str(i + 1) for i in range(len(seeds))] + ['Mean'])
    ax.tick_params(axis='x', labelsize=FS_MIN)
    ax.set_xlim(-0.7, xm + 0.6)
    # Labelled range 0-0.12 encloses every source (largest 0.104); the axis
    # starts just below 0 so the near-zero sources stay clear of the x axis.
    ax.set_ylim(-0.012, 0.12)
    ax.set_yticks([0, 0.04, 0.08, 0.12])
    ax.set_yticks([0.02, 0.06, 0.10], minor=True)
    ax.set_ylabel('Seen-short accuracy difference')
    ax.set_xlabel('Navigation source $s$')
    closed_frame(ax)


def panel_horizon(ax):
    """B: accuracy as a function of path length (trained 1-5, extrapolation 6-12)."""
    specs = [
        (PRIMARY_FRESH, 'Fresh-random control', role('fresh')),
        (PRIMARY_MATCH, 'Matched control', role('matched')),
        (PRIMARY_NAV, 'Navigation', role('navigation')),
    ]
    handles = {}
    for cond, label, spec in specs:
        z = source_length(cond)
        x = z.length.to_numpy(float)
        y = z['mean'].to_numpy(float)
        se = z['sem'].to_numpy(float)
        ax.fill_between(x, y - se, y + se, color=spec['color'], alpha=BAND_ALPHA, lw=0, zorder=1)
        ax.plot(x, y, **line_kw(spec, zorder=5 if cond == PRIMARY_NAV else 3))
        handles[label] = legend_handle(spec, label)
    ax.axvline(5.5, color=PALETTE['neutral_mid'], lw=.8, ls=(0, (3, 2)), zorder=0)
    reference_line(ax, 1 / 64)
    accuracy_axis(ax, label='Test accuracy')
    integer_axis(ax, lo=1, hi=12, step=2, label='Path length $L_{\\mathrm{p}}$')
    # The dashed line at L = 5.5 separates target-training lengths (1-5) from
    # longer paths; it is explained in the caption (no in-panel labels).
    # Shorter handles keep the key text clear of the dashed L = 5.5 line.
    legend(ax, loc='lower left', bbox_to_anchor=(0.0, 0.40), handlelength=1.7,
           handles=[handles[k] for k in ('Navigation', 'Matched control', 'Fresh-random control')])
    closed_frame(ax)


def panel_gain(ax, split='id_seen_short', title=None):
    """C: short-path accuracy of frozen cores under fixed relation-input gains."""
    gains = [1, 3, 10]
    handles = []
    for origin, spec, label in [
        ('matched', role('matched'), 'Matched control'),
        ('navigation', role('navigation'), 'Navigation'),
    ]:
        means = []
        for gain in gains:
            vals = source_vals(FIXED[origin][gain], split)
            xs = gain * (1.075 if origin == 'navigation' else 1 / 1.075)   # side by side on the log axis
            jit = np.asarray(deterministic_jitter(len(vals), width=.035))
            ax.scatter(xs * 10 ** jit, vals, zorder=3, s=15 * spec.get('ms_scale', 1) ** 2,
                       marker=spec['marker'], facecolor=spec['color'], edgecolor='white',
                       linewidth=0.4, alpha=0.45)
            m, lo, hi = summary_errorbar(ax, xs, vals, spec=spec, zorder=6)
            means.append((xs, m))
        arr = np.asarray(means)
        kw = line_kw(spec, zorder=5)
        kw.update(marker='none')
        ax.plot(arr[:, 0], arr[:, 1], **kw)
        handles.append(legend_handle(spec, label))
    ax.set_xscale('log')
    ax.set_xticks(gains, ['1', '3', '10'])
    ax.minorticks_off()
    ax.set_xlim(.74, 13.5)
    accuracy_axis(ax, label='Seen-short test accuracy' if split == 'id_seen_short' else 'Long-path test accuracy')
    ax.set_xlabel('Fixed relation-input gain $g$')
    legend(ax, loc='lower right', handles=handles[::-1], bbox_to_anchor=(1.0, 0.02))
    closed_frame(ax)


def panel_transfer_regime(ax):
    """D: long-path accuracy, frozen core + learned scalar gain vs full fine-tuning."""
    origins = ['Navigation', 'Matched control', 'Fresh-random control']
    x = np.arange(3.)
    for i, origin in enumerate(origins):
        spec = role(ROLE_OF[origin])
        if origin == 'Fresh-random control':
            spec = dict(spec, color='#9EA3A8')    # a touch darker so light markers keep their edge
        pair = paired_source_frame(SCALAR[origin], FULL[origin], 'seen_long')
        left, right = i - .17, i + .17
        a = pair.left.to_numpy(float)
        b = pair.right.to_numpy(float)
        # Regime = marker (filled circle: frozen + scalar gain; open square: full
        # fine-tuning); origin = colour and position.
        scalar_spec = dict(spec, marker='o', filled=True, ms_scale=1.0)
        full_spec = dict(spec, marker='s', filled=False, ms_scale=0.92)
        off_a = swarm(ax, left, a, spec=scalar_spec, width=0.06, s=16)
        off_b = swarm(ax, right, b, spec=full_spec, width=0.06, s=16)
        pair_lines(ax, [left + off_a, right + off_b], [a, b], alpha=0.25)
        summary_errorbar(ax, left - 0.13, a, spec=scalar_spec)
        summary_errorbar(ax, right + 0.13, b, spec=full_spec)
    reference_line(ax, 1 / 64)
    categorical_xlim(ax, x, pad=.62)
    ax.set_xticks(x, ['Navigation', 'Matched\ncontrol', 'Fresh-random\ncontrol'])
    accuracy_axis(ax, label='Seen-long test accuracy', hi=1.12)
    dark = PALETTE['neutral_dark']
    handles = [
        legend_handle(dict(role('navigation'), color=dark), 'Frozen core + learned scalar gain', line=False),
        legend_handle(dict(role('navigation'), color=dark, filled=False, marker='s'),
                      'Full recurrent fine-tuning', line=False),
    ]
    legend(ax, loc='upper right', points=True, handles=handles, bbox_to_anchor=(1.0, 1.03))
    closed_frame(ax)


def make_figure6():
    apply_publication_style()
    fig = plt.figure(figsize=(FIG_WIDTH, 4.45))
    gs = fig.add_gridspec(
        2, 2, left=.085, right=.99, top=.965, bottom=.10,
        wspace=.30, hspace=.42, width_ratios=[1, 1], height_ratios=[1, 1.02],
    )
    axs = [
        fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]),
    ]
    panel_replication(axs[0])
    panel_horizon(axs[1])
    panel_gain(axs[2])
    panel_transfer_regime(axs[3])
    fig.align_ylabels([axs[0], axs[2]])
    place_panel_labels(fig, list(zip(axs, 'ABCD')))
    save_publication_figure(fig, OUT / 'Figure6')


if __name__ == '__main__':
    make_figure6()
