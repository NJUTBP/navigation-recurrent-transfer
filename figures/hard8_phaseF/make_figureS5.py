"""Supplementary Figure S5: controls for extended 8 × 8 transfer.

Uses the frozen Phase-F source-data interface.

v2.6: new panel E, test accuracy on held-out-composition paths versus path
length L_p for the three core types (conventions as in Figure 6B; held-out
compositions need at least two relations, so the curves begin at L_p = 2;
symbols as in panel A, so E has no key of its own).
v2.7: one three-column grid (A = B = C; the heatmap D spans the A-B width with
its row labels from A's left edge and unrotated two-line column labels; E has
exactly C's geometry).
Layout: A-C share the top row; the condition heatmap D (row labels in the left
margin) and E share the bottom row.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

try:
    from .nature_style import (
        apply_publication_style, PALETTE, LINEWIDTH,
        ERROR_LINEWIDTH, CAPSIZE, deterministic_jitter,
        mean_ci95, style_axis, place_panel_labels, legend, save_publication_figure,
    )
except ImportError:  # allow direct execution from hard8_phaseF/
    from nature_style import (
        apply_publication_style, PALETTE, LINEWIDTH,
        ERROR_LINEWIDTH, CAPSIZE, deterministic_jitter,
        mean_ci95, style_axis, place_panel_labels, legend, save_publication_figure,
    )

# Data interface unchanged.
AGGREGATE = Path(os.environ.get('CTP_AGGREGATE_DIR', Path.cwd() / 'aggregate_full')).resolve()
DATA = Path(os.environ.get('CTP_HARD8_FIGURE_DATA', AGGREGATE / 'figure_source_data' / 'FigureA5')).resolve()
FOLLOW = Path(os.environ.get('CTP_HARD8_FOLLOWUP', AGGREGATE / 'hard8x8_phaseF_followup')).resolve()
OUT = Path(os.environ.get('CTP_FIGURE_OUT', AGGREGATE / 'figures')).resolve()
OUT.mkdir(parents=True, exist_ok=True)

NAV = PALETTE['navigation']
# v2.5: same colours and dash patterns as Figure 6 (role registry).
MATCHED = PALETTE['matched']
FRESH = PALETTE['fresh']
BASELINE = PALETTE['relation']
ACC_CMAP = LinearSegmentedColormap.from_list(
    'manuscript_accuracy',
    [PALETTE['neutral_lighter'], '#B8CAD9', PALETTE['input'], NAV, PALETTE['navigation_dark']],
)

try:
    from .nature_style import DASH, accuracy_axis
except ImportError:
    from nature_style import DASH, accuracy_axis

LS = {NAV: '-', MATCHED: DASH['dash'], FRESH: DASH['dot']}

try:
    from .nature_style import (
        BAND_ALPHA, mean_sem, role, line_kw, legend_handle, reference_line,
        integer_axis, equalize_column_gaps,
    )
except ImportError:
    from nature_style import (
        BAND_ALPHA, mean_sem, role, line_kw, legend_handle, reference_line,
        integer_axis, equalize_column_gaps,
    )

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


def source_vals(condition, split, col='accuracy'):
    z = metrics[(metrics.condition == condition) & (metrics.split == split)]
    a = pd.to_numeric(z[col], errors='coerce').to_numpy(float)
    return a[np.isfinite(a)]


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


def panel_matrix_primary(ax):
    x = np.arange(4)
    specs = [
        (PRIMARY_NAV, 'Navigation', NAV, 'o', -.09),
        # v2.7: short names (as in the heatmap rows of D) keep the key clear of
        # the fresh-random points at 'Seen long'.
        (PRIMARY_MATCH, 'Matched', MATCHED, 's', 0.0),
        (PRIMARY_FRESH, 'Fresh-random', FRESH, '^', .09),
    ]
    for cond, label, color, marker, offset in specs:
        means, los, his = [], [], []
        for j, split in enumerate(SPLITS):
            vals = source_vals(cond, split)
            m, lo, hi = mean_ci95(vals)
            means.append(m)
            los.append(lo)
            his.append(hi)
            ax.scatter(
                np.full(len(vals), x[j] + offset) + deterministic_jitter(len(vals), width=.018),
                vals, s=11, color=color, alpha=.22, lw=0, zorder=2,
            )
        means = np.array(means)
        los = np.array(los)
        his = np.array(his)
        xpos = x + offset
        ax.plot(
            xpos, means, color=color, marker=marker, lw=LINEWIDTH, ms=4.2, ls=LS.get(color, '-'),
            mec='white', mew=.35, label=label, zorder=4,
        )
        ax.errorbar(
            xpos, means, yerr=[means - los, his - means], fmt='none', color=color,
            capsize=CAPSIZE, elinewidth=ERROR_LINEWIDTH, zorder=4,
        )
    ax.set_xticks(x, SLABEL)
    ax.set_ylim(0, 1.03)
    accuracy_axis(ax, label='Test accuracy')
    # v2: panel title removed (content belongs in the caption): 'Composition × horizon performance'
    legend(ax, loc='lower left')
    style_axis(ax, max_y_ticks=5, max_x_ticks=4)


def panel_gain(ax, split='seen_long', title='Fixed relation-input gain modulates long-horizon transfer'):
    gains = [1, 3, 10]
    for origin, color, marker, label in [
        ('navigation', NAV, 'o', 'Navigation'),
        ('matched', MATCHED, 's', 'Matched control'),
    ]:
        means, los, his = [], [], []
        for gain in gains:
            vals = source_vals(FIXED[origin][gain], split)
            m, lo, hi = mean_ci95(vals)
            means.append(m)
            los.append(lo)
            his.append(hi)
            ax.scatter(
                np.full(len(vals), gain) + deterministic_jitter(len(vals), width=.05 * gain),
                vals, s=13, color=color, marker=marker, alpha=.25, lw=0, zorder=2,
            )
        means = np.array(means)
        los = np.array(los)
        his = np.array(his)
        ax.plot(
            gains, means, color=color, marker=marker, lw=LINEWIDTH, ms=4.4, ls=LS.get(color, '-'),
            markeredgecolor='white', markeredgewidth=.4, label=label, zorder=4,
        )
        ax.errorbar(
            gains, means, yerr=[means - los, his - means], fmt='none', color=color,
            capsize=CAPSIZE, elinewidth=ERROR_LINEWIDTH, zorder=4,
        )
    ax.set_xscale('log')
    ax.set_xticks(gains, ['1', '3', '10'])
    ax.set_xlim(.82, 12.2)
    ax.set_ylim(0, 1.03)
    ax.set_xlabel('Fixed relation-input gain $g$')
    accuracy_axis(ax, label='Long-path test accuracy')
    # v2: panel title removed (content belongs in the caption): title
    legend(ax, loc='lower right')
    style_axis(ax, max_y_ticks=5, max_x_ticks=3)


def panel_zeroing(ax):
    z = interv[(interv.condition == PRIMARY_NAV)].copy()
    x = np.arange(4)
    left_offset = -.105
    right_offset = .105
    for j, split in enumerate(SPLITS):
        q = z[z.split == split].sort_values('source_seed')
        intact = q.real_accuracy.to_numpy(float)
        zero = q.zero_accuracy.to_numpy(float)
        jitter = deterministic_jitter(len(q), width=.018)
        for k in range(len(q)):
            ax.plot(
                [x[j] + left_offset + jitter[k], x[j] + right_offset + jitter[k]],
                [intact[k], zero[k]], color=PALETTE['neutral_mid'], lw=.55,
                alpha=.20, zorder=1,
            )
        ax.scatter(
            np.full(len(q), x[j] + left_offset) + jitter, intact, s=12,
            color=NAV, edgecolor='white', lw=.3, alpha=.32, zorder=2,
        )
        ax.scatter(
            np.full(len(q), x[j] + right_offset) + jitter, zero, s=14,
            facecolor='white', edgecolor=NAV, lw=.75, alpha=.75, zorder=2,
        )
        _mean_ci_errorbar(ax, x[j] + left_offset, intact, color=NAV, marker='o')
        _mean_ci_errorbar(ax, x[j] + right_offset, zero, color=NAV, marker='o', face='white')

    ax.set_xticks(x, SLABEL)
    ax.set_ylim(0, 1.03)
    accuracy_axis(ax, label='Test accuracy')
    # v2: panel title removed (content belongs in the caption): 'Navigation transfer depends on relation input'
    handles = [
        Line2D([0], [0], marker='o', ls='none', ms=4.5, mfc=NAV, mec='white', mew=.4,
               label='Intact'),
        Line2D([0], [0], marker='o', ls='none', ms=4.5, mfc='white', mec=NAV, mew=.8,
               label=r'$\mathbf{u}_t$ = 0'),
    ]

    # Requested: legend at the left-middle of the panel.
    legend(ax, loc='center left', bbox_to_anchor=(.015, .50), points=True, handles=handles)
    style_axis(ax, max_y_ticks=5, max_x_ticks=4)


def panel_heatmap(ax, cax):
    order = [
        'navigation_identity_frozen', 'matched_random_identity_frozen',
        'navigation_fixed_gain3_frozen', 'matched_random_fixed_gain3_frozen',
        'navigation_fixed_gain10_frozen', 'matched_random_fixed_gain10_frozen',
        'navigation_trainable_scalar_gain', 'matched_random_trainable_scalar_gain',
        'fresh_random_trainable_scalar_gain',
        'navigation_trainable_input_adapter', 'matched_random_trainable_input_adapter',
        'fresh_random_trainable_input_adapter',
        'navigation_full_core_trainable', 'matched_random_full_core_trainable',
        'fresh_random_full_core_trainable',
    ]
    labels = [
        r'Navigation, fixed $g$ = 1', r'Matched, fixed $g$ = 1',
        r'Navigation, fixed $g$ = 3', r'Matched, fixed $g$ = 3',
        r'Navigation, fixed $g$ = 10', r'Matched, fixed $g$ = 10',
        r'Navigation, learned $g$', r'Matched, learned $g$', r'Fresh-random, learned $g$',
        r'Navigation, learned $\mathbf{A}$', r'Matched, learned $\mathbf{A}$', r'Fresh-random, learned $\mathbf{A}$',
        'Navigation, full fine-tuning', 'Matched, full fine-tuning', 'Fresh-random, full fine-tuning',
    ]
    z = condmat.set_index('condition').loc[order]
    arr = z[['id_seen_short_accuracy', 'heldout_short_accuracy', 'seen_long_accuracy', 'heldout_long_accuracy']].to_numpy(float)
    im = ax.imshow(arr, aspect='auto', vmin=0, vmax=1, cmap=ACC_CMAP, interpolation='nearest')
    # v2.7: two-line column labels without rotation, as on the x axes of A and C.
    ax.set_xticks(range(4), SLABEL)
    ax.set_yticks(range(len(labels)), labels, fontsize=6.0)
    ax.tick_params(axis='x', labelsize=6.1, length=0, pad=3)
    ax.tick_params(axis='y', length=0, pad=2)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            value = arr[i, j]
            ax.text(
                j, i, f'{value:.2f}', ha='center', va='center', fontsize=6.0,
                color='white' if value >= .58 else PALETTE['black'],
            )
    for y in [1.5, 3.5]:
        ax.axhline(y, color='white', lw=.6, alpha=.65)
    for y in [5.5, 8.5, 11.5]:
        ax.axhline(y, color='white', lw=1.3, alpha=.95)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # v2: panel title removed (content belongs in the caption): 'Performance across transfer conditions'
    cb = ax.figure.colorbar(im, cax=cax, ticks=[0, .5, 1])   # v2.7: explicit colour-bar axes
    cb.set_label('Test accuracy', fontsize=6.3)
    cb.ax.tick_params(length=0, labelsize=6.0, pad=2)
    cb.outline.set_linewidth(.55)


def source_length_heldout(condition):
    """Per-length source means for held-out compositions (L_p = 2-5 and 6-12)."""
    z = perlen[perlen.condition == condition].copy()
    length = pd.to_numeric(z.path_length, errors='coerce')
    z = z[((z.split == 'heldout_short') & (length <= 5)) |
          ((z.split == 'heldout_long') & (length >= 6))]
    s = z.groupby(['source_seed', 'path_length'], as_index=False).accuracy.mean()
    rows = []
    for length_value, q in s.groupby('path_length'):
        m, se = mean_sem(q.accuracy.to_numpy(float))
        rows.append((int(length_value), m, se))
    return pd.DataFrame(rows, columns=['length', 'mean', 'sem']).sort_values('length')


def panel_heldout_horizon(ax):
    """E: held-out-composition accuracy versus path length (as Figure 6B)."""
    specs = [
        (PRIMARY_FRESH, 'Fresh-random control', role('fresh')),
        (PRIMARY_MATCH, 'Matched control', role('matched')),
        (PRIMARY_NAV, 'Navigation', role('navigation')),
    ]
    handles = {}
    for cond, label, spec in specs:
        z = source_length_heldout(cond)
        x = z.length.to_numpy(float)
        y = z['mean'].to_numpy(float)
        se = z['sem'].to_numpy(float)
        ax.fill_between(x, y - se, y + se, color=spec['color'], alpha=BAND_ALPHA, lw=0, zorder=1)
        ax.plot(x, y, **line_kw(spec, zorder=5 if cond == PRIMARY_NAV else 3))
        handles[label] = legend_handle(spec, label)
    # Same conventions as Figure 6B: dashed line between the target-training
    # lengths (1-5) and longer paths, chance level 1/64.
    ax.axvline(5.5, color=PALETTE['neutral_mid'], lw=.8, ls=(0, (3, 2)), zorder=0)
    reference_line(ax, 1 / 64)
    accuracy_axis(ax, label='Held-out test accuracy')
    integer_axis(ax, lo=1, hi=12, step=2, label=r'Path length $L_{\mathrm{p}}$')
    # No separate key: colours, markers and dash patterns are those of panel A
    # (stated in the caption); in this narrow panel an in-axes key covers data.
    style_axis(ax, max_y_ticks=5, max_x_ticks=12)


def _align_bottom_row(fig, a, c, d, cax, e, gap_in, iterations=3):
    """E takes C's horizontal geometry; D's row labels start at A's left ink and
    its colour bar ends one top-row gap before E's left ink."""
    width_in = fig.get_figwidth()
    pc, pe = c.get_position(), e.get_position()
    e.set_position([pc.x0, pe.y0, pc.width, pe.height])
    for _ in range(iterations):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        px = fig.bbox.width
        a_ink = a.get_tightbbox(r).x0 / px
        e_ink = e.get_tightbbox(r).x0 / px
        pd_, pcb = d.get_position(), cax.get_position()
        label_w = pd_.x0 - d.get_tightbbox(r).x0 / px      # heatmap row labels
        cb_right = cax.get_tightbbox(r).x1 / px - pcb.x1    # colour-bar ticks and label
        cb_x0 = e_ink - gap_in / width_in - cb_right - pcb.width
        d_x0 = a_ink + label_w
        d.set_position([d_x0, pd_.y0, cb_x0 - 0.016 - d_x0, pd_.height])
        cax.set_position([cb_x0, pcb.y0, pcb.width, pcb.height])


def make_s5():
    apply_publication_style()
    # v2.7: one three-column grid. A, B and C are identical; the heatmap D (with
    # its colour bar) spans the A-B width and E has exactly C's geometry.
    fig = plt.figure(figsize=(7.0, 5.5))
    top = fig.add_gridspec(1, 3, left=.08, right=.985, top=.958, bottom=.655, wspace=.40)
    a = fig.add_subplot(top[0])
    b = fig.add_subplot(top[1])
    c = fig.add_subplot(top[2])
    y0, h = .108, .385
    d = fig.add_axes([.30, y0, .30, h])
    cax = fig.add_axes([.62, y0, .011, h])
    e = fig.add_axes([.70, y0, .25, h])
    panel_matrix_primary(a)
    panel_gain(b)
    panel_zeroing(c)
    panel_heatmap(d, cax)
    panel_heldout_horizon(e)
    gap_in = equalize_column_gaps(fig, [a, b, c])
    _align_bottom_row(fig, a, c, d, cax, e, gap_in)
    place_panel_labels(fig, list(zip([a, b, c, d, e], 'ABCDE')))
    save_publication_figure(fig, OUT / 'SupplementaryFigureS5')

if __name__ == '__main__':
    make_s5()
