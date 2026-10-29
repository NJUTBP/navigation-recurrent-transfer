"""Supplementary Figure S7: optimization, gain and additive-solution controls.

Reads figure_source_data/SupplementaryFigureS8 (panel tables, PANEL_CONTRACT.json,
s8_statistics.json); nothing is recomputed.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter

try:
    from .nature_style import (
        apply_publication_style, LINEWIDTH, reference_line, style_axis,
        save_publication_figure,
    )
    from .mechanism_plot_common import (
        OUT, NAV, MATCHED, REL, MID, figure_data, read_csv, read_json,
        mean_curve, source_swarm_with_ci, paired_source_lines, annotate, legend,
        categorical_xlim, place_panel_labels, num, FS_INPANEL, MS_MEAN,
    )
except ImportError:
    from nature_style import (
        apply_publication_style, LINEWIDTH, reference_line, style_axis,
        save_publication_figure,
    )
    from mechanism_plot_common import (
        OUT, NAV, MATCHED, REL, MID, figure_data, read_csv, read_json,
        mean_curve, source_swarm_with_ci, paired_source_lines, annotate, legend,
        categorical_xlim, place_panel_labels, num, FS_INPANEL, MS_MEAN,
    )

# The source-data folder keeps its builder name; see figures/figure_numbering.py.
DATA = figure_data("SupplementaryFigureS8")
A = read_csv(DATA / "panelA_extended_learning_curves.csv")
B = read_csv(DATA / "panelB_32k_source_differences.csv")
C = read_csv(DATA / "panelC_gain_operating_curve.csv")
D = read_csv(DATA / "panelD_additive_horizon.csv")
STATS = read_json(DATA / "s8_statistics.json")

try:
    from .nature_style import role, line_kw, accuracy_axis, integer_axis, difference_spec
except ImportError:
    from nature_style import role, line_kw, accuracy_axis, integer_axis, difference_spec

#: Paired difference Nav. - matched (difference rule: mixed colour, matched marker).
NAV_MINUS_MATCHED = difference_spec(role("navigation"), role("matched"))

ORIGIN = [
    ("navigation", "Navigation", NAV, "o"),
    ("matched_random", "Matched control", MATCHED, "s"),
]
ACC_TICKS = [0, .3, .6, .9]


def panel_learning(ax):
    """A: validation accuracy over the extended optimization budget."""
    # The budget is sampled at 160 points per source: markers are thinned so the
    # curve reads as a trajectory instead of a chain of symbols.
    n_updates = A["update"].nunique()
    every = max(1, int(round(n_updates / 14)))
    for origin, label, color, marker in ORIGIN:
        # Matched-control training curve as a light solid line (Figs 3A/5A).
        mean_curve(ax, A[A.origin == origin], "update", "validation_accuracy",
                   spec=role("navigation" if origin == "navigation" else "matched"),
                   label=label, markevery=every, light=origin != "navigation")
    ax.set_xlim(0, 32800)
    ax.set_xticks([0, 8000, 16000, 24000, 32000])
    accuracy_axis(ax, label="Validation accuracy")
    ax.set_xlabel("Gradient updates")
    legend(ax, loc="lower right")
    style_axis(ax, max_y_ticks=5, max_x_ticks=5)


def panel_final_differences(ax):
    """B: source-level navigation advantage at the end of the extended budget."""
    order = [("id_seen_short", "Short paths\n($L_{\\mathrm{p}}$ = 1–5)"), ("seen_long", "Long paths\n($L_{\\mathrm{p}}$ = 6–12)")]
    x = np.arange(len(order), dtype=float)
    wide = B.pivot_table(index="source_seed", columns="split", values="difference")
    if all(split in wide.columns for split, _ in order):
        paired_source_lines(ax, x, wide[[order[0][0], order[1][0]]].to_numpy(float))
    for i, (split, _label) in enumerate(order):
        vals = B[B.split == split].difference.to_numpy(float)
        source_swarm_with_ci(ax, x[i], vals, color=NAV_MINUS_MATCHED["color"],
                             marker=NAV_MINUS_MATCHED["marker"], jitter_width=.055)
    reference_line(ax, 0)
    categorical_xlim(ax, x, pad=.62)
    ax.set_xticks(x, [label for _s, label in order])
    ax.set_ylim(-.06, .66)
    ax.set_yticks([0, .2, .4, .6])
    ax.set_ylabel("Δ test accuracy\n(Nav. − matched)")
    style_axis(ax, max_y_ticks=5, max_x_ticks=2)


def panel_gain(ax):
    """C: long-path accuracy across the fixed relation-input gain sweep."""
    for origin, label, color, marker in ORIGIN:
        mean_curve(ax, C[C.origin == origin], "gain", "accuracy",
                   spec=role("navigation" if origin == "navigation" else "matched"), label=label)
    ax.set_xscale("log", base=2)
    gains = sorted(float(g) for g in C.gain.unique())
    ax.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
    # Every sampled gain is marked; only the powers of four are labelled.
    ax.xaxis.set_minor_locator(FixedLocator([g for g in gains if g not in (1, 4, 16, 64, 256)]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(.8, 320)
    accuracy_axis(ax, label="Long-path test accuracy")
    ax.set_xlabel("Fixed relation-input gain $g$")
    env = STATS["panelC_seen_long_envelope"]
    shift = STATS["panelC_log2_gstar_matched_minus_navigation_mean"]
    annotate(ax, f"best-gain gap = {num(env['mean_difference'], '{:.3f}')}\n"
                 f"matched $g_{{\\mathrm{{opt}}}}$ shift = {num(shift)} log$_2$ units ($\\approx${2 ** float(shift):.1f}×)", loc="lower left")
    legend(ax, loc="lower right")
    style_axis(ax, max_y_ticks=5)


def panel_additive(ax):
    """D: an explicit additive solution against both cores."""
    # Role styles shared with Figures 5A/6B (2D accumulator = purple plus).
    specs = [
        ("linear_accumulator_2d_accuracy", "2D accumulator", role("accumulator"), 2),
        ("matched_accuracy", "Matched control", role("matched"), 3),
        ("navigation_accuracy", "Navigation", role("navigation"), 4),
    ]
    for col, label, spec, zorder in specs:
        ax.plot(D.path_length, D[col], **line_kw(spec, label=label, zorder=zorder))
    # Dashed line: target-training lengths 1-5 vs longer paths (see caption).
    ax.axvline(5.5, color=MID, lw=.8, ls=(0, (3, 2)), zorder=0)
    accuracy_axis(ax, label="Test accuracy")
    integer_axis(ax, lo=1, hi=12, step=2, label="Path length $L_{\\mathrm{p}}$")
    # Ordered as drawn, so the legend keys match the visual stacking.
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(n) for n in ["Navigation", "Matched control", "2D accumulator"]]
    legend(ax, loc="lower left", handles=[handles[i] for i in order])
    style_axis(ax, max_y_ticks=5, max_x_ticks=12)


def make_s7():
    apply_publication_style()
    fig = plt.figure(figsize=(7.25, 5.1))
    gs = fig.add_gridspec(2, 2, left=.085, right=.995, top=.945, bottom=.105,
                          wspace=.34, hspace=.40)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
           fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    panel_learning(axs[0])
    panel_final_differences(axs[1])
    panel_gain(axs[2])
    panel_additive(axs[3])
    fig.align_ylabels([axs[0], axs[2]])
    fig.align_ylabels([axs[1], axs[3]])
    place_panel_labels(fig, list(zip(axs, "ABCD")))
    return save_publication_figure(fig, OUT / "SupplementaryFigureS7")


if __name__ == "__main__":
    make_s7()
