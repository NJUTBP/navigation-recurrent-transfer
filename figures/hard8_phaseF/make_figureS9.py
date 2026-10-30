"""Supplementary Figure S11 (builder SupplementaryFigureS9): prespecified source-measure analyses.

Reads figure_source_data/SupplementaryFigureS10 (panel tables, PANEL_CONTRACT.json,
s10_statistics.json); nothing is recomputed.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .nature_style import (
        apply_publication_style, PALETTE, LINEWIDTH_SECONDARY, CAPSIZE,
        ERROR_LINEWIDTH, reference_line, style_axis, save_publication_figure,
    )
    from .mechanism_plot_common import (
        OUT, NAV, MATCHED, NEG, POS, MID, figure_data, read_csv, read_json,
        source_swarm_with_ci, annotate, categorical_xlim, place_panel_labels,
        group_brackets, num, sci, FS_INPANEL, FS_TICK_DENSE,
    )
except ImportError:
    from nature_style import (
        apply_publication_style, PALETTE, LINEWIDTH_SECONDARY, CAPSIZE,
        ERROR_LINEWIDTH, reference_line, style_axis, save_publication_figure,
    )
    from mechanism_plot_common import (
        OUT, NAV, MATCHED, NEG, POS, MID, figure_data, read_csv, read_json,
        source_swarm_with_ci, annotate, categorical_xlim, place_panel_labels,
        group_brackets, num, sci, FS_INPANEL, FS_TICK_DENSE,
    )

# The source-data folder keeps its builder name; see figures/figure_numbering.py.
DATA = figure_data("SupplementaryFigureS10")
A = read_csv(DATA / "panelA_Rret_diff_working_points.csv")
B = read_csv(DATA / "panelB_primary_per_source_slopes.csv")
C = read_csv(DATA / "panelC_expected_null_per_source_slopes.csv")
D = read_csv(DATA / "panelD_anchor_forest.csv")
STATS = read_json(DATA / "s10_statistics.json")


try:
    from .nature_style import role, difference_spec
except ImportError:
    from nature_style import role, difference_spec

NAV_MINUS_MATCHED = difference_spec(role("navigation"), role("matched"))


def panel_working_points(ax):
    """A: the navigation-minus-control retention contrast at four working points."""
    order = ["A", "g3", "g10", "Bstar"]
    labels = ["$g = 1$", "$g = 3$", "$g = 10$", r"$g^\ast$ (primary)"]
    x = np.arange(len(order), dtype=float)
    for i, wp in enumerate(order):
        vals = A[A.working_point == wp].difference.to_numpy(float)
        primary = wp == "Bstar"
        # Filled navigation-blue marks the prespecified primary working point;
        # exploratory working points stay open and neutral, as in Figure 6D.
        source_swarm_with_ci(ax, x[i], vals,
                             color=NAV_MINUS_MATCHED["color"],
                             marker=NAV_MINUS_MATCHED["marker"],
                             face=None if primary else "white",
                             jitter_width=.06, point_alpha=.30)
    reference_line(ax, 0)
    categorical_xlim(ax, x, pad=.58)
    ax.set_xticks(x, labels)
    ax.set_ylim(-.30, .40)
    ax.set_yticks([-.2, 0, .2, .4])
    ax.set_ylabel(r"$\Delta R_{\mathrm{ret}}^{\mathrm{diff}}$ (Nav. − matched)")
    style_axis(ax, max_y_ticks=5, max_x_ticks=4)


def _per_source_slopes(ax, frame, column, *, color, marker, measure, verdict_text,
                       ylim, yticks):
    """One dot per navigation source, as in Figure 6A.

    A single column of ten jittered points wastes the panel and hides which
    source is which; the source axis is the manuscript's own idiom for
    per-source replication.
    """
    z = frame[["source_seed", column]].dropna().sort_values("source_seed")
    seeds = z.source_seed.astype(int).to_numpy()
    vals = z[column].to_numpy(float)
    x = np.arange(len(seeds), dtype=float)
    reference_line(ax, 0)
    ax.axhline(float(np.mean(vals)), color=color, lw=LINEWIDTH_SECONDARY, zorder=2)
    ax.scatter(x, vals, s=23, marker=marker, color=color, edgecolor="white",
               lw=.45, zorder=3)
    ax.set_xticks(x, [str(i + 1) for i in range(len(seeds))])   # s = 1..10 in seed order
    ax.set_xlim(-.7, len(seeds) - .3)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.set_xlabel("Navigation source $s$")
    ax.set_ylabel(r"Per-source slope across $\rho$")
    annotate(ax, f"{measure}\n{verdict_text}", loc="upper right")
    style_axis(ax, max_y_ticks=5, max_x_ticks=10)


def panel_primary_slopes(ax):
    """B: per-source slopes for the prespecified primary source measure."""
    v = STATS["panelB_primary_verdict"]
    _per_source_slopes(
        ax, B, "seen_long_on_R_ret_diff_Bstar_slope", color=POS, marker="o",
        measure=r"$R_{\mathrm{ret}}^{\mathrm{diff}}$ at $g^\ast$",
        verdict_text=(f"pooled slope = {num(v['estimate'])}\n"
                      f"permutation $P \\leq$ {sci(v['p'])}"),
        ylim=(-.9, 8.6), yticks=[0, 2, 4, 6],
    )


def panel_null_slopes(ax):
    """C: per-source slopes for the expected-null control measure."""
    v = STATS["panelC_expected_null_S3_verdict"]
    _per_source_slopes(
        ax, C, "seen_long_on_S_cons_Bstar_slope", color=NEG, marker="s",
        measure=r"$S_{\mathrm{cons}}$ at $g^\ast$",
        verdict_text=(f"pooled slope = {num(v['estimate'], '{:.1f}')}\n"
                      f"Holm $P$ = {v['holm_p']:.4f}"),
        ylim=(-35, 15.5), yticks=[-30, -20, -10, 0],
    )


def panel_anchors(ax):
    """D: the six implementation anchors, grouped by contrast."""
    anchor_labels = {
        "anchor_rho0_minus_navigation_E09a": r"$\rho$ = 0 − Nav.",
        "anchor_rho0p9_minus_rho0_E6": r"$\rho$ = 0.90 − $\rho$ = 0",
        "symmetry_rho0p9_minus_rho0p1": r"$\rho$ = 0.90 − $\rho$ = 0.10",
    }
    split_labels = {"id_seen_short": "Seen short", "seen_long": "Seen long",
                    "heldout_long": "Held-out long"}

    # Rows are grouped by anchor, so the contrast name is written once per group
    # instead of being repeated in every row label.
    anchors = list(dict.fromkeys(D.anchor))
    y, rows, groups = 0.0, [], []
    for anchor in anchors:
        block = D[D.anchor == anchor]
        start = y
        for _, row in block.iterrows():
            rows.append((y, row))
            y += 1.0
        groups.append((anchor, start, y - 1.0))
        y += 0.8  # visual gap between contrast groups

    for ypos, row in rows:
        color = POS if bool(row.interval_contains_zero) else NEG
        ax.errorbar(row.difference, ypos,
                    xerr=[[row.difference - row.ci_low], [row.ci_high - row.difference]],
                    fmt="o", ms=4.2, color=color, mfc=color, mec="white", mew=.35,
                    capsize=CAPSIZE, elinewidth=ERROR_LINEWIDTH, zorder=3)
    reference_line(ax, 0, orientation="v")

    ax.set_yticks([ypos for ypos, _ in rows],
                  [split_labels.get(row.split, row.split) for _, row in rows],
                  fontsize=FS_TICK_DENSE)
    ax.set_ylim(max(yp for yp, _ in rows) + .55, -.55)  # inverted: first contrast on top
    ax.set_xlim(-.038, .060)
    ax.set_xticks([-.02, 0, .02, .04])
    ax.set_xlabel("Test accuracy difference")

    style_axis(ax, max_y_ticks=6, max_x_ticks=5)
    # Group brackets live outside the data area, in the tick-label column.
    group_brackets(ax, [(anchor_labels.get(a, a), top, bottom)
                        for a, top, bottom in groups])


def make_s9():
    apply_publication_style()
    fig = plt.figure(figsize=(7.25, 5.15))
    gs = fig.add_gridspec(2, 2, left=.085, right=.995, top=.945, bottom=.115,
                          wspace=.46, hspace=.56, width_ratios=[1, 1])
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
           fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    panel_working_points(axs[0])
    panel_primary_slopes(axs[1])
    panel_null_slopes(axs[2])
    panel_anchors(axs[3])
    fig.align_ylabels([axs[0], axs[2]])
    place_panel_labels(fig, list(zip(axs, "ABCD")))
    return save_publication_figure(fig, OUT / "SupplementaryFigureS9")


if __name__ == "__main__":
    make_s9()
