"""Supplementary Figure S8: multi-step dynamical signatures and the rejected one-step account.

Reads figure_source_data/SupplementaryFigureS9 (panels a-d).  The table
panelE_Hnav_autonomous_retention.csv in the same folder is not plotted.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

try:
    from .nature_style import (
        apply_publication_style, reference_line, style_axis,
        save_publication_figure,
    )
    from .mechanism_plot_common import (
        OUT, NAV, MATCHED, NEG, MID, figure_data, read_csv, read_json,
        mean_curve, source_swarm_with_ci, annotate, legend, categorical_xlim,
        place_panel_labels, num,
    )
except ImportError:
    from nature_style import (
        apply_publication_style, reference_line, style_axis,
        save_publication_figure,
    )
    from mechanism_plot_common import (
        OUT, NAV, MATCHED, NEG, MID, figure_data, read_csv, read_json,
        mean_curve, source_swarm_with_ci, annotate, legend, categorical_xlim,
        place_panel_labels, num,
    )

# The source-data folder keeps its builder name; see figures/figure_numbering.py.
DATA = figure_data("SupplementaryFigureS9")
A = read_csv(DATA / "panelA_state_maintenance.csv")
B = read_csv(DATA / "panelB_recurrence_order_drops.csv")
C = read_csv(DATA / "panelC_error_growth_by_length.csv")
D = read_csv(DATA / "panelD_delta_prediction.csv")
STATS = read_json(DATA / "s9_statistics.json")

try:
    from .nature_style import role, integer_axis
except ImportError:
    from nature_style import role, integer_axis

ORIGIN = {
    "navigation": ("Navigation", NAV, "o"),
    "matched_random": ("Matched control", MATCHED, "s"),
}


def _origin_handles():
    return [
        Line2D([0], [0], marker=marker, ls="none", ms=4.3, mfc=color,
               mec="white", mew=.35, label=label)
        for _origin, (label, color, marker) in ORIGIN.items()
    ]


def panel_state_maintenance(ax):
    """A: hidden-state norm relative to step 1."""
    for origin, (label, color, marker) in ORIGIN.items():
        mean_curve(ax, A[A.origin == origin], "path_step",
                   "hidden_norm_ratio_to_step1", spec=role("navigation" if origin == "navigation" else "matched"),
                   label=label)
    reference_line(ax, 1.0)
    integer_axis(ax, lo=1, hi=12, step=2)             # as Figure 7A
    ax.set_ylim(.40, 1.08)  # lowest source trace is 0.455
    ax.set_yticks([.45, .6, .75, .9, 1.05])
    ax.set_xlabel("Relation step $t$")
    ax.set_ylabel("Normalized hidden-state norm $n_t$")
    legend(ax, loc="lower left")
    style_axis(ax, max_y_ticks=5, max_x_ticks=5)


def panel_ablations(ax):
    """B: chance-normalized drop under recurrence removal and order shuffling."""
    modes = [("no_recurrence", "Recurrence\nablation"), ("shuffle_order", "Relation-order\nshuffling")]
    x = np.arange(len(modes), dtype=float)
    offsets = {"navigation": -.14, "matched_random": .14}
    for origin, (label, color, marker) in ORIGIN.items():
        for i, (mode, _mlabel) in enumerate(modes):
            vals = B[(B.origin == origin) & (B["mode"] == mode)].normalized_drop.to_numpy(float)
            source_swarm_with_ci(ax, x[i] + offsets[origin], vals, color=color,
                                 marker=marker, jitter_width=.035, point_alpha=.26)
    categorical_xlim(ax, x, pad=.62)
    ax.set_xticks(x, [m[1] for m in modes])
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0, .3, .6, .9])
    ax.set_ylabel("Chance-normalized accuracy drop")
    # The upper right is empty in every rendering of this contrast; the legend
    # goes there instead of floating over the centre of the panel.
    legend(ax, loc="upper right", points=True, handles=_origin_handles())
    ax.set_box_aspect(1)
    ax.set_anchor("W")
    style_axis(ax, max_y_ticks=5, max_x_ticks=2)


def panel_error_growth(ax):
    """C: magnitude of the residual error on incorrect long-path trials."""
    for origin, (label, color, marker) in ORIGIN.items():
        mean_curve(ax, C[C.origin == origin], "path_length",
                   "conditional_error_magnitude", spec=role("navigation" if origin == "navigation" else "matched"),
                   label=label)
    ax.set_xlim(5.7, 12.3)
    ax.set_xticks(range(6, 13))
    ax.set_ylim(.8, 3.8)
    ax.set_yticks([1, 2, 3])
    ax.set_xlabel("Path length $L_{\\mathrm{p}}$")
    ax.set_ylabel(r"Endpoint error $\bar{e}(L_{\mathrm{p}})$ (lattice steps)")
    legend(ax, loc="upper left")
    style_axis(ax, max_y_ticks=5, max_x_ticks=7)


def panel_delta_prediction(ax):
    """D: the one-step delta account against the observed gain shift."""
    x = D.predicted_log2_gain_shift.to_numpy(float)
    y = D.observed_log2_gain_shift.to_numpy(float)
    lo = min(np.nanmin(x), np.nanmin(y)) - .6
    hi = max(np.nanmax(x), np.nanmax(y)) + .6
    ax.plot([lo, hi], [lo, hi], color=MID, ls=(0, (3, 2)), lw=.85, zorder=0)
    ax.scatter(x, y, s=24, color=NEG, edgecolor="white", lw=.45, zorder=3)

    # The gap between the cloud and the identity line *is* the reported mean
    # error, so it is drawn once at the centroid instead of being asserted only
    # in text.
    xbar, ybar = float(np.mean(x)), float(np.mean(y))
    ax.annotate("", xy=(xbar, ybar), xytext=(xbar, xbar),
                arrowprops=dict(arrowstyle="<->", lw=.7, color=MATCHED,
                                shrinkA=0, shrinkB=0), zorder=2)
    stat = STATS["panelD_prediction_test"]

    # Equal aspect: on an identity plot the reference line must read as 45°.
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_box_aspect(1)
    ax.set_anchor("W")
    ax.set_xticks([-4, -2, 0, 2])
    ax.set_yticks([-4, -2, 0, 2])
    ax.text(hi - .35, hi - .35, "identity", ha="right", va="bottom",
            rotation=45, rotation_mode="anchor", fontsize=5.8, color=MATCHED)
    annotate(ax, f"mean error = {num(stat['mean_error_log2'])} log$_2$\n"
                 f"exact $P$ = {stat['exact_p']:.6f}", loc="lower right")
    ax.set_xlabel("Predicted log$_2$ gain shift")
    ax.set_ylabel("Observed log$_2$ gain shift")
    style_axis(ax, max_y_ticks=5, max_x_ticks=5)


def make_s8():
    apply_publication_style()
    fig = plt.figure(figsize=(7.25, 5.0))
    gs = fig.add_gridspec(2, 2, left=.085, right=.995, top=.945, bottom=.105,
                          wspace=.32, hspace=.42)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0])
    d = fig.add_subplot(gs[1, 1])
    panel_state_maintenance(a)
    panel_ablations(b)
    panel_error_growth(c)
    panel_delta_prediction(d)
    fig.align_ylabels([a, c])
    place_panel_labels(fig, [(a, "A"), (b, "B"), (c, "C"), (d, "D")])
    return save_publication_figure(fig, OUT / "SupplementaryFigureS8")


if __name__ == "__main__":
    make_s8()
