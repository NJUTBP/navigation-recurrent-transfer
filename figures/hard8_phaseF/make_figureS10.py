"""Supplementary Figure S9 (builder SupplementaryFigureS10): refitting of the navigation-specific mappings.

Reads figure_source_data/SupplementaryFigureS11.  Panel b shows the stored means and confidence
intervals of s11_statistics.json; nothing is recomputed.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .nature_style import (
        apply_publication_style, LINEWIDTH, CAPSIZE, ERROR_LINEWIDTH,
        reference_line, style_axis, save_publication_figure,
    )
    from .mechanism_plot_common import (
        OUT, NAV, REL, NEG, POS, figure_data, read_csv, read_json, mean_curve,
        annotate, legend, categorical_xlim, place_panel_labels,
        deterministic_jitter_safe, MS_SUMMARY, MS_SMALL, SWARM_SIZE,
    )
except ImportError:
    from nature_style import (
        apply_publication_style, LINEWIDTH, CAPSIZE, ERROR_LINEWIDTH,
        reference_line, style_axis, save_publication_figure,
    )
    from mechanism_plot_common import (
        OUT, NAV, REL, NEG, POS, figure_data, read_csv, read_json, mean_curve,
        annotate, legend, categorical_xlim, place_panel_labels,
        deterministic_jitter_safe, MS_SUMMARY, MS_SMALL, SWARM_SIZE,
    )

# The source-data folder keeps its builder name; see figures/figure_numbering.py.
DATA = figure_data("SupplementaryFigureS11")
A = read_csv(DATA / "panelA_transfer_vs_navigation_competence.csv")
B = read_csv(DATA / "panelB_excess_transfer_drop.csv")
C = read_csv(DATA / "panelC_convergence_gate.csv")
STATS = read_json(DATA / "s11_statistics.json")

RHO_TICKS = [0, .1, .2, .35, .5]
RHO_LIM = (-.025, .525)
RHO_LABEL = r"Perturbation fraction $\rho$"
RHO_TICKLABELS = ["0", "0.1", "0.2", "0.35", "0.5"]   # as Figure 7B/C


try:
    from .nature_style import role, difference_spec, summary_errorbar, accuracy_axis, DASH, LINEWIDTH
    from .mechanism_plot_common import LONG
except ImportError:
    from nature_style import role, difference_spec, summary_errorbar, accuracy_axis, DASH, LINEWIDTH
    from mechanism_plot_common import LONG

#: Long-path transfer accuracy styled as in Figure 7B; the excess drop
#: (transfer drop - navigation drop) follows the difference rule.
LONG_SPEC = dict(color=LONG, marker="s", ls=DASH["dash"], filled=True, lw=LINEWIDTH, ms_scale=0.92)
EXCESS_SPEC = difference_spec(LONG_SPEC, role("navigation"))


def panel_performance(ax):
    """A: transfer performance against refitted navigation performance."""
    for col, label, spec in [
        ("transfer_seen_long", "Long-path test accuracy", LONG_SPEC),
        ("nav_frac_loss_captured", "Refitted navigation performance", role("navigation")),
    ]:
        q = A[["source_seed", "rho", col]].rename(columns={col: "value"})
        mean_curve(ax, q, "rho", "value", spec=spec, label=label)
    ax.set_xlim(*RHO_LIM)
    ax.set_xticks(RHO_TICKS, RHO_TICKLABELS)
    accuracy_axis(ax, label="Performance")
    ax.set_xlabel(RHO_LABEL)
    legend(ax, loc="lower left")
    style_axis(ax, max_y_ticks=5, max_x_ticks=5)


def panel_excess(ax):
    """B: source-level excess transfer drop at the two low doses."""
    tests = {round(float(t["rho"]), 2): t for t in STATS["low_rho_tests"]}
    x = np.arange(2, dtype=float)
    for i, rho in enumerate([0.05, 0.10]):
        vals = B[np.isclose(B.rho, rho)].excess_transfer_drop.to_numpy(float)
        jitter = deterministic_jitter_safe(len(vals), width=.05)
        ax.scatter(np.full(len(vals), x[i]) + jitter, vals, s=SWARM_SIZE,
                   color=EXCESS_SPEC["color"], edgecolor="white", lw=.35, alpha=.32, zorder=2)
        # Frozen mean and confidence interval, not recomputed from the points.
        stat = tests[round(rho, 2)]
        m, lo, hi = stat["excess_transfer_drop"], stat["ci_low"], stat["ci_high"]
        summary_errorbar(ax, x[i], spec=EXCESS_SPEC, mean=m, lo=lo, hi=hi, zorder=4)
    reference_line(ax, 0)
    categorical_xlim(ax, x, pad=.62)
    ax.set_xticks(x, [r"0.05", r"0.10"])
    ax.set_ylim(-.03, .40)
    ax.set_yticks([0, .1, .2, .3])
    ax.set_xlabel(RHO_LABEL)
    ax.set_ylabel("Excess transfer drop\n(transfer − navigation)")
    #annotate(ax, "10/10 sources > 0\nat both doses", loc="upper left")
    style_axis(ax, max_y_ticks=5, max_x_ticks=2)


def panel_convergence(ax):
    """C: refit convergence against the prespecified limit."""
    x = C.rho.to_numpy(float)
    y = C.fraction_best_in_last_eval.to_numpy(float)
    allowed = float(C.allowed_fraction.iloc[0])
    ax.plot(x, y, color=NEG, marker="s", ms=MS_SMALL, lw=LINEWIDTH,
            markeredgecolor="white", markeredgewidth=.35, label="Observed", zorder=3)
    ax.axhline(allowed, color=POS, ls=(0, (3, 2)), lw=.9, label="Prespecified limit",
               zorder=2)
    ax.set_xlim(*RHO_LIM)
    ax.set_xticks(RHO_TICKS, RHO_TICKLABELS)
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, .3, .6, .9])
    ax.set_xlabel(RHO_LABEL)
    ax.set_ylabel("Fraction best in final window")
    # The band between the observed line and the limit is empty by construction,
    # so the verdict and the key both sit there instead of over the data.
    #annotate(ax, "formal verdict:\n" + STATS["formal_verdict"], loc="center left",
             #y=.58, color=NEG)
    legend(ax, loc="lower right")
    style_axis(ax, max_y_ticks=5, max_x_ticks=5)


def make_s10():
    apply_publication_style()
    fig = plt.figure(figsize=(7.25, 2.85))
    gs = fig.add_gridspec(1, 3, left=.075, right=.995, top=.90, bottom=.205,
                          wspace=.38)
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]
    panel_performance(axs[0])
    panel_excess(axs[1])
    panel_convergence(axs[2])
    place_panel_labels(fig, list(zip(axs, "ABC")))
    return save_publication_figure(fig, OUT / "SupplementaryFigureS10")


if __name__ == "__main__":
    make_s10()
