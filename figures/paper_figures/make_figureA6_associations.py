#!/usr/bin/env python3
"""FigureA6 — exploratory source-level grid-fraction / transfer associations."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .common import FigureContext, empty_panel, load_csv, manifest_for_figure, write_json
from .nature_style import PALETTE, place_panel_labels, annotate, apply_publication_style, save_publication_figure, style_axis
from .nature_style import FIG_WIDTH, difference_spec, role

#: Every panel is a Nav. - DR difference: colour/marker follow the v2.5
#: difference rule (mix of navigation and DR colours, DR marker), as in Fig. 3C.
NAV_MINUS_DR = difference_spec(role("navigation"), role("dr"))
SPECS = [
    ("aligned_nav_minus_dr", "Aligned endpoint task", "Δ test accuracy (Nav. − DR)", NAV_MINUS_DR["color"]),
    ("symbolic1000_nav_minus_dr", "Arbitrary symbols, 1,000 trials", "Δ test accuracy (Nav. − DR)", NAV_MINUS_DR["color"]),
    ("symbolic_low_auc_nav_minus_dr", "Arbitrary symbols, 10–200 trials", r"$\Delta\mathrm{AUC}_{\mathrm{10\!:\!200}}$ (Nav. − DR)", NAV_MINUS_DR["color"]),
]


def _regression_with_ci(ax, x, y, color):
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(float(x.min()), float(x.max()), 160)
    yy = np.polyval(coef, xx)
    ax.plot(xx, yy, color=color, lw=1.1, zorder=2)
    if len(x) > 2 and np.std(x) > 0:
        fitted = np.polyval(coef, x)
        resid = y - fitted
        s_err = np.sqrt(np.sum(resid**2) / max(1, len(x) - 2))
        xbar = np.mean(x); sxx = np.sum((x - xbar)**2)
        if sxx > 0:
            tcrit = stats.t.ppf(.975, len(x) - 2)
            se_mean = s_err * np.sqrt(1/len(x) + (xx - xbar)**2 / sxx)
            ax.fill_between(xx, yy - tcrit*se_mean, yy + tcrit*se_mean, color=color, alpha=.09, lw=0, zorder=1)


def build(ctx: FigureContext):
    apply_publication_style()
    data = ctx.source_data_dir / "FigureA6"
    out = ctx.figures_dir
    tab = load_csv(data / "grid_transfer_associations.csv")

    fig, axs = plt.subplots(1, 3, figsize=(FIG_WIDTH, 2.55))
    fig.subplots_adjust(left=.08, right=.99, top=.92, bottom=.24, wspace=.48)
    stats_rows = []
    for i, (ax, (col, title, yl, color)) in enumerate(zip(axs, SPECS)):
        if tab.empty or col not in tab or "grid_fraction" not in tab:
            empty_panel(ax); continue
        z = tab[["source_seed", "grid_fraction", col]].dropna()
        x = z.grid_fraction.to_numpy(float)          # fraction (0-1), as in Figure 2B
        y = z[col].to_numpy(float)
        if len(z) < 3:
            empty_panel(ax, "Too few complete sources"); continue
        r, p = stats.pearsonr(x, y)
        rho, ps = stats.spearmanr(x, y)
        ax.scatter(x, y, s=18, marker=NAV_MINUS_DR["marker"], color=color, edgecolor="white", linewidth=.35, alpha=.88, zorder=3)
        _regression_with_ci(ax, x, y, color)
        # v2: no panel title; the endpoint name becomes the first line of the y label.
        ax.set_xlabel(r"Grid-like unit fraction $f_{\mathrm{G}}$")
        ax.set_ylabel(f"{title}\n{yl}")
        r_lo, r_hi = np.tanh(np.arctanh(r) + np.array([-1.0, 1.0]) * 1.959964 / np.sqrt(len(x) - 3))  # Fisher 95% CI
        stat_text = fr"$r = {r:.2f}$ $[{r_lo:.2f},\ {r_hi:.2f}]$, $P = {p:.3f}$" + "\n" + fr"$r_s = {rho:.2f}$, $P = {ps:.3f}$"
        # Panel B's cloud reaches its top-left corner, so its statistics move
        # to the opposite corner.  Both corners use the same shared anchor.
        annotate(ax, stat_text, loc="upper left" if i == 0 else ("upper right" if i == 1 else "upper center"))

        ax.set_xlim(0.235, 0.335)                   # encloses every source (0.245-0.326)
        ax.set_xticks([0.24, 0.28, 0.32])
        style_axis(ax, max_y_ticks=4, max_x_ticks=4)
        stats_rows.append({
            "metric": col, "n_sources": len(z), "pearson_r": r, "pearson_p": p,
            "spearman_rho": rho, "spearman_p": ps,
        })

    place_panel_labels(fig, [(ax, "ABC"[i]) for i, ax in enumerate(axs)])

    paths = save_publication_figure(fig, out / "FigureA6")
    pd.DataFrame(stats_rows).to_csv(data / "association_statistics.csv", index=False)
    inputs = [data / "grid_transfer_associations.csv", data / "association_statistics.csv"]
    write_json(
        out / "manifests" / "FigureA6.json",
        manifest_for_figure(
            ctx, "FigureA6", Path(__file__), inputs, paths,
            notes=[
                "Exploratory/descriptive source-level association only; this figure is not used for causal inference or checkpoint selection.",
                "Regression bands show 95% confidence intervals for the fitted mean relation and do not alter the reported source-level statistics.",
                "Grid-like unit fraction is measured at each source's selected navigation checkpoint, matching the final transfer policy.",
                "All panels use the canonical grid-like unit fraction label to match Figure 2 terminology.",
                "Panel C reports the 10–200-trial log-AUC difference (Nav − DR), aligned with the Figure 5 metric naming.",
            ],
        ),
    )
    return paths
