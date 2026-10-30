#!/usr/bin/env python3
"""Figure 5: arbitrary relation symbols, finite-budget learning and recurrence use.

Reads figure_source_data/Figure5 (symbolic, contrasts, paired_differences, recurrence).  a learning
curves over target training trials (comparison curves as light solid lines); b and c paired
differences "Nav. - comparator" at 1,000 trials (b, test-accuracy difference) and as Delta AUC_10:200 (c),
coloured like the same contrasts in Figure 3c; d recurrence ablation as a paired dot plot.  Every
comparator keeps the colour and marker of its curve in a.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import FigureContext, empty_panel, load_csv, manifest_for_figure, write_json
from .nature_style import (
    FIG_WIDTH, PALETTE, BAND_ALPHA, apply_publication_style, place_panel_labels,
    legend, legend_handle, line_kw, role, reference_line, save_publication_figure,
    accuracy_axis, closed_frame, swarm, summary_errorbar, pair_lines,
    categorical_xlim, mean_sem, mean_ci95, equalize_column_gaps,
    light_line_kw, light_legend_handle, difference_spec,
)

ARMS = [
    "nav_frozen",
    "ctl01_frozen",
    "random_frozen",
    "baseline_linear_accumulator",
    "baseline_action_count_linear",
]
REC = ["nav_frozen", "ctl01_frozen", "random_frozen"]
CHANCE = 1 / 16

ARM_STYLE = {
    "nav_frozen": (role("navigation"), "Navigation"),
    "ctl01_frozen": (role("dr"), "DR"),
    "random_frozen": (role("random"), "Random"),
    "baseline_linear_accumulator": (role("accumulator"), "2D accumulator"),
    "baseline_action_count_linear": (role("relation_count"), "Relation-count"),
}
#: Rows of B/C: navigation minus comparator.  Colour = mix of navigation and
#: comparator colours, marker = comparator's (difference_spec), as in Figure 3C.
COMPARATORS = [
    ("nav__minus__ctl01_frozen", "Nav. − DR", "ctl01_frozen"),
    ("nav__minus__random_frozen", "Nav. − random", "random_frozen"),
    ("nav__minus__baseline_action_count_linear", "Nav. − relation-count", "baseline_action_count_linear"),
    ("nav__minus__baseline_linear_accumulator", "Nav. − 2D accumulator", "baseline_linear_accumulator"),
]


def _panel_learning(ax, symbolic):
    acc = symbolic[symbolic.metric == "accuracy"] if not symbolic.empty else symbolic
    if acc.empty:
        empty_panel(ax); return
    handles = []
    for arm in ARMS[::-1]:          # navigation drawn last = on top
        g = acc[acc.arm.astype(str) == arm]
        if g.empty:
            continue
        spec, text = ARM_STYLE[arm]
        rows = sorted((float(b), *mean_sem(q.value.to_numpy(float))) for b, q in g.groupby("budget"))
        arr = np.asarray(rows, dtype=float)
        ax.fill_between(arr[:, 0], arr[:, 1] - arr[:, 2], arr[:, 1] + arr[:, 2],
                        color=spec["color"], alpha=BAND_ALPHA, lw=0, zorder=1)
        hero = arm == "nav_frozen"
        # Navigation = full-colour solid line; comparators = light solid lines.
        ax.plot(arr[:, 0], arr[:, 1], **(line_kw if hero else light_line_kw)(spec, zorder=5 if hero else 3))
        handles.append((arm, legend_handle(spec, text) if hero else light_legend_handle(spec, text)))
    handles = [h for key in ARMS for a, h in handles if a == key]
    reference_line(ax, CHANCE)
    ax.set_xscale("log")
    ax.set_xlim(8.0, 9000)
    ax.set_xticks([10, 100, 1000, 7000], ["10", "100", "1,000", "7,000"])
    ax.minorticks_off()
    accuracy_axis(ax, label="Test accuracy")
    ax.set_xlabel("Target training trials")
    # Key starts just right of the (now full-height) y axis.
    legend(ax, loc="upper left", handles=handles, bbox_to_anchor=(0.025, 1.03), handlelength=1.8)
    closed_frame(ax)


def _panel_forest(ax, paired, contrasts, family, *, xlabel, xlim, xticks, xminor=()):
    q = paired[paired.family == family] if not paired.empty else paired
    if q.empty:
        empty_panel(ax); return
    cq = contrasts[contrasts.family == family] if not contrasts.empty else contrasts
    n = len(COMPARATORS)
    for i, (key, _text, arm) in enumerate(COMPARATORS):
        y = n - 1 - i
        spec = difference_spec(ARM_STYLE["nav_frozen"][0], ARM_STYLE[arm][0])
        vals = q[q.contrast.astype(str) == key].difference.to_numpy(float)
        swarm(ax, y - 0.24, vals, spec=spec, width=0.07, orientation="h", s=14)
        row = cq[cq.contrast.astype(str) == key]
        if not row.empty:
            r = row.iloc[0]
            m, lo, hi = float(r["mean"]), float(r["ci_low"]), float(r["ci_high"])
        else:
            m, lo, hi = mean_ci95(vals)
        summary_errorbar(ax, y + 0.10, spec=spec, mean=m, lo=lo, hi=hi, orientation="h", ms=5.6)
    reference_line(ax, 0, orientation="v")
    ax.set_yticks(range(n), [c[1] for c in COMPARATORS][::-1])
    ax.set_ylim(-0.60, n - 0.40)
    ax.set_xlim(*xlim)
    ax.set_xticks(xticks)
    ax.set_xticks(list(xminor), minor=True)
    ax.set_xlabel(xlabel)
    closed_frame(ax)


def _panel_recurrence(ax, recurrence: pd.DataFrame) -> None:
    if recurrence.empty:
        empty_panel(ax); return
    cats = [
        ("nav_frozen", "Navigation", role("navigation")),
        ("ctl01_frozen", "DR", role("dr")),
        ("random_frozen", "Random", role("random")),
    ]
    x = np.arange(len(cats), dtype=float)
    for i, (arm, _text, spec) in enumerate(cats):
        wide = recurrence[recurrence.arm.astype(str) == arm].pivot_table(
            index="source_seed", columns="state", values="value", aggfunc="mean").sort_index()
        if not {"intact", "no_recurrence"}.issubset(wide.columns):
            continue
        a = wide["intact"].to_numpy(float)
        b = wide["no_recurrence"].to_numpy(float)
        intact_spec = dict(spec, filled=True)       # filled = intact
        zero_spec = dict(spec, filled=False)        # open = W_rec set to zero
        left, right = i - 0.17, i + 0.17
        off_a = swarm(ax, left, a, spec=intact_spec, width=0.06, s=16)
        off_b = swarm(ax, right, b, spec=zero_spec, width=0.06, s=16)
        pair_lines(ax, [left + off_a, right + off_b], [a, b], alpha=0.25)
        summary_errorbar(ax, left - 0.13, a, spec=intact_spec)
        summary_errorbar(ax, right + 0.13, b, spec=zero_spec)
    reference_line(ax, CHANCE)
    categorical_xlim(ax, x, pad=0.62)
    ax.set_xticks(x, [c[1] for c in cats])
    accuracy_axis(ax, label="Test accuracy")
    dark = PALETTE["neutral_dark"]
    handles = [
        legend_handle(dict(role("navigation"), color=dark), "Intact", line=False),
        legend_handle(dict(role("navigation"), color=dark, filled=False), r"$\mathbf{W}_{\mathrm{rec}}$ = 0", line=False),
    ]
    legend(ax, loc="upper right", points=True, handles=handles, bbox_to_anchor=(1.02, 1.03))
    closed_frame(ax)


def build(ctx: FigureContext):
    apply_publication_style()
    data = ctx.source_data_dir / "Figure5"
    out = ctx.figures_dir
    symbolic = load_csv(data / "symbolic.csv")
    contrasts = load_csv(data / "contrasts.csv")
    paired = load_csv(data / "paired_differences.csv")
    recurrence = load_csv(data / "recurrence.csv")

    W, H = FIG_WIDTH, 2.65
    fig = plt.figure(figsize=(W, H))
    add = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])   # inches -> fraction
    # B/C row names are "Nav. − comparator", so A, B/C and D are a
    # little narrower to keep the figure at FIG_WIDTH.
    ax_a = add(0.45, 0.40, 1.62, 2.07)
    ax_b = add(3.20, 1.73, 1.10, 0.74)     # B over C: two independent panels
    ax_c = add(3.20, 0.40, 1.10, 0.74)
    ax_d = add(5.12, 0.40, 1.70, 2.07)

    _panel_learning(ax_a, symbolic)
    # Labelled ends enclose every source-level difference (B: -0.15 to 0.77,
    # C: -0.057 to 0.085); the zero reference line stays clear of the y axis.
    _panel_forest(ax_b, paired, contrasts, "F4_accuracy_1000",
                  xlabel=r"$\Delta\mathrm{Acc}_{\mathrm{test}}$", xlim=(-0.4, 0.88),
                  xticks=[-0.4, 0, 0.4, 0.8], xminor=[-0.2, 0.2, 0.6])
    _panel_forest(ax_c, paired, contrasts, "F4_logAUC_10_200",
                  xlabel=r"$\Delta\mathrm{AUC}_{\mathrm{10\!:\!200}}$", xlim=(-0.1, 0.1),
                  xticks=[-0.1, 0, 0.1], xminor=[-0.05, 0.05])
    _panel_recurrence(ax_d, recurrence)
    # Identical white space between the three columns (ink to ink); B and C
    # form the middle column and move together.
    equalize_column_gaps(fig, [ax_a, ax_b, ax_d], partners={ax_b: [ax_c]})
    place_panel_labels(fig, [(ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D")])

    paths = save_publication_figure(fig, out / "Figure5")
    inputs = [data / "symbolic.csv", data / "contrasts.csv",
              data / "paired_differences.csv", data / "recurrence.csv"]
    write_json(
        out / "manifests" / "Figure5.json",
        manifest_for_figure(
            ctx, "Figure5", Path(__file__), inputs, paths,
            notes=[
                "A: source-level mean +/- SEM at every budget present in symbolic.csv (no interpolation); dashed line = chance.",
                "B/C: Delta = navigation - comparator (row 'Nav. - X'); B = test accuracy at 1,000 trials, C = normalized log-budget AUC over 10-200 trials (AUC_10:200, Supplementary Methods S5.4); points = sources; large symbol and bar = mean and 95% t-CI from contrasts.csv; significance results are reported in the main text. Row colour = equal sRGB mix of navigation and comparator colours, marker = comparator's (same rule as Figure 3C).",
                "D: test accuracy at 1,000 trials; filled = intact, open = W_rec set to zero at evaluation; grey lines pair each source.",
                "Layout: closed axes; A, B/C and D are separated by identical ink-to-ink white space (equalize_column_gaps).",
            ],
        ),
    )
    return paths
