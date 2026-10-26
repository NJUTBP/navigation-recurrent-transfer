#!/usr/bin/env python3
"""Figure4 — fixed-interface perturbations and learned adapter rescue.

v2.1 presentation (data interface unchanged: test_transforms.csv, adapter.csv,
adapter_contrasts.csv, adapter_paired_differences.csv; same output stem).
Concise axis labels; no in-panel significance annotations (reported in the main text);
B and C share the ten perturbation rows (alternating bands).

v2.4 presentation: closed axes (x and y meet in one corner) in all panels,
including the categorical axis of A and the row axes of B and C; the gain axis
of C starts below 0 so the zero-gain rows stay clear of the y axis.

v2.5 presentation: terms follow the main text -- A "original" (not "intact");
B legend states what is learned or fixed (the relation-input adapter A); C is
the paired difference "Δ test accuracy (A learned − fixed)" -- "gain" is
reserved for the relation-input gain g of Figures 6-7.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import FigureContext, empty_panel, load_csv, manifest_for_figure, write_json
from .nature_style import (
    FIG_WIDTH, apply_publication_style, place_panel_labels, legend,
    legend_handle, role, reference_line, save_publication_figure, accuracy_axis,
    closed_frame, swarm, summary_errorbar, pair_lines, mean_ci95,
    deterministic_jitter,
)

PREFERRED_TEST = ["none", "rotate90", "scale2x", "signflip"]
PREFERRED_ADAPTER = [
    "rot30", "rot90", "rot135", "rot270", "signflip", "shear0.5",
    "aniso2_0.5", "scale0.5x", "scale2x", "scale4x",
]
CHANCE = 1 / 16

BAND = "#F1F3F5"

ARM_STYLE = {
    "nav_adapter": (role("navigation"), r"Nav., $\mathbf{A}$ learned"),
    "nav_noadapter": (role("navigation_open"), r"Nav., $\mathbf{A}$ fixed"),
    "random_adapter": (role("random"), r"Random, $\mathbf{A}$ learned"),
}


def _tf_label(x, *, two_line=False):
    text = {
        "none": "original", "rotate90": "rot 90°", "scale2x": "scale ×2",
        "signflip": "sign reversal", "rot30": "rot 30°", "rot90": "rot 90°",
        "rot135": "rot 135°", "rot270": "rot 270°", "shear0.5": "shear 0.5",
        "aniso2_0.5": "aniso ×(2, 0.5)", "scale0.5x": "scale ×0.5",
        "scale4x": "scale ×4",
    }.get(str(x), str(x).replace("_", " "))
    return text.replace(" ", "\n", 1) if two_line and text != "original" else text


def _fixed_interface_panel(ax, tests):
    if tests.empty:
        empty_panel(ax); return
    avail = list(dict.fromkeys(tests["transform"].astype(str)))
    tfs = [t for t in PREFERRED_TEST if t in avail] or avail[:4]
    x = np.arange(len(tfs), dtype=float)
    arms = [("random_frozen", role("random"), -0.17, "Random core"),
            ("transfer_freeze", role("navigation"), 0.17, "Navigation core")]
    wide = {arm: tests[tests.arm == arm].pivot_table(index="source_seed", columns="transform",
                                                     values="value", aggfunc="mean")
            for arm, *_ in arms}
    seeds = sorted(set(wide["random_frozen"].index) & set(wide["transfer_freeze"].index))
    jit = deterministic_jitter(len(seeds), width=0.045)

    def column(arm, t):
        w = wide[arm].reindex(seeds)
        return w[t].to_numpy(float) if t in w else np.full(len(seeds), np.nan)

    # Transformations stay categorical: thin lines pair each source's random and
    # navigation cores within a transformation, never across transformations.
    for i, t in enumerate(tfs):
        pair_lines(ax, [x[i] - 0.17 + jit, x[i] + 0.17 + jit],
                   [column("random_frozen", t), column("transfer_freeze", t)], alpha=0.28)
    handles = []
    for arm, spec, off, text in arms:
        for i, t in enumerate(tfs):
            col = column(arm, t)
            swarm(ax, x[i] + off, col, spec=spec, offsets=jit, s=16, alpha=0.5)
            summary_errorbar(ax, x[i] + off, col, spec=spec, ms=6.4)
        handles.append(legend_handle(spec, text, line=False))
    reference_line(ax, CHANCE)
    ax.set_xlim(-0.55, len(tfs) - 0.35)
    ax.set_xticks(x, [_tf_label(t, two_line=True) for t in tfs])
    accuracy_axis(ax, label="Test accuracy", hi=1.24)
    # Key starts just right of the (now full-height) y axis.
    legend(ax, loc="upper left", points=True, handles=handles[::-1], bbox_to_anchor=(0.025, 1.04),
           ncol=2, columnspacing=0.9)
    closed_frame(ax)


def _row_bands(ax, n):
    for y in range(n):
        if y % 2 == 0:
            ax.axhspan(y - 0.5, y + 0.5, color=BAND, lw=0, zorder=0)


def _adapter_panel(ax, adapter):
    if adapter.empty:
        empty_panel(ax); return []
    avail = list(dict.fromkeys(adapter["transform"].astype(str)))
    tfs = [t for t in PREFERRED_ADAPTER if t in avail] + [
        t for t in avail if t not in PREFERRED_ADAPTER and t != "none"
    ]
    n = len(tfs)
    y = np.arange(n)[::-1]
    _row_bands(ax, n)
    offsets = {"nav_adapter": .20, "nav_noadapter": 0.0, "random_adapter": -.20}
    for arm in ["random_adapter", "nav_noadapter", "nav_adapter"]:
        spec, _ = ARM_STYLE[arm]
        for yi, tf in zip(y, tfs):
            vals = adapter[(adapter.arm == arm) & (adapter["transform"].astype(str) == tf)].value.to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals):
                summary_errorbar(ax, yi + offsets[arm], vals, spec=spec, orientation="h", ms=5.4)
    reference_line(ax, CHANCE, orientation="v")
    handles = [legend_handle(ARM_STYLE[k][0], ARM_STYLE[k][1], line=False)
               for k in ("nav_adapter", "nav_noadapter", "random_adapter")]
    legend(ax, loc="center left", points=True, handles=handles, bbox_to_anchor=(0.335, 0.60),
           alignment="left", handletextpad=0.3, borderaxespad=0.0)
    ax.set_yticks(y, [_tf_label(t) for t in tfs])
    ax.set_ylim(-0.5, n - 0.5)
    accuracy_axis(ax, which="x", label="Test accuracy")
    closed_frame(ax)
    return tfs


def _gain_panel(ax, paired, contrasts, tfs):
    n = len(tfs)
    y = np.arange(n)[::-1]
    _row_bands(ax, n)
    spec = role("navigation")
    cmap = {} if contrasts.empty else {str(r.contrast): r for _, r in contrasts.iterrows()}
    for yi, tf in zip(y, tfs):
        key = "adapter_gain_" + tf
        vals = (paired[paired.contrast.astype(str) == key].difference.to_numpy(float)
                if not paired.empty else np.array([]))
        swarm(ax, yi - 0.17, vals, spec=spec, width=0.06, orientation="h", s=11, alpha=0.45)
        if key in cmap:
            r = cmap[key]
            m, lo, hi = float(r["mean"]), float(r["ci_low"]), float(r["ci_high"])
        else:
            m, lo, hi = mean_ci95(vals)
        summary_errorbar(ax, yi + 0.10, spec=spec, mean=m, lo=lo, hi=hi, orientation="h", ms=5.4)
    reference_line(ax, 0, orientation="v")
    ax.set_yticks(y, [""] * n)
    ax.set_ylim(-0.5, n - 0.5)
    # Starts below 0: the zero-gain rows (scale x2, x4) sit on the dashed zero
    # line, clear of the y axis.
    ax.set_xlim(-0.025, 0.31)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xlabel(r"Δ test accuracy ($\mathbf{A}$ learned − fixed)")
    closed_frame(ax)


def build(ctx: FigureContext):
    apply_publication_style()
    data = ctx.source_data_dir / "Figure4"
    out = ctx.figures_dir
    tests = load_csv(data / "test_transforms.csv")
    adapter = load_csv(data / "adapter.csv")
    contrasts = load_csv(data / "adapter_contrasts.csv")
    paired = load_csv(data / "adapter_paired_differences.csv")

    fig = plt.figure(figsize=(FIG_WIDTH, 2.85))
    bottom, height = 0.19, 0.68
    ax_a = fig.add_axes([0.065, bottom, 0.250, height])
    ax_b = fig.add_axes([0.435, bottom, 0.255, height])
    ax_c = fig.add_axes([0.712, bottom, 0.250, height])
    _fixed_interface_panel(ax_a, tests)
    tfs = _adapter_panel(ax_b, adapter)
    _gain_panel(ax_c, paired, contrasts, tfs or PREFERRED_ADAPTER)
    place_panel_labels(fig, [(ax_a, "A"), (ax_b, "B"), (ax_c, "C")])

    paths = save_publication_figure(fig, out / "Figure4")
    inputs = [data / "test_transforms.csv", data / "adapter.csv",
              data / "adapter_contrasts.csv", data / "adapter_paired_differences.csv"]
    write_json(
        out / "manifests" / "Figure4.json",
        manifest_for_figure(
            ctx, "Figure4", Path(__file__), inputs, paths,
            notes=[
                "Categorical transforms are shown as dot plots, never joined across transformations.",
                "A: no relearning; points = sources, thin lines pair each source's random and navigation cores within a transformation; large symbols = mean +/- 95% t-CI; dashed line = chance.",
                "B/C share the same ten perturbations (the intact baseline is in A); alternating bands mark rows.",
                "C: within-source differences, adapter A learned minus A fixed to the identity (axis 'Δ test accuracy'; 'gain' is reserved for the relation-input gain g); mean and 95% CI from adapter_contrasts.csv; significance results are reported in the main text.",
                "Layout: closed axes in all panels; C's gain axis starts at -0.025 so zero-gain rows stay clear of the y axis.",
            ],
        ),
    )
    return paths
