#!/usr/bin/env python3
"""Figure 3: frozen-core transfer on the aligned 4 x 4 endpoint task.

Reads figure_source_data/Figure3 (training_curves, frozen_transfer, contrasts, paired_differences).
a learning curves (comparison curves as light solid lines); b final test accuracy per condition
with the condition means; c paired differences A - B, drawn in the mix of the colours of A and B
with the marker of B (nature_style.difference_spec).  Significance brackets are derived from the
stored contrast table.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import FigureContext, empty_panel, load_csv, manifest_for_figure, write_json
from .nature_style import (
    FIG_WIDTH, apply_publication_style, place_panel_labels, legend, legend_handle,
    line_kw, role, reference_line, save_publication_figure, accuracy_axis, closed_frame,
    swarm, summary_errorbar, pair_lines, estimation_row, equalize_column_gaps,
    significance_bracket, categorical_xlim, mean_sem, stars, BAND_ALPHA,
    light_line_kw, light_legend_handle, difference_spec, halo, FS_ANNOT, PALETTE,
)
from matplotlib.ticker import NullFormatter

TRAINABLE = ["random_init", "transfer_finetune", "transfer_input_only", "transfer_recurrent_only"]
CHANCE = 1 / 16
#: Target-epoch axis of A.  "log": 5-100 with 1-2-5 ticks (keeps the early-epoch
#: separation between arms legible; default).  "linear": 0-100, ticks every 20.
#: Either way the axis ends at a labelled tick that includes the last epoch.
EPOCH_AXIS = "log"
#: Frozen P value behind the bracket asterisks (contrasts.csv also has p_t_holm / p_sign_holm).
P_COLUMN = "p_sign_flip_holm"
#: Mean labels in B use the precision of the main text (0.9985, 0.2810, 0.1899).
MEAN_LABEL_FMT = "{:.4f}"

INPUT_ONLY_COLOR = "#4E9A55"   # L* 57
CURVE_STYLE = {
    #"transfer_finetune": (role("navigation_dark"), r"Nav. $W_{in}$ + $W_{rec}$"),
    #"transfer_input_only": (role("input"), r"Nav. $W_{in}$ only"),
    #"transfer_recurrent_only": (role("recurrent"), r"Nav. $W_{rec}$ only"),
    "transfer_finetune": (role("navigation_dark"), r"Nav. $\mathbf{W}_{\mathrm{in}}$ + $\mathbf{W}_{\mathrm{rec}}$"),
    # Final revision: green instead of the light teal "input" tint, so that the three
    # navigation-initialised arms differ in hue (navy / teal / green), not only in lightness.
    "transfer_input_only": (role("input", color=INPUT_ONLY_COLOR), r"Nav. $\mathbf{W}_{\mathrm{in}}$ only"),
    "transfer_recurrent_only": (role("recurrent"), r"Nav. $\mathbf{W}_{\mathrm{rec}}$ only"),
    
    
    "random_init": (role("random"), "Random init."),
}
FROZEN = [
    ("transfer_freeze", "Navigation", role("navigation")),
    ("ctl01_frozen", "DR", role("dr")),
    ("random_frozen", "Random", role("random")),
]
#: Paired differences A - B: colour = mix of A and B, marker = B's (difference_spec).
CONTRASTS = [
    ("transfer_freeze__minus__ctl01_frozen", "Nav. − DR", difference_spec(role("navigation"), role("dr"))),
    ("transfer_freeze__minus__random_frozen", "Nav. − random", difference_spec(role("navigation"), role("random"))),
    ("ctl01_frozen__minus__random_frozen", "DR − random", difference_spec(role("dr"), role("random"))),
]


def _panel_learning(ax, curves):
    q = curves[curves.arm.isin(TRAINABLE)] if not curves.empty else curves
    if q.empty:
        empty_panel(ax); return
    handles = []
    for arm in ["random_init", "transfer_recurrent_only", "transfer_input_only", "transfer_finetune"]:
        g = q[q.arm.astype(str) == arm]
        if g.empty:
            continue
        spec, text = CURVE_STYLE[arm]
        rows = sorted((float(x), *mean_sem(v.value.to_numpy(float))) for x, v in g.groupby("x"))
        arr = np.asarray(rows, dtype=float)
        ax.fill_between(arr[:, 0], arr[:, 1] - arr[:, 2], arr[:, 1] + arr[:, 2],
                        color=spec["color"], alpha=BAND_ALPHA, lw=0, zorder=1)
        every = 2 if EPOCH_AXIS == "linear" else 0.085
        # Both pretrained matrices = full-colour solid line; the other
        # initialisations = light solid lines (markers keep the full colour).
        hero = arm == "transfer_finetune"
        kw_fn = line_kw if hero else light_line_kw
        ax.plot(arr[:, 0], arr[:, 1], **kw_fn(spec, markevery=every, zorder=4 if hero else 3))
        handles.append((arm, legend_handle(spec, text) if hero else light_legend_handle(spec, text)))
    # Two-column key in the head-room band above the ceiling.
    order = ["transfer_finetune", "transfer_recurrent_only", "transfer_input_only", "random_init"]
    handles = [h for key in order for a, h in handles if a == key]
    reference_line(ax, CHANCE)
    last = float(q.x.max())                      # 100 target epochs
    if EPOCH_AXIS == "linear":
        top = float(np.ceil(last / 20.0) * 20.0)
        ax.set_xlim(0, top + 0.04 * top)        # small margin: the epoch-100 markers are not clipped
        ax.set_xticks(np.arange(0, top + 1e-9, 20))
        ax.set_xticks(np.arange(10, top, 20), minor=True)
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_xscale("log")
        ticks = [t for t in (5, 10, 20, 50, 100, 200, 500) if t <= max(last, 5) * 1.0001]
        if ticks[-1] < last:
            ticks.append(int(np.ceil(last)))
        ax.set_xlim(4.4, ticks[-1] * 1.08)
        ax.set_xticks(ticks, [str(t) for t in ticks])
        ax.minorticks_off()
    accuracy_axis(ax, label="Validation accuracy", hi=1.40)
    ax.set_xlabel("Target training epoch")
    # Key starts just right of the (now full-height) y axis.
    legend(ax, loc="upper left", handles=handles, ncol=2, bbox_to_anchor=(0.025, 1.03),
           handlelength=1.75, columnspacing=0.8)
    closed_frame(ax)


def _panel_frozen(ax, frozen, contrasts):
    if frozen.empty:
        empty_panel(ax); return
    wide = frozen.pivot_table(index="source_seed", columns="arm", values="value", aggfunc="mean")
    xs, ys = [], []
    for i, (arm, _text, spec) in enumerate(FROZEN):
        vals = wide[arm].to_numpy(float) if arm in wide else np.array([])
        off = swarm(ax, i, vals, spec=spec, width=0.13)
        xs.append(i + off)
        ys.append(vals)
        m, _lo, _hi = summary_errorbar(ax, i + 0.30, vals, spec=spec)
        if np.isfinite(m):
            # Mean value beside its summary symbol (white outline over pair lines).
            ax.annotate(MEAN_LABEL_FMT.format(m), xy=(i + 0.30, m), xytext=(6.5, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=FS_ANNOT, color=PALETTE["black"], path_effects=halo(),
                        zorder=7, annotation_clip=False)
    pair_lines(ax, xs, ys)
    reference_line(ax, CHANCE)
    if not contrasts.empty and P_COLUMN in contrasts:
        pmap = dict(zip(contrasts.contrast.astype(str), contrasts[P_COLUMN]))
        brackets = [
            ("ctl01_frozen__minus__random_frozen", 1, 2, 0.42),
            ("transfer_freeze__minus__ctl01_frozen", 0, 1, 1.07),
            ("transfer_freeze__minus__random_frozen", 0, 2, 1.19),
        ]
        for key, a, b, y in brackets:
            if key in pmap:
                significance_bracket(ax, a, b + 0.30, y, stars(pmap[key]), h=0.03,
                                     fontsize=7.5, text_pad_pt=-1.0)
    # Category tick at the centre of each source cloud + summary pair.
    ax.set_xticks([i + 0.15 for i in range(len(FROZEN))], [t for _, t, _ in FROZEN])
    categorical_xlim(ax, [0, 1, 2.3], pad=0.40)
    ax.set_xlim(ax.get_xlim()[0], 3.02)          # room for the mean label of the last group
    accuracy_axis(ax, label="Test accuracy", hi=1.32)
    closed_frame(ax)


def _panel_differences(ax, contrasts, paired):
    if paired.empty:
        empty_panel(ax); return
    n = len(CONTRASTS)
    for i, (key, _text, spec) in enumerate(CONTRASTS):
        y = n - 1 - i
        diffs = paired[paired.contrast.astype(str) == key].difference.to_numpy(float)
        row = contrasts[contrasts.contrast.astype(str) == key]
        if row.empty:
            continue
        r = row.iloc[0]
        estimation_row(ax, y, diffs, spec=spec, mean=float(r["mean"]), lo=float(r["ci_low"]),
                       hi=float(r["ci_high"]), sem=float(r["sem"]), df=int(r["df"]),
                       density_height=0.0, point_offset=-0.17)
    reference_line(ax, 0, orientation="v")
    ax.set_yticks(range(n), [c[1] for c in CONTRASTS][::-1])
    ax.set_ylim(-0.55, n - 0.45)
    # Same 0-1 scale as the accuracies in A and B; starts below 0 so the zero
    # reference line stays clear of the y axis.
    accuracy_axis(ax, which="x", label=r"$\Delta\mathrm{Acc}_{\mathrm{test}}$")
    closed_frame(ax)


def build(ctx: FigureContext):
    apply_publication_style()
    data = ctx.source_data_dir / "Figure3"
    out = ctx.figures_dir
    curves = load_csv(data / "training_curves.csv")
    frozen = load_csv(data / "frozen_transfer.csv")
    contrasts = load_csv(data / "contrasts.csv")
    paired = load_csv(data / "paired_differences.csv")

    fig = plt.figure(figsize=(FIG_WIDTH, 2.28))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.62, 1.05, 1.08],
                          left=.075, right=.985, top=.90, bottom=.20, wspace=.52)
    ax_a, ax_b, ax_c = (fig.add_subplot(gs[i]) for i in range(3))
    _panel_learning(ax_a, curves)
    _panel_frozen(ax_b, frozen, contrasts)
    _panel_differences(ax_c, contrasts, paired)
    # Identical white space between the three columns (ink to ink).
    equalize_column_gaps(fig, [ax_a, ax_b, ax_c])
    place_panel_labels(fig, [(ax_a, "A"), (ax_b, "B"), (ax_c, "C")])

    paths = save_publication_figure(fig, out / "Figure3")
    inputs = [data / "training_curves.csv", data / "frozen_transfer.csv",
              data / "contrasts.csv", data / "paired_differences.csv"]
    write_json(
        out / "manifests" / "Figure3.json",
        manifest_for_figure(
            ctx, "Figure3", Path(__file__), inputs, paths,
            notes=[
                "All navigation transfer arms use the validation-best navigation checkpoint; target repeats are averaged within source before between-source inference.",
                "A: mean +/- SEM across navigation sources; dashed line = chance (1/16); the epoch axis includes the last target epoch (100).",
                f"B: points = sources, grey lines pair each source; large symbols = mean +/- 95% t-CI; asterisks from {P_COLUMN} (contrasts.csv).",
                "C: points = per-source paired differences; large symbol and bar = mean and 95% t-CI from contrasts.csv; x axis on the 0-1 accuracy scale. Colour of A - B = equal sRGB mix of the colours of A and B; marker = B's.",
                "A: comparison initialisations drawn as light solid lines (no dashes); B: condition means printed beside the summary symbols.",
                "Layout: A, B and C are separated by identical ink-to-ink white space (equalize_column_gaps).",
            ],
        ),
    )
    return paths
