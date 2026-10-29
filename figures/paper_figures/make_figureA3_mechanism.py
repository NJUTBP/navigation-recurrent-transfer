#!/usr/bin/env python3
"""Supplementary Figure S2 (builder FigureA3): topology, reduced-dynamics and path-invariance diagnostics."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import FigureContext, empty_panel, label, load_csv, manifest_for_figure, write_json
from .nature_style import (
    CAPSIZE, ERROR_LINEWIDTH, PALETTE, POINT_ALPHA, place_panel_labels,
    apply_publication_style, deterministic_jitter, mean_ci95, reference_line,
    save_publication_figure, style_axis, annotate, categorical_ylim,
    group_brackets, FS_INPANEL, FS_TICK_DENSE,
)

PATH_ORDER = [
    "Ohh-navigation__Oih-navigation__Thh-frozen__Tih-frozen",
    "Ohh-navigation__Oih-navigation__Thh-trainable__Tih-trainable",
]

CRITERION_LABELS = {
    "H1_two_loop_gap": r"$H_1$: two-loop gap",
    "H2_single_void_gap": r"$H_2$: single-void gap",
    "H1_second_to_third": r"$H_1$: 2nd / 3rd",
    "H2_first_to_second": r"$H_2$: 1st / 2nd",
    "surface_mfe": "Surface error",
    "rollout_nmte": "Rollout error",
    "torus_tangent_fraction": "Tangent fraction",
    "normalized_additive_phase_rmse": "Additive phase RMSE",
    "velocity_specificity": "Velocity specificity",
    "base_plus_fibre_tangent_fraction": "Tangent fraction",
    "phase_null_ratio": "Phase-null rollout ratio",
    "validated_low_dimensional_fibre": "Low-dimensional residual",
    "loop_holonomy": "Loop holonomy",
    "zero_padding_phase_difference": "Zero-padding phase",
    "path_identity_decoder_accuracy": "Path identity decoding",
    "residual_time_decoder_r2": "Residual time decoding",
}


def _shade_positive(ax, vals, threshold=0.0, color=None):
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    if not len(vals):
        return
    upper = max(float(vals.max()), threshold) + max(.15, .06 * max(1.0, abs(float(vals.max()))))
    ax.axvspan(threshold, upper, color=color or PALETTE["positive"], alpha=.045, lw=0, zorder=0)


def _horizontal_source_summary(ax, frame, cats, colors, *, xlabel, labels, ref=0.0, positive_side=True, shade_region=False, xlim=None, show_reference=True, xticks=None):
    if frame.empty or not cats:
        empty_panel(ax); return
    all_vals = frame.value.to_numpy(float)
    if shade_region and positive_side:
        _shade_positive(ax, all_vals, threshold=ref)
    y = np.arange(len(cats))[::-1]
    for yi, c in zip(y, cats):
        vals = frame[frame["arm"].astype(str) == str(c)].value.to_numpy(float)
        vals = vals[np.isfinite(vals)]
        color = colors[c]
        ax.scatter(
            vals, np.full(len(vals), yi) + deterministic_jitter(len(vals), width=.065),
            s=13, color=color, alpha=POINT_ALPHA, linewidth=0, zorder=2,
        )
        if len(vals):
            m, lo, hi = mean_ci95(vals)
            ax.errorbar(
                m, yi, xerr=[[m-lo], [hi-m]], fmt="o", ms=4.0,
                color=color, mfc=color, mec=PALETTE["black"], mew=.6,
                capsize=CAPSIZE, elinewidth=ERROR_LINEWIDTH, zorder=4,
            )
    ax.set_yticks(y, [labels[c] for c in cats], fontsize=FS_TICK_DENSE)
    categorical_ylim(ax, y, pad=.55)
    ax.set_xlabel(xlabel)
    if show_reference:
        reference_line(ax, ref, orientation="v")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    style_axis(ax, max_x_ticks=4)


def _margin_panel(ax, frame):
    if frame.empty:
        empty_panel(ax); return
    cats = list(dict.fromkeys(frame["arm"].astype(str).tolist()))
    colors = {c: PALETTE["navigation"] for c in cats}
    labels = {c: CRITERION_LABELS.get(c, label(c)) for c in cats}
    _horizontal_source_summary(
        ax, frame, cats, colors,
        xlabel=r"Signed log$_2$ criterion margin", labels=labels, ref=0.0,
    )


# Rows of panel B, grouped by reduced model in the order the criteria are listed
# in Supplementary Methods S3.2 (group label, gate key, criterion keys).
REDUCED_GROUPS = [
    ("Exact 2D", "exact_2d_gate",
     ["surface_mfe", "rollout_nmte", "torus_tangent_fraction",
      "normalized_additive_phase_rmse", "velocity_specificity"]),
    ("Base + residual", "toroidal_base_plus_fibre_gate",
     ["surface_mfe", "rollout_nmte", "base_plus_fibre_tangent_fraction",
      "phase_null_ratio", "velocity_specificity"]),
]


def _reduced_panel(ax, frame):
    """B: reduced-model criteria grouped by model (grouped rows + brackets, as in C)."""
    if frame.empty:
        empty_panel(ax); return []
    z = frame[frame.metric == "log2_margin"].copy()
    if z.empty:
        empty_panel(ax); return []
    rows = []
    for group_label, gate, crits in REDUCED_GROUPS:
        present = set(z[z["split"].astype(str) == gate]["arm"].astype(str))
        ordered = [c for c in crits if c in present] + sorted(present - set(crits))
        rows += [(group_label, gate, c) for c in ordered]
    y_positions, y = [], float(len(rows)) + .6
    previous = None
    for group_label, _gate, _crit in rows:
        if previous is not None and group_label != previous:
            y -= .8  # visual gap between the two reduced models
        y_positions.append(y); previous = group_label; y -= 1.0
    for yi, (_group, gate, crit) in zip(y_positions, rows):
        q = z[(z["split"].astype(str) == gate) & (z["arm"].astype(str) == crit)]
        vals = q.value.to_numpy(float); vals = vals[np.isfinite(vals)]
        color = PALETTE["navigation"] if gate == "exact_2d_gate" else PALETTE["recurrent"]
        ax.scatter(vals, np.full(len(vals), yi) + deterministic_jitter(len(vals), width=.06), s=12, color=color, alpha=POINT_ALPHA, linewidth=0)
        if len(vals):
            m, lo, hi = mean_ci95(vals)
            ax.errorbar(m, yi, xerr=[[m-lo], [hi-m]], fmt="o", ms=4.6, color=color, mfc=color, mec=PALETTE["black"], mew=.6, capsize=CAPSIZE, elinewidth=ERROR_LINEWIDTH)
    reference_line(ax, 0, orientation="v")
    ax.set_yticks(y_positions, [CRITERION_LABELS.get(c, label(c)) for _g, _gate, c in rows], fontsize=FS_TICK_DENSE)
    ax.set_ylim(min(y_positions) - .7, max(y_positions) + .7)
    ax.set_xlabel(r"Signed log$_2$ criterion margin")
    style_axis(ax, max_x_ticks=4)
    groups = []
    for group_label, _gate, _crits in REDUCED_GROUPS:
        ys = [yp for yp, (g, _ga, _c) in zip(y_positions, rows) if g == group_label]
        if ys:
            groups.append((group_label, min(ys), max(ys)))
    return groups



def _path_panel(ax, path):
    zp = path[path.metric == "threshold_ratio"].copy() if not path.empty else path
    if not zp.empty:
        zp = zp[zp["arm"].astype(str).isin(PATH_ORDER)].copy()
    if zp.empty:
        empty_panel(ax); return

    arm_short = {PATH_ORDER[0]: "Frozen core", PATH_ORDER[1]: "Trainable core"}
    metric_order = [
        "loop_holonomy",
        "path_identity_decoder_accuracy",
        "residual_time_decoder_r2",
        "zero_padding_phase_difference",
    ]
    group_order = [PATH_ORDER[0], PATH_ORDER[1]]
    rows = []
    for arm in group_order:
        present = set(zp[zp["arm"].astype(str) == arm]["transform"].astype(str))
        for metric in metric_order:
            if metric in present:
                rows.append((arm, metric))

    # Add a visual gap between frozen-core and trainable-core groups.
    y_positions = []
    top = len(rows) + 0.6
    for i, (arm, metric) in enumerate(rows):
        y = top - i
        if i >= 4:
            y -= 0.8
        y_positions.append(y)

    metric_labels = {
        "loop_holonomy": "Loop holonomy",
        "path_identity_decoder_accuracy": "Path identity decoding",
        "residual_time_decoder_r2": "Residual time decoding",
        "zero_padding_phase_difference": "Zero-padding phase",
    }

    for yi, (arm, metric) in zip(y_positions, rows):
        vals = zp[(zp["arm"].astype(str) == arm) & (zp["transform"].astype(str) == metric)].value.to_numpy(float)
        vals = vals[np.isfinite(vals)]
        color = PALETTE["relation"] if arm == PATH_ORDER[1] else PALETTE["navigation"]
        ax.scatter(vals, np.full(len(vals), yi) + deterministic_jitter(len(vals), width=.055), s=12, color=color, alpha=POINT_ALPHA, linewidth=0)
        if len(vals):
            m, lo, hi = mean_ci95(vals)
            ax.errorbar(m, yi, xerr=[[m-lo], [hi-m]], fmt="o", ms=4.6, color=color, mfc=color, mec=PALETTE["black"], mew=.6, capsize=CAPSIZE, elinewidth=ERROR_LINEWIDTH)

    reference_line(ax, 1, orientation="v")
    ax.set_yticks(y_positions, [metric_labels[m] for _, m in rows], fontsize=FS_TICK_DENSE)
    ax.set_ylim(min(y_positions) - .7, max(y_positions) + .7)
    ax.set_xlabel("Observed / predefined upper limit")
    style_axis(ax, max_x_ticks=4)
    ax.set_xticks([1, 2, 4, 6])  # label the pass/fail limit at 1
    groups = []
    for arm in group_order:
        ys = [y for y, (a, _m) in zip(y_positions, rows) if a == arm]
        if ys:
            groups.append((arm_short[arm], min(ys), max(ys)))
    return groups



def build(ctx: FigureContext):
    apply_publication_style()
    data = ctx.source_data_dir / "FigureA3"
    out = ctx.figures_dir
    formal = load_csv(data / "formal_topology.csv")
    reduced = load_csv(data / "source_reduced_gates.csv")
    path = load_csv(data / "path.csv")
    qual = load_csv(data / "source_qualification.csv")

    fig = plt.figure(figsize=(7.25, 4.8))
    gs = fig.add_gridspec(
        2, 2, left=.16, right=.992, top=.94, bottom=.12,
        wspace=.68, hspace=.60, height_ratios=[1.0, 1.0],
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    z = formal[formal.metric == "log2_margin"] if not formal.empty else formal
    _margin_panel(ax_a, z)
    #if not qual.empty:
       # n = int(pd.Series(qual.schoyen_formal_topology_passed).astype(str).str.lower().eq("true").sum()) if "schoyen_formal_topology_passed" in qual else 0
       # annotate(ax_a, f"Criterion met: {n}/{ctx.n_sources}", loc="lower right")

    reduced_groups = _reduced_panel(ax_b, reduced)
    if reduced_groups:
        group_brackets(ax_b, reduced_groups)
    #if not qual.empty:
      #  exact = int(pd.Series(qual.source_exact_gate).astype(str).str.lower().eq("true").sum()) if "source_exact_gate" in qual else 0
       # fibre = int(pd.Series(qual.source_fibre_gate).astype(str).str.lower().eq("true").sum()) if "source_fibre_gate" in qual else 0
        # annotate(ax_b, f"Exact 2D: {exact}/{ctx.n_sources}\nBase + residual: {fibre}/{ctx.n_sources}", loc="center left")

    path_groups = _path_panel(ax_c, path)
    if path_groups:
        group_brackets(ax_c, path_groups)

    place_panel_labels(fig, [(ax_a, "A"), (ax_b, "B"), (ax_c, "C")])

    paths = save_publication_figure(fig, out / "FigureA3")
    inputs = [
        data / "formal_topology.csv", data / "source_reduced_gates.csv",
        data / "path.csv", data / "source_qualification.csv",
        data / "source_gate_pass_counts.csv", data / "mechanism_replication_counts.csv",
    ]
    write_json(
        out / "manifests" / "FigureA3.json",
        manifest_for_figure(
            ctx, "FigureA3", Path(__file__), inputs, paths,
            notes=[
                "Formal topology is the Euclidean persistent-homology result of the already locked Schøyen-selected candidate.",
                "No legacy formal_topology/cohomological coordinate contract is used.",
                "Path-invariance statistics are conditional on evaluability and use Schøyen-anchored source_reduced_schoyen_v1 artifacts.",
            ],
        ),
    )
    return paths
