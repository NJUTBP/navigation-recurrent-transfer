#!/usr/bin/env python3
"""Supplementary Figure S4 (builder FigureA2): relation-label and endpoint-mapping controls."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .common import (
    FigureContext, load_csv, manifest_for_figure, plot_named_source_points,
    plot_paired_named, write_json,
)
from .nature_style import PALETTE, place_panel_labels, categorical_xlim, apply_publication_style, save_publication_figure
from .nature_style import FIG_WIDTH, accuracy_axis, mix_colors


def build(ctx: FigureContext):
    apply_publication_style()
    data = ctx.source_data_dir / "FigureA2"
    out = ctx.figures_dir
    mapping = load_csv(data / "mapping.csv")
    contrasts = load_csv(data / "contrasts.csv")
    paired = load_csv(data / "paired_differences.csv")

    fig = plt.figure(figsize=(FIG_WIDTH, 2.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 0.92])
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    fig.subplots_adjust(left=.075, right=.990, top=.86, bottom=.22, wspace=.42)

    present = set(mapping.arm.astype(str)) if not mapping.empty and "arm" in mapping else set()
    specs = [
        ("aligned", "Aligned", PALETTE["navigation"], "o"),
        ("shuffled", "Shuffled labels", PALETTE["relation"], "D"),
        ("mismatched_target", "Mismatched targets", PALETTE["random"], "s"),
    ]
    specs = [s for s in specs if s[0] in present]
    plot_named_source_points(a, mapping, specs, reference=1/16, ylabel="Test accuracy (trainable core)", paired_lines=True)
    accuracy_axis(a, label="Test accuracy (trainable core)")   # shared 0-1 scale

    avail = set(paired.contrast.astype(str)) if not paired.empty and "contrast" in paired else set()
    pspecs = [
        # Difference rule: colour of A - B = mix of A and B, marker of B.
        ("aligned__minus__shuffled", "Aligned − shuffled",
         mix_colors(PALETTE["navigation"], PALETTE["relation"]), "D"),
        ("aligned__minus__mismatched_target", "Aligned − mismatched",
         mix_colors(PALETTE["navigation"], PALETTE["random"]), "s"),
    ]
    pspecs = [s for s in pspecs if s[0] in avail]
    # Fall back to contrast-table names without changing the source-data interface.
    if not pspecs and not contrasts.empty:
        for c in contrasts.contrast.astype(str):
            if c in avail:
                name = c.replace("aligned__minus__", "").replace("_", " ").title()
                pspecs.append((c, name, PALETTE["navigation"]))
    plot_paired_named(
        b, paired, pspecs,
        xlabel="Test accuracy difference",
        xlim=(-0.03, 1.00),
        shade_positive=False,
    )

    place_panel_labels(fig, [(a, "A"), (b, "B")])

    paths = save_publication_figure(fig, out / "FigureA2")
    inputs = [data / "mapping.csv", data / "contrasts.csv", data / "paired_differences.csv"]
    write_json(
        out / "manifests" / "FigureA2.json",
        manifest_for_figure(
            ctx, "FigureA2", Path(__file__), inputs, paths,
            notes=["Mapping controls support interpretation and therefore remain outside the primary interface-relearning figure."],
        ),
    )
    return paths
