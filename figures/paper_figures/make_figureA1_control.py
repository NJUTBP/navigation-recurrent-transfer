#!/usr/bin/env python3
"""FigureA1 — delayed-reconstruction control pretraining and selected-delay error."""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from .common import FigureContext, empty_panel, load_csv, manifest_for_figure, write_json
from .nature_style import lighten
from .nature_style import (
    BAND_ALPHA, LINEWIDTH, PALETTE, place_panel_labels, legend, apply_publication_style,
    mean_sem, save_publication_figure, style_axis,
)

_EPOCH_TICKS = [1, 4, 16, 64, 128]
_DELAY_TICKS = [1, 4, 7, 10]


def _curve(frame, arm):
    """Return source-level mean ± SEM after the pre-update epoch-0 row.

    The control table contains an x=0 initialization row with NMSE ~1.  Panel A
    uses a logarithmic epoch axis to resolve the long post-update learning
    trajectory, so the initialization row is deliberately excluded rather than
    letting an unplottable x=0 observation distort the y-axis limits.
    """
    z = frame[(frame.metric == "nmse") & (frame.arm == arm)] if not frame.empty else frame
    if not z.empty:
        z = z[pd_numeric(z["x"]) > 0]
    rows = []
    for x, g in z.groupby("x"):
        m, se = mean_sem(g.value.to_numpy(float))
        rows.append((float(x), m, se))
    return np.asarray(sorted(rows), float) if rows else np.empty((0, 3))


def pd_numeric(series):
    """Small local numeric coercion helper without changing the data interface."""
    import pandas as pd
    return pd.to_numeric(series, errors="coerce")


def _sem_band(ax, x, mean, sem, *, color, alpha):
    """Draw a log-safe SEM band without changing the underlying estimates."""
    lower = np.maximum(mean - sem, np.finfo(float).tiny)
    upper = mean + sem
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, lw=0)


def build(ctx: FigureContext):
    apply_publication_style()
    data = ctx.source_data_dir / "FigureA1"
    out = ctx.figures_dir
    control = load_csv(data / "control_source_means.csv")

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.55))
    fig.subplots_adjust(left=.088, right=.988, top=.88, bottom=.22, wspace=.36)
    primary = _curve(control, "primary_delay")
    aux = _curve(control, "auxiliary_mean")

    if len(primary) == 0:
        empty_panel(a)
    else:
        a.plot(
            primary[:, 0], primary[:, 1], color=PALETTE["control"],
            lw=LINEWIDTH, label=r"$\tau$ = 10 (primary)",
        )
        _sem_band(
            a, primary[:, 0], primary[:, 1], primary[:, 2],
            color=PALETTE["control"], alpha=BAND_ALPHA,
        )
        if len(aux):
            a.plot(
                # v2.5: comparison curve = light solid line (no dashes).
                aux[:, 0], aux[:, 1], color=lighten(PALETTE["neutral_dark"]),
                lw=1.0, ls="-", label=r"$\tau$ = 1–9 (mean)",
            )
            _sem_band(
                a, aux[:, 0], aux[:, 1], aux[:, 2],
                color=PALETTE["neutral_dark"], alpha=.07,
            )
        a.set_xscale("log", base=2)
        a.set_yscale("log")
        # Same epoch axis as Figure 2A/B: 1-256 so every epoch lies inside.
        a.set_xlim(0.8, 256)
        a.set_xticks([1, 4, 16, 64, 256], ["1", "4", "16", "64", "256"])
        a.set_xticks([2, 8, 32, 128], minor=True)
        a.xaxis.set_minor_formatter(NullFormatter())
        a.set_ylim(5e-5, 5e-2)
        a.set_xlabel("Pretraining epoch")
        a.set_ylabel("Validation NMSE")
        a.legend(
            fontsize=6.0, loc="upper right", frameon=False,
            handlelength=1.6, handletextpad=.45, labelspacing=.30,
            borderaxespad=.25,
        )
        style_axis(a)
    # v2: panel title removed (content belongs in the caption): "Delayed-reconstruction learning"

    d = control[control.metric == "selected_delay_nmse"] if not control.empty else control
    if d.empty:
        empty_panel(b)
    else:
        p = d.pivot_table(index="x", columns="source_seed", values="value").sort_index()
        for s in p:
            b.plot(
                p.index, p[s], color=PALETTE["neutral_light"],
                lw=.55, alpha=.85, zorder=1,
            )
        m = p.mean(axis=1)
        se = p.std(axis=1, ddof=1) / np.sqrt(max(1, p.shape[1]))
        b.plot(
            p.index, m, "-o", ms=3.2, color=PALETTE["control"],
            lw=LINEWIDTH, markeredgecolor="white", markeredgewidth=.35,
            zorder=3,
        )
        _sem_band(
            b, p.index.to_numpy(float), m.to_numpy(float), se.to_numpy(float),
            color=PALETTE["control"], alpha=BAND_ALPHA,
        )
        b.set_yscale("log")
        b.set_xlim(.7, 10.3)
        b.set_xticks(_DELAY_TICKS, [str(x) for x in _DELAY_TICKS])
        b.set_xlabel(r"Reconstruction delay $\tau$ (steps)")
        b.set_ylabel("Validation NMSE")
        style_axis(b, max_x_ticks=5)
    # v2: panel title removed (content belongs in the caption): "Delay profile at selected checkpoint"

    place_panel_labels(fig, [(a, "A"), (b, "B")])

    paths = save_publication_figure(fig, out / "FigureA1")
    inputs = [data / "control_source_means.csv", data / "source_qualification.csv"]
    write_json(
        out / "manifests" / "FigureA1.json",
        manifest_for_figure(
            ctx, "FigureA1", Path(__file__), inputs, paths,
            notes=[
                "This figure qualifies the matched delayed-reconstruction source control used in Figure3 and Figure5.",
                "Panel A displays post-update epochs 1–150 on a log2 epoch axis; the x=0 initialization row is excluded because zero is undefined on a logarithmic axis and would otherwise distort the y-range.",
                "Panel B shows source-level selected-checkpoint delay profiles with the population mean ± SEM.",
            ],
        ),
    )
    return paths
