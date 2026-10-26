"""Supplementary Figure S12: recurrent rescaling interventions by path length and source.

Reads ``figure_source_data/SupplementaryFigureS12`` (written by
tools/mechanism_v1/E6_retention_intervention_v1/scripts/export_figureS12_source_data.py) and draws
  a  seen-composition accuracy by path length (rho = 0, rho = 0.10, impairment, rescue)
  b  one-step retention on navigation-derived reference states, per source, before/after rescaling
  c  the same ratio on states visited during 12-relation seen-composition test trials
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .nature_style import (apply_publication_style, save_publication_figure, FIG_WIDTH, PALETTE, LINEWIDTH, DASH,
                               accuracy_axis, integer_axis, closed_frame, end_labels, reference_line,
                               summary_errorbar, place_panel_labels, FS_INPANEL, deterministic_jitter)
    from .mechanism_plot_common import OUT, figure_data, read_csv, mean_curve, RHO_COLORS, MID
except ImportError:
    from nature_style import (apply_publication_style, save_publication_figure, FIG_WIDTH, PALETTE, LINEWIDTH, DASH,
                              accuracy_axis, integer_axis, closed_frame, end_labels, reference_line,
                              summary_errorbar, place_panel_labels, FS_INPANEL, deterministic_jitter)
    from mechanism_plot_common import OUT, figure_data, read_csv, mean_curve, RHO_COLORS, MID

ARMS = {
    "rho0": dict(label=r"$\rho$ = 0", spec=dict(color=RHO_COLORS[0.0], marker="o", ls=DASH["solid"], filled=True, lw=LINEWIDTH, ms_scale=1.0)),
    "nav_impair": dict(label="Impairment", spec=dict(color=PALETTE["negative"], marker="v", ls=DASH["dash"], filled=True, lw=LINEWIDTH, ms_scale=1.0)),
    "rho0.1": dict(label=r"$\rho$ = 0.10", spec=dict(color=RHO_COLORS[0.10], marker="s", ls=DASH["dot"], filled=True, lw=LINEWIDTH, ms_scale=0.92)),
    "rho01_rescue": dict(label="Rescue", spec=dict(color=PALETTE["recurrent"], marker="D", ls=DASH["dashdot"], filled=True, lw=LINEWIDTH, ms_scale=0.88)),
}

# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_profile(ax, prof):
    order = ["rho0", "rho0.1", "rho01_rescue", "nav_impair"]
    for arm in order[::-1]:
        mean_curve(ax, prof[prof.arm == arm], "path_length", "accuracy", spec=ARMS[arm]["spec"],
                   zorder=5 if arm == "rho0" else 4)
    ax.axvline(5.5, color=MID, lw=.8, ls=(0, (3, 2)), zorder=0)
    accuracy_axis(ax, label="Test accuracy")
    integer_axis(ax, lo=1, hi=12, step=2, label="Path length $L_{\\mathrm{p}}$")
    items = []
    for arm in order:
        q = prof[(prof.arm == arm) & (prof.path_length == 12)]
        items.append((float(q.accuracy.mean()), ARMS[arm]["label"], ARMS[arm]["spec"]["color"]))
    end_labels(ax, items, x=12.2, dx_pt=4, fontsize=FS_INPANEL)
    closed_frame(ax)


POS = {"rho0": 0.0, "nav_impair": 1.0, "rho0.1": 2.45, "rho01_rescue": 3.45}
GROUPS = (("Impairment", "rho0", "nav_impair"), ("Rescue", "rho0.1", "rho01_rescue"))


def _paired_dots(ax, df, targets=None):
    """Per-source values before and after each rescaling, joined within source."""
    jit = deterministic_jitter(10, width=0.10)
    for _, a, b in GROUPS:
        va = df[df.arm == a].set_index("source_seed").value
        vb = df[df.arm == b].set_index("source_seed").value.reindex(va.index)
        for k, s in enumerate(va.index):
            ax.plot([POS[a] + jit[k], POS[b] + jit[k]], [va[s], vb[s]], color=MID, lw=.55, alpha=.35, zorder=1)
    for arm in POS:
        spec = ARMS[arm]["spec"]
        vals = df[df.arm == arm].sort_values("source_seed").value.to_numpy(float)
        ax.scatter(POS[arm] + jit[:len(vals)], vals, s=16, marker=spec["marker"], facecolor=spec["color"],
                   edgecolor="white", linewidth=.45, alpha=.75, zorder=3)
        summary_errorbar(ax, POS[arm] + .27, vals, spec=spec)
    if targets:
        for arm, value in targets.items():
            x0, x1 = POS[arm] - .30, POS[arm] + .55
            ax.plot([x0, x1], [value, value], color=PALETTE["black"], lw=.9, ls=(0, (2.2, 1.4)), zorder=7)
            if arm == "nav_impair":
                xy, offset, ha, va = ((x0 + x1) / 2, value), (0, -7), "center", "top"
            else:
                xy, offset, ha, va = (x1, value), (2, 0), "left", "center"
            ax.annotate("target", xy=xy, xytext=offset, textcoords="offset points",
                        ha=ha, va=va, fontsize=FS_INPANEL - .3, color=PALETTE["neutral_dark"])
    ax.set_xticks([POS[a] for a in POS], ["Before", "After", "Before", "After"])
    ax.set_xlim(-0.45, 4.25)
    ax.tick_params(axis="x", length=0, pad=2.5)
    trans = ax.get_xaxis_transform()
    for name, a, b in GROUPS:
        ax.text((POS[a] + POS[b]) / 2, -0.165, name, transform=trans, ha="center", va="top",
                fontsize=FS_INPANEL + .4, color=PALETTE["black"], clip_on=False)


def panel_reference(ax, ref):
    k0 = ref[ref.arm == "rho0"].value.mean()
    k1 = ref[ref.arm == "rho0.1"].value.mean()
    # intended targets: impairment -> mean kappa(0.10); rescue -> mean kappa(0)
    _paired_dots(ax, ref, targets={"nav_impair": k1, "rho01_rescue": k0})
    reference_line(ax, 1.0)
    ax.set_ylim(0.70, 1.06)
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.set_yticks([0.75, 0.85, 0.95, 1.05], minor=True)
    ax.set_ylabel("Retention $\\langle\\kappa\\rangle$\n(reference states)")
    closed_frame(ax)


def panel_task(ax, task):
    _paired_dots(ax, task)
    reference_line(ax, 1.0)
    ax.set_ylim(0.35, 1.18)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.5, 0.7, 0.9, 1.1], minor=True)
    ax.set_ylabel("Retention $\\langle\\kappa\\rangle$\n(test-trial states)")
    closed_frame(ax)


def make_s12():
    data = figure_data("SupplementaryFigureS12")
    prof = read_csv(data / "panelA_rescaling_profile.csv", ("arm", "source_seed", "path_length", "accuracy"))
    ref = read_csv(data / "panelB_retention_reference_states.csv", ("arm", "source_seed", "retention")).rename(columns={"retention": "value"})
    task = read_csv(data / "panelC_retention_task_states.csv", ("arm", "source_seed", "retention")).rename(columns={"retention": "value"})
    apply_publication_style()
    fig = plt.figure(figsize=(FIG_WIDTH, 2.45))
    gs = fig.add_gridspec(1, 3, left=.07, right=.985, top=.93, bottom=.22, width_ratios=[1.5, 1.0, 1.0], wspace=.58)
    a, b, c = (fig.add_subplot(gs[i]) for i in range(3))
    panel_profile(a, prof)
    pa = a.get_position()
    a.set_position([pa.x0, pa.y0, pa.width - 0.045, pa.height])
    panel_reference(b, ref)
    panel_task(c, task)
    place_panel_labels(fig, [(a, "a"), (b, "b"), (c, "c")])
    return save_publication_figure(fig, OUT / "SupplementaryFigureS12")


if __name__ == "__main__":
    make_s12()
