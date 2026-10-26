"""Final mechanism Figure 7 from formally frozen source-data.

Presentation-only.  Panel identity, panel-to-table assignment, terminology and
every plotted number follow ``Figure7/PANEL_CONTRACT.json`` and the frozen
``figure7_statistics.json`` exactly as before.

v2: no panel titles; path-length / step axes labelled 1, 3, ..., 11 with minor
ticks; accuracy axes start below 0; short/long splits and the three rho doses
are ~20 L* apart (greyscale safe) with distinct markers and dashes; D has
direct end-of-line labels.  v2.1: no in-panel result numbers (reported in the
main text); E shows short- and long-path effects of both retention interventions.
v2.4: closed axes (x and y meet in one corner) in all panels; every axis
contains all plotted values -- A now shows the full per-source traces (up to
2.1; previously clipped at 1.48), C uses the 0-1 scale of B, E's labelled
range reaches -0.8; A, B and C are separated by identical white space.
v2.5 (notation follows the main text): A = relative rectified input drive
delta_t over relation step t (Eq. 10); B/D y axes "Test accuracy"; C = autonomous
retention kappa-bar (thick line = mean over sources, kappa-bar(rho)); "Perturbation
fraction rho" without comma; D starts at 0.12 (data >= 0.16) and has no
"trained"/"extrapolation" labels; E names the two arms as in Section 2.6
(impairment: c W_rec(0) of the rho = 0 core; rescue: W_rec(0.10)/c) and its y
axis is Delta alpha_q, the difference from the corresponding unmanipulated
baseline.
v2.8: the v2.5 layout is kept unchanged; only the panel letters follow the order
of citation in Section 3.6: top row A, B, D (D = autonomous retention
<kappa>(rho), sharing the dose axis of B); bottom row C (path-length profile)
and E.  Path length is written L_p and retention <kappa>, as in the main text.
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
        save_publication_figure, FIG_WIDTH, PALETTE, LINEWIDTH, FS_MIN, role,
        legend_handle, accuracy_axis, integer_axis, closed_frame, swarm,
        summary_errorbar, end_labels, fmt_p, equalize_column_gaps,
    )
    from .mechanism_plot_common import (
        OUT, NAV, MATCHED, REL, REC, NEG, MID, SHORT, LONG, RHO_COLORS, figure_data,
        read_csv, read_json, mean_curve, source_swarm_with_ci, annotate, legend,
        categorical_xlim, place_panel_labels, num, FS_INPANEL,
    )
except ImportError:
    from nature_style import (
        apply_publication_style, reference_line, style_axis,
        save_publication_figure, FIG_WIDTH, PALETTE, LINEWIDTH, FS_MIN, role,
        legend_handle, accuracy_axis, integer_axis, closed_frame, swarm,
        summary_errorbar, end_labels, fmt_p, equalize_column_gaps,
    )
    from mechanism_plot_common import (
        OUT, NAV, MATCHED, REL, REC, NEG, MID, SHORT, LONG, RHO_COLORS, figure_data,
        read_csv, read_json, mean_curve, source_swarm_with_ci, annotate, legend,
        categorical_xlim, place_panel_labels, num, FS_INPANEL,
    )

DATA = figure_data("Figure7")
A = read_csv(DATA / "panelA_multistep_dynamics.csv", ("source_seed", "origin", "path_step", "delta_rectified"))
B = read_csv(DATA / "panelB_accuracy_dose.csv", ("source_seed", "rho", "id_seen_short", "seen_long"))
C = read_csv(DATA / "panelC_retention_dose.csv", ("source_seed", "rho", "autonomous_retention_mean"))
D = read_csv(DATA / "panelD_horizon_profile.csv", ("source_seed", "rho", "path_length", "accuracy"))
E = read_csv(DATA / "panelE_intervention_effects.csv", ("source_seed", "arm", "split", "difference"))
E_TARGETS = read_csv(DATA / "panelE_retention_targets.csv", ("arm", "source_retention_before_mean", "source_retention_after_exact_mean"))
STATS = read_json(DATA / "figure7_statistics.json")

ORIGIN = {
    "navigation": ("Navigation", role("navigation")),
    "matched_random": ("Matched control", role("matched")),
}
SHORT_SPEC = dict(color=SHORT, marker="o", ls="-", filled=True, lw=LINEWIDTH, ms_scale=1.0)
LONG_SPEC = dict(color=LONG, marker="s", ls=(0, (4.2, 1.8)), filled=True, lw=LINEWIDTH, ms_scale=0.92)
RHO_SPECS = [
    (0.0, r"$\rho$ = 0", dict(color=RHO_COLORS[0.0], marker="o", ls="-", filled=True, lw=LINEWIDTH, ms_scale=1.0)),
    (0.05, r"$\rho$ = 0.05", dict(color=RHO_COLORS[0.05], marker="D", ls=(0, (4.2, 1.8)), filled=True, lw=LINEWIDTH, ms_scale=0.88)),
    (0.10, r"$\rho$ = 0.10", dict(color=RHO_COLORS[0.10], marker="s", ls=(0, (1.1, 1.5)), filled=True, lw=LINEWIDTH, ms_scale=0.92)),
]

# Shared dose axis for panels B and C: identical ticks, limits and label.
RHO_TICKS = [0, .1, .2, .35, .5]
RHO_TICKLABELS = ["0", "0.1", "0.2", "0.35", "0.5"]
RHO_LIM = (-.025, .525)
RHO_LABEL = r"Perturbation fraction $\rho$"


def _dose_axis(ax) -> None:
    ax.set_xlim(*RHO_LIM)
    ax.set_xticks(RHO_TICKS, RHO_TICKLABELS)
    ax.set_xlabel(RHO_LABEL)


def panel_multistep(ax):
    """A: multi-step recurrent drive for navigation and matched cores."""
    handles = []
    for origin in ("matched_random", "navigation"):
        label, spec = ORIGIN[origin]
        mean_curve(ax, A[A.origin == origin], "path_step", "delta_rectified", spec=spec,
                   zorder=5 if origin == "navigation" else 3)
        handles.append(legend_handle(spec, label))
    integer_axis(ax, lo=1, hi=12, step=2, label="Relation step $t$")  # manuscript: delta_t at relation step t
    # The axis holds every per-source trace (matched sources reach 2.1 at step
    # 12; the former 0.30-1.48 window clipped three of them).
    # Same 0.4 grid as before: labels every 0.8, minor ticks in between; the
    # axis ends at the first labelled value above the data (2.11 -> 2.4).
    step = 0.8
    top = float(np.ceil(A.delta_rectified.max() / step) * step)
    ax.set_ylim(0, top)
    ax.set_yticks(np.arange(0, top + 1e-9, step))
    ax.set_yticks(np.arange(step / 2, top, step), minor=True)
    ax.set_ylabel("Relative rectified\ninput drive $\\delta_t$")   # Eq. 10
    legend(ax, loc="upper left", handles=handles[::-1], bbox_to_anchor=(0.025, 1.02))
    closed_frame(ax)


def panel_dose_accuracy(ax):
    """B: dose response of short- and long-path accuracy."""
    handles = []
    for col, label, spec in [
        ("seen_long", "Long, $L_{\\mathrm{p}}$ = 6–12", LONG_SPEC),
        ("id_seen_short", "Short, $L_{\\mathrm{p}}$ = 1–5", SHORT_SPEC),
    ]:
        q = B[["source_seed", "rho", col]].rename(columns={col: "value"})
        mean_curve(ax, q, "rho", "value", spec=spec, zorder=5 if col == "seen_long" else 4)
        handles.append(legend_handle(spec, label))
    _dose_axis(ax)
    accuracy_axis(ax, label="Test accuracy")
    # Key kept just clear of the corner where the two axes meet.
    legend(ax, loc="lower left", handles=handles[::-1], bbox_to_anchor=(0.025, 0.02))
    closed_frame(ax)


def panel_retention(ax):
    """D (top right): autonomous state retention on the same dose axis as B."""
    spec = dict(color=REC, marker="o", ls="-", filled=True, lw=LINEWIDTH, ms_scale=1.0)
    mean_curve(ax, C, "rho", "autonomous_retention_mean", spec=spec)
    _dose_axis(ax)
    # Same 0-1 scale as B (retention reaches 0.19 at rho = 0.5, below the old
    # 0.25 bottom tick); 0 and 1 line up across the two dose panels.
    # kappa: core-level one-step autonomous retention (Eq. 12, averaged over
    # reference states); the thick line is its mean over sources, kappa-bar(rho).
    accuracy_axis(ax, label="Autonomous retention $\\langle\\kappa\\rangle$")
    closed_frame(ax)


def panel_horizon(ax):
    """C (bottom left): horizon profile at the three doses (direct end-of-line labels)."""
    for rho, label, spec in RHO_SPECS[::-1]:
        mean_curve(ax, D[np.isclose(D.rho, rho)], "path_length", "accuracy", spec=spec,
                   zorder=5 if rho == 0 else 4)
    ax.axvline(5.5, color=MID, lw=.8, ls=(0, (3, 2)), zorder=0)
    # The axis starts above 0: every source-level value is >= 0.16 (rho = 0.10,
    # L = 12) and no chance line is drawn in this panel.
    accuracy_axis(ax, label="Test accuracy", lo=0.12, hi=1.04,
                  major=(0.2, 0.4, 0.6, 0.8, 1.0), minor=())
    integer_axis(ax, lo=1, hi=12, step=2, label="Path length $L_{\\mathrm{p}}$")
    # Dashed line at L = 5.5: target-training lengths 1-5 vs longer paths
    # (explained in the caption; no in-panel labels).
    items = []
    for rho, label, spec in RHO_SPECS:
        q = D[np.isclose(D.rho, rho)]
        last = q[q.path_length == q.path_length.max()].accuracy.mean()
        items.append((float(last), label, spec["color"] if rho != 0.10 else "#C46A57"))
    end_labels(ax, items, x=12.2, dx_pt=4, fontsize=FS_INPANEL)
    closed_frame(ax)


def panel_intervention(ax):
    """E: bidirectional retention intervention (short vs long paths)."""

    # Section 2.6 terms: the impairment arm rescales W_rec(0) of the rho = 0
    # core by c (lowering retention to the rho = 0.10 level); the rescue arm
    # rescales W_rec(0.10) by 1/c (restoring the rho = 0 level).
    groups = [
        ("nav_impair", "Impairment\n" + r"$c\,\mathbf{W}_{\mathrm{rec}}(0)$"),
        ("rho01_rescue", "Rescue\n" + r"$\mathbf{W}_{\mathrm{rec}}(0.10)/c$"),
    ]


    split_specs = [
        ("id_seen_short", SHORT_SPEC, -.17),
        ("seen_long", LONG_SPEC, .17),
    ]
    stats_long = {e["arm"]: e for e in STATS["panelE"]
                  if e.get("split") == "seen_long" and e.get("role") == "primary"}
    x = np.arange(len(groups), dtype=float)
    for i, (arm, _label) in enumerate(groups):
        for split, spec, offset in split_specs:
            vals = E[(E.arm == arm) & (E.split == split)].difference.to_numpy(float)
            xs = x[i] + offset
            swarm(ax, xs, vals, spec=spec, width=0.07, s=17)
            if split == "seen_long" and arm in stats_long:
                st = stats_long[arm]   # frozen long-path effect (mean, 95% CI)
                summary_errorbar(ax, xs + 0.14, spec=spec, mean=st["difference"],
                                 lo=st["ci_low"], hi=st["ci_high"])
            else:
                summary_errorbar(ax, xs + 0.14, vals, spec=spec)
    reference_line(ax, 0)
    ax.set_xlim(-0.55, 1.62)
    # Labelled range -0.8 to 0 encloses every source (largest drop -0.67).
    ax.set_ylim(-.80, .10)
    ax.set_yticks([-.8, -.6, -.4, -.2, 0])
    ax.set_yticks([-.7, -.5, -.3, -.1], minor=True)
    ax.set_xticks(x + 0.07, [g[1] for g in groups])
    # Within-source difference from the corresponding unmanipulated baseline
    # (rho = 0 core for impairment, rho = 0.10 core for rescue; Section 2.8).
    ax.set_ylabel(r"$\Delta\alpha_q$")
    handles = [legend_handle(SHORT_SPEC, "Short, $L_{\\mathrm{p}}$ = 1–5", line=False),
               legend_handle(LONG_SPEC, "Long, $L_{\\mathrm{p}}$ = 6–12", line=False)]
    legend(ax, loc="lower right", points=True, handles=handles, bbox_to_anchor=(1.02, 0.02))
    closed_frame(ax)


def make_figure7():
    apply_publication_style()
    fig = plt.figure(figsize=(FIG_WIDTH, 4.55))
    outer = fig.add_gridspec(2, 1, left=.075, right=.985, top=.965, bottom=.115,
                             hspace=.44, height_ratios=[1.0, 1.08])
    top = outer[0].subgridspec(1, 3, wspace=.42)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.80, 1.20], wspace=.36)
    a = fig.add_subplot(top[0])
    b = fig.add_subplot(top[1])
    c = fig.add_subplot(top[2])
    d = fig.add_subplot(bottom[0])
    e = fig.add_subplot(bottom[1])
    panel_multistep(a)
    panel_dose_accuracy(b)
    panel_retention(c)
    panel_horizon(d)
    panel_intervention(e)
    # C (path-length profile) keeps a gutter on its right for the direct labels.
    pd_ = d.get_position()
    d.set_position([pd_.x0, pd_.y0, pd_.width - 0.052, pd_.height])
    # Identical white space between A, B and D (ink to ink).
    equalize_column_gaps(fig, [a, b, c])
    fig.align_ylabels([a, d])
    # v2.8: letters follow the order of citation (C = profile, D = retention).
    place_panel_labels(fig, [(a, "A"), (b, "B"), (c, "D"), (d, "C"), (e, "E")])
    return save_publication_figure(fig, OUT / "Figure7")


if __name__ == "__main__":
    make_figure7()
