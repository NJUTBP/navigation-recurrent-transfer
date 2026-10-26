"""Shared publication plotting utilities for the final MultiNav manuscript.

Revision v2 (presentation only)
-------------------------------
The style targets Neuron / Cell Press double-column artwork (174 mm wide, text
6-8 pt at print size) and also fits eLife and PLOS Computational Biology
(<= 7.5 in wide).  Compared with v1 it changes *presentation only*:

* no in-panel titles -- what a title used to say now lives in the axis label,
  a legend, a direct line label or the figure caption;
* larger markers and a stricter "one condition = one colour + one marker +
  one line style" contract, so every figure stays readable when printed in
  black and white (colour, lightness, marker shape, fill and dash pattern are
  redundant cues);
* "range-frame" axes: spines are offset outward and trimmed to the first/last
  major tick, so points at 0 or at the ceiling never sit on an axis line;
* shared helpers for accuracy / path-length axes, significance brackets,
  direct end-of-line labels and t-based estimation densities.

Revision v2.4 (presentation only, main Figures 2-7)
---------------------------------------------------
* closed axes: :func:`closed_frame` replaces the range frame in the main
  figures.  The x and y axes always meet in one corner point and run over the
  full axis limits.  Values that sit at 0 are kept off the frame by starting
  the limit at a small negative value, never by offsetting a spine;
* axis limits always contain every plotted value, and the outermost labelled
  ticks enclose the data wherever that costs little space;
* :func:`equalize_column_gaps` gives the panels of one row identical white
  space (measured ink to ink); :func:`axis_break_marks` draws the "//" marks
  of a broken axis.
:func:`range_frame` is unchanged, so the supplementary figures look as before.

Revision v2.5 (presentation only, main Figures 2-7)
---------------------------------------------------
* :func:`light_line_kw` / :func:`light_legend_handle`: comparison curves are
  drawn as light solid lines (dash patterns removed); markers keep the full
  condition colour, so marker shape and fill carry the identity.
* :func:`mix_colors` / :func:`difference_spec`: a paired difference A - B is
  drawn in the equal sRGB mix of the colours of A and B, with the marker of
  the comparator B, so the same contrast looks identical in every figure.
* :func:`lighten` and :func:`halo` (white outline for in-panel numbers).

Nothing in this module reads data, changes an aggregation rule, a statistical
estimate or a checkpoint policy.  Every public name of v1 is kept so the
existing builders continue to import it unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import shutil
import subprocess

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, MaxNLocator, NullFormatter

# ---------------------------------------------------------------------------
# Page geometry
# ---------------------------------------------------------------------------
MM = 1.0 / 25.4
#: Main-figure width.  174 mm = Cell Press / Neuron double column; it is also
#: inside the PLOS CB range (2.63-7.5 in) and eLife has no fixed width.  Set to
#: 183 * MM if the target is a Nature-family journal.
FIG_WIDTH = 174 * MM

# ---------------------------------------------------------------------------
# Colour contract.  Keys of v1 are preserved; values were re-tuned so that the
# conditions sharing a panel are separated in CIE L* (lightness) as well as in
# hue.  L* values are noted for the black-and-white check.
# ---------------------------------------------------------------------------
PALETTE = {
    "navigation": "#2E5FA8",       # L* 41  navigation-pretrained core (hero colour)
    "navigation_dark": "#1D4580",  # L* 30  fully trainable navigation-derived model
    "navigation_light": "#9DB8DF", # L* 74  light tint for bands / secondary nav marks
    "control": "#E8842C",          # L* 65  delayed-reconstruction (DR) control
    "relation": "#7B5EA7",         # L* 46  relation / coordinate-remapping accent
    "relation_dark": "#5F4A84",    # L* 36
    "relation_light": "#A58FCB",   # L* 62
    "random": "#8D9296",           # L* 60  random recurrent controls
    "matched": "#80858A",          # L* 55  matched random control (Phase F / mechanism)
    "fresh": "#B4B8BC",            # L* 75  fresh-random control
    "input": "#6FA8B8",            # L* 66  input-path / W_in manipulations
    "recurrent": "#3F7486",        # L* 46  core / W_rec manipulations
    "full": "#1D4580",
    "accumulator": "#7B5EA7",      # 2-D coordinate accumulator baseline
    "relation_count": "#4E9A95",   # L* 58  relation-count linear baseline
    "readout": "#7B5EA7",
    "positive": "#5F8F72",
    "negative": "#C0504D",         # L* 49  muted impairment accent (not saturated red)
    "negative_light": "#DE8A7C",   # L* 66  lighter impairment accent
    "neutral_light": "#D9DEE2",
    "neutral_lighter": "#EEF1F3",
    "neutral_mid": "#9AA0A6",
    "neutral_dark": "#555A60",
    "black": "#202124",
    "grid_fraction": "#7B5EA7",
}

# Line weights (pt).  Cell Press asks for >= 0.5 pt strokes; the hero line is
# heavier than comparison lines so that the key condition reads first.
LINEWIDTH = 1.45
LINEWIDTH_SECONDARY = 1.05
AXES_LINEWIDTH = 0.70
MARKERSIZE = 5.0            # was 4.2
ERROR_LINEWIDTH = 1.00
CAPSIZE = 2.0
BAND_ALPHA = 0.14
POINT_ALPHA = 0.55

# ---------------------------------------------------------------------------
# Typography (pt at 174 mm print width) and marks.
# ---------------------------------------------------------------------------
FS_LABEL = 7.0          # axis labels
FS_TICK = 6.5           # tick labels
FS_ANNOT = 6.3          # in-panel statistics / verdict annotations
FS_LEGEND = 6.3         # in-panel legends
FS_INPANEL = 6.3        # in-panel category or region labels
FS_TICK_DENSE = 6.3     # tick labels in dense forest-style panels
FS_MIN = 6.0            # absolute minimum anywhere in a main figure

MS_MEAN = MARKERSIZE    # marker on a mean curve
MS_SUMMARY = 6.2        # marker on a mean +/- 95% CI summary (was 4.6)
MS_SMALL = 4.6          # marker on a single-series reference curve

SOURCE_LW = 0.50        # per-source trajectory line
SOURCE_ALPHA = 0.16
PAIR_LW = 0.55          # within-source pairing line in categorical panels
PAIR_ALPHA = 0.30
SWARM_SIZE = 20         # scatter size (pt^2) of one source (was 13-15)
SWARM_ALPHA = 0.55

PANEL_LABEL_SIZE = 9.5
# v2.9: Nonlinear Dynamics (Springer) denotes figure parts by lowercase letters
# (a, b, c); set to "upper" to restore A, B, C.
PANEL_LETTER_CASE = "lower"


def panel_letter(label) -> str:
    """A panel letter in the journal case set by PANEL_LETTER_CASE."""
    s = str(label)
    return s.lower() if PANEL_LETTER_CASE == "lower" else s.upper()

# ---------------------------------------------------------------------------
# Condition registry: one condition = colour + marker + dash + fill + weight.
# Builders ask for a *role* instead of hand-picking colours so that the same
# condition looks identical in every figure (and in greyscale).
# ---------------------------------------------------------------------------
DASH = {
    "solid": "-",
    "dash": (0, (4.2, 1.8)),
    "dot": (0, (1.1, 1.5)),
    "dashdot": (0, (4.2, 1.5, 1.1, 1.5)),
    "longdash": (0, (7.0, 2.0)),
}

ROLE = {
    "navigation":      dict(color=PALETTE["navigation"], marker="o", ls=DASH["solid"], filled=True, lw=LINEWIDTH, ms_scale=1.00),
    "navigation_dark": dict(color=PALETTE["navigation_dark"], marker="o", ls=DASH["solid"], filled=True, lw=LINEWIDTH, ms_scale=1.00),
    "navigation_open": dict(color=PALETTE["navigation"], marker="o", ls=DASH["dash"], filled=False, lw=LINEWIDTH_SECONDARY, ms_scale=1.00),
    "dr":              dict(color=PALETTE["control"], marker="D", ls=DASH["dash"], filled=True, lw=LINEWIDTH_SECONDARY, ms_scale=0.88),
    "random":          dict(color=PALETTE["random"], marker="s", ls=DASH["dot"], filled=False, lw=LINEWIDTH_SECONDARY, ms_scale=0.92),
    "matched":         dict(color=PALETTE["matched"], marker="s", ls=DASH["dash"], filled=True, lw=LINEWIDTH_SECONDARY, ms_scale=0.92),
    "fresh":           dict(color=PALETTE["fresh"], marker="^", ls=DASH["dot"], filled=True, lw=LINEWIDTH_SECONDARY, ms_scale=1.02),
    "accumulator":     dict(color=PALETTE["accumulator"], marker="P", ls=DASH["dashdot"], filled=True, lw=LINEWIDTH_SECONDARY, ms_scale=1.05),
    "relation_count":  dict(color=PALETTE["relation_count"], marker="v", ls=DASH["longdash"], filled=True, lw=LINEWIDTH_SECONDARY, ms_scale=1.02),
    "input":           dict(color=PALETTE["input"], marker="^", ls=DASH["dash"], filled=True, lw=LINEWIDTH_SECONDARY, ms_scale=1.02),
    "recurrent":       dict(color=PALETTE["recurrent"], marker="D", ls=DASH["dashdot"], filled=True, lw=LINEWIDTH_SECONDARY, ms_scale=0.88),
}


def role(name: str, **overrides) -> dict:
    """Return a copy of a registered condition style (see :data:`ROLE`)."""
    spec = dict(ROLE[name])
    spec.update(overrides)
    return spec


def line_kw(spec: dict, *, ms: float | None = None, markevery=None, label=None,
            zorder: float = 3) -> dict:
    """Keyword arguments for ``ax.plot`` that realise a role spec."""
    size = (MARKERSIZE if ms is None else ms) * spec.get("ms_scale", 1.0)
    filled = spec.get("filled", True)
    return dict(
        color=spec["color"], ls=spec.get("ls", "-"), lw=spec.get("lw", LINEWIDTH),
        marker=spec.get("marker", "o"), ms=size,
        mfc=spec["color"] if filled else "white",
        mec="white" if filled else spec["color"],
        mew=0.55 if filled else 1.0,
        markevery=markevery, label=label, zorder=zorder,
        solid_capstyle="round", dash_capstyle="butt",
    )


def marker_kw(spec: dict, *, ms: float | None = None) -> dict:
    """Marker-only keyword arguments (for ``errorbar`` / legend handles)."""
    size = (MS_SUMMARY if ms is None else ms) * spec.get("ms_scale", 1.0)
    filled = spec.get("filled", True)
    return dict(
        marker=spec.get("marker", "o"), ms=size,
        mfc=spec["color"] if filled else "white",
        mec="white" if filled else spec["color"],
        mew=0.6 if filled else 1.05,
    )


def scatter_kw(spec: dict, *, s: float | None = None, alpha: float | None = None) -> dict:
    """Keyword arguments for ``ax.scatter`` of individual sources."""
    filled = spec.get("filled", True)
    return dict(
        s=(SWARM_SIZE if s is None else s) * spec.get("ms_scale", 1.0) ** 2,
        marker=spec.get("marker", "o"),
        facecolor=spec["color"] if filled else "white",
        edgecolor="white" if filled else spec["color"],
        linewidth=0.45 if filled else 0.85,
        alpha=POINT_ALPHA if alpha is None else alpha,
    )


def legend_handle(spec: dict, label: str, *, line: bool = True, ms: float | None = None):
    """A legend proxy that shows colour, marker, fill *and* dash pattern."""
    from matplotlib.lines import Line2D
    kw = line_kw(spec, ms=ms, label=label)
    if not line:
        kw["ls"] = "none"
    kw.pop("markevery", None)
    kw.pop("zorder", None)
    return Line2D([0], [0], **kw)


def lighten(color, amount: float = 0.5) -> str:
    """Blend ``color`` toward white (0 = unchanged, 1 = white)."""
    r, g, b = mpl.colors.to_rgb(color)
    a = float(amount)
    return mpl.colors.to_hex((r + (1 - r) * a, g + (1 - g) * a, b + (1 - b) * a))


def mix_colors(*colors, weights=None) -> str:
    """Weighted sRGB average of colours (equal weights by default)."""
    rgb = np.array([mpl.colors.to_rgb(c) for c in colors], dtype=float)
    w = np.ones(len(rgb)) if weights is None else np.asarray(weights, dtype=float)
    w = w / w.sum()
    return mpl.colors.to_hex(np.clip((rgb * w[:, None]).sum(axis=0), 0, 1))


def difference_spec(minuend: dict, subtrahend: dict, **overrides) -> dict:
    """Style of a paired difference ``A - B`` (A = ``minuend``, B = ``subtrahend``).

    Colour: equal sRGB mix of the two condition colours (e.g. navigation blue
    + DR orange -> grey-brown).  Marker: the comparator's (B) marker, always
    filled.  Used for every paired-difference panel (Figs 3C, 5B/C, 6A), so a
    contrast such as Nav. - DR looks the same wherever it appears.
    """
    spec = dict(minuend)
    spec.update(
        color=mix_colors(minuend["color"], subtrahend["color"]),
        marker=subtrahend.get("marker", "o"),
        ms_scale=subtrahend.get("ms_scale", 1.0),
        filled=True,
    )
    spec.update(overrides)
    return spec


#: How far the connecting line of a comparison curve is lightened toward white.
LIGHT_LINE_AMOUNT = 0.45


def light_line_kw(spec: dict, *, ms: float | None = None, markevery=None, label=None,
                  zorder: float = 3, amount: float = LIGHT_LINE_AMOUNT) -> dict:
    """Like :func:`line_kw`, but the line is a light *solid* tint of the colour.

    Markers keep the full colour (filled or open as in the role), so the
    marker shape carries the condition and no dash pattern is needed.
    """
    kw = line_kw(spec, ms=ms, markevery=markevery, label=label, zorder=zorder)
    kw["color"] = lighten(spec["color"], amount)   # mfc / mec stay at full colour
    kw["ls"] = "-"
    return kw


def light_legend_handle(spec: dict, label: str, *, ms: float | None = None,
                        amount: float = LIGHT_LINE_AMOUNT):
    """Legend proxy matching :func:`light_line_kw`."""
    from matplotlib.lines import Line2D
    kw = light_line_kw(spec, ms=ms, label=label, amount=amount)
    kw.pop("markevery", None)
    kw.pop("zorder", None)
    return Line2D([0], [0], **kw)


def halo(width: float = 2.2, color: str = "white") -> list:
    """Path effect that keeps in-panel text legible over thin lines."""
    import matplotlib.patheffects as pe
    return [pe.withStroke(linewidth=width, foreground=color)]


def apply_publication_style() -> None:
    """Apply the single manuscript-wide plotting contract."""
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Arimo", "Liberation Sans", "DejaVu Sans", "sans-serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Arial:regular",
        "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold",
        "mathtext.sf": "Arial:regular",
        "mathtext.fallback": "stixsans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": FS_LABEL,
        "axes.titlesize": FS_LABEL,
        "axes.titleweight": "regular",
        "axes.labelsize": FS_LABEL,
        "axes.labelpad": 2.5,
        "axes.linewidth": AXES_LINEWIDTH,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": PALETTE["black"],
        "axes.labelcolor": PALETTE["black"],
        "axes.titlecolor": PALETTE["black"],
        "axes.xmargin": 0.03,
        "axes.ymargin": 0.05,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.color": PALETTE["black"],
        "ytick.color": PALETTE["black"],
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": AXES_LINEWIDTH,
        "ytick.major.width": AXES_LINEWIDTH,
        "xtick.minor.width": 0.55,
        "ytick.minor.width": 0.55,
        "xtick.major.size": 2.8,
        "ytick.major.size": 2.8,
        "xtick.minor.size": 1.6,
        "ytick.minor.size": 1.6,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "legend.fontsize": FS_LEGEND,
        "legend.frameon": False,
        "legend.handlelength": 1.9,
        "legend.handletextpad": 0.45,
        "legend.columnspacing": 0.9,
        "legend.labelspacing": 0.30,
        "lines.linewidth": LINEWIDTH,
        "lines.markersize": MARKERSIZE,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": PALETTE["black"],
    })
    # Fall back gracefully when Arial is not installed (e.g. Linux build hosts):
    # use the metric-compatible Liberation Sans / Arimo for mathtext as well.
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Arial" not in names:
        for alt in ("Arimo", "Liberation Sans", "DejaVu Sans"):
            if alt in names:
                mpl.rcParams.update({
                    "mathtext.rm": f"{alt}:regular", "mathtext.it": f"{alt}:italic",
                    "mathtext.bf": f"{alt}:bold", "mathtext.sf": f"{alt}:regular",
                })
                break


def add_panel_label(ax: Axes, label: str, *, x: float = -0.13, y: float = 1.04) -> None:
    """Add a manuscript panel label (case set by PANEL_LETTER_CASE)."""
    ax.text(
        x, y, panel_letter(label), transform=ax.transAxes,
        fontsize=PANEL_LABEL_SIZE, fontweight="bold", color=PALETTE["black"],
        ha="left", va="bottom", clip_on=False,
    )


def style_axis(
    ax: Axes,
    *,
    max_y_ticks: int = 5,
    max_x_ticks: int | None = None,
    zero_baseline: bool = False,
) -> None:
    """Apply unobtrusive axes and restrained tick density."""
    ax.spines["left"].set_linewidth(AXES_LINEWIDTH)
    ax.spines["bottom"].set_linewidth(AXES_LINEWIDTH)
    ax.spines["left"].set_color(PALETTE["black"])
    ax.spines["bottom"].set_color(PALETTE["black"])
    ax.tick_params(axis="both", which="major", width=AXES_LINEWIDTH, length=2.8)
    if ax.get_yscale() == "linear" and not isinstance(ax.yaxis.get_major_locator(), FixedLocator):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=max(2, max_y_ticks - 1), min_n_ticks=3))
    if max_x_ticks and ax.get_xscale() == "linear" and not isinstance(ax.xaxis.get_major_locator(), FixedLocator):
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max(2, max_x_ticks - 1), min_n_ticks=3))
    if zero_baseline:
        ax.axhline(0.0, color=PALETTE["neutral_mid"], lw=0.8, zorder=0)


def reference_line(ax: Axes, value: float, *, orientation: str = "h", zorder: float = 0.5,
                   label: str | None = None, label_pos: float = 0.985,
                   label_side: str = "above") -> None:
    """Draw a manuscript-wide chance/threshold/reference line (optionally labelled)."""
    kw = dict(color=PALETTE["neutral_mid"], ls=(0, (3, 2.2)), lw=0.8, zorder=zorder)
    if orientation == "h":
        ax.axhline(value, **kw)
        if label:
            ax.annotate(
                label, xy=(label_pos, value), xycoords=ax.get_yaxis_transform(),
                xytext=(0, 1.2 if label_side == "above" else -1.2), textcoords="offset points",
                ha="right", va="bottom" if label_side == "above" else "top",
                fontsize=FS_MIN, color=PALETTE["neutral_dark"], zorder=zorder,
            )
    else:
        ax.axvline(value, **kw)
        if label:
            ax.annotate(
                label, xy=(value, label_pos), xycoords=ax.get_xaxis_transform(),
                xytext=(1.5, 0), textcoords="offset points", ha="left", va="top",
                fontsize=FS_MIN, color=PALETTE["neutral_dark"], rotation=90, zorder=zorder,
            )


def set_fixed_y_ticks(ax: Axes, ticks: Sequence[float]) -> None:
    if len(ticks) > 5:
        raise ValueError("Manuscript y axes are limited to at most five major ticks.")
    ax.set_yticks(ticks)


def mean_ci95(values: Iterable[float]) -> tuple[float, float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    if len(values) < 2:
        return mean, mean, mean
    from scipy import stats
    sem = float(values.std(ddof=1) / np.sqrt(len(values)))
    q = float(stats.t.ppf(0.975, len(values) - 1))
    return mean, mean - q * sem, mean + q * sem


def mean_sem(values: Iterable[float]) -> tuple[float, float]:
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), 0.0
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(values.size))


def deterministic_jitter(n: int, *, width: float = 0.10) -> np.ndarray:
    """Symmetric deterministic jitter: reproducible without hiding observations."""
    if n <= 1:
        return np.zeros(n, dtype=float)
    base = np.linspace(-width, width, n)
    # Interleave left/right so input order does not visually imply rank.
    order = np.ravel(np.column_stack((np.arange((n + 1) // 2), np.arange(n - 1, n // 2 - 1, -1))))
    order = order[order < n][:n]
    return base[order]


def beeswarm_offsets(values, *, width: float = 0.16, resolution: float | None = None) -> np.ndarray:
    """Deterministic 1-D beeswarm: overlapping values spread sideways, others stay put.

    Unlike uniform jitter, a point only moves sideways when it would overlap a
    point already placed, so a tight cluster reads as a cluster and isolated
    values stay on the centre line.  ``resolution`` is the value-axis distance
    treated as an overlap (default: 3 % of the data range).
    """
    v = np.asarray(values, dtype=float)
    n = v.size
    if n <= 1:
        return np.zeros(n)
    span = float(np.nanmax(v) - np.nanmin(v))
    res = resolution if resolution is not None else max(span * 0.03, 1e-12)
    order = np.argsort(v, kind="mergesort")
    offsets = np.zeros(n)
    placed: list[tuple[float, float]] = []
    step = width / 3.0
    candidates = [0.0]
    for k in range(1, 12):
        candidates += [k * step, -k * step]
    for idx in order:
        for c in candidates:
            if all(abs(v[idx] - pv) >= res or abs(c - po) >= step * 0.999 for pv, po in placed):
                offsets[idx] = float(np.clip(c, -width, width))
                break
        placed.append((v[idx], offsets[idx]))
    return offsets


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------
_RANGE_FRAME_ATTR = "_ctp_range_frame"


def range_frame(ax: Axes, *, x: bool = True, y: bool = True, offset: float = 3.0,
                trim_x: bool | None = None, trim_y: bool | None = None) -> None:
    """Offset the left/bottom spines outward and trim them to the tick range.

    Trimming is applied when the figure is saved (after every tick has been
    fixed), see :func:`save_publication_figure`.
    """
    if x:
        ax.spines["bottom"].set_position(("outward", offset))
    if y:
        ax.spines["left"].set_position(("outward", offset))
    setattr(ax, _RANGE_FRAME_ATTR, {
        "x": x if trim_x is None else trim_x,
        "y": y if trim_y is None else trim_y,
    })


def _trim_spines(fig: Figure) -> None:
    for ax in fig.axes:
        spec = getattr(ax, _RANGE_FRAME_ATTR, None)
        if not spec:
            continue
        for axis_name, spine_name in (("x", "bottom"), ("y", "left")):
            if not spec.get(axis_name):
                continue
            axis = ax.xaxis if axis_name == "x" else ax.yaxis
            lo, hi = sorted(ax.get_xlim() if axis_name == "x" else ax.get_ylim())
            # Major *and* minor ticks: an integer axis labelled 1, 3, ..., 11 must
            # still run to its unlabelled tick at 12.
            locs = list(axis.get_majorticklocs()) + list(axis.get_minorticklocs())
            ticks = [t for t in locs if lo - 1e-9 <= t <= hi + 1e-9]
            if len(ticks) >= 2:
                ax.spines[spine_name].set_bounds(min(ticks), max(ticks))


#: Main figures: also draw the top and right spines (a full box) when True.
#: False keeps the classic L-shaped frame whose two axes meet in one corner.
CLOSED_FRAME_BOX = False


def closed_frame(ax: Axes, *, box: bool | None = None) -> None:
    """Closed frame: the x and y axes intersect in one corner point.

    The left and bottom spines sit on the axes edge (no outward offset) and
    span the full axis limits (no trimming), so they always meet at
    ``(xlim[0], ylim[0])`` and every plotted value lies inside the frame.
    Tick marks are restored on categorical axes as well.  Values at 0 are kept
    off the frame by starting the limit slightly below 0 (as
    :func:`accuracy_axis` does), not by moving the spine.
    """
    box = CLOSED_FRAME_BOX if box is None else bool(box)
    for name in ("left", "bottom", "top", "right"):
        sp = ax.spines[name]
        visible = name in ("left", "bottom") or box
        sp.set_visible(visible)
        if visible:
            sp.set_position(("outward", 0.0))
            sp.set_linewidth(AXES_LINEWIDTH)
            sp.set_color(PALETTE["black"])
            sp.set_capstyle("projecting")   # square corner where the axes meet
    setattr(ax, _RANGE_FRAME_ATTR, None)    # never trimmed at save time
    ax.tick_params(axis="both", which="major", length=2.8, width=AXES_LINEWIDTH, pad=2.0)
    ax.tick_params(axis="both", which="minor", length=1.6, width=0.55)


def axis_break_marks(ax_low: Axes, ax_high: Axes, *, size: float = 5.0,
                     slope: float = 0.55) -> None:
    """Diagonal '//' marks on the left spine where a y axis is broken.

    ``ax_low`` holds the lower value range, ``ax_high`` the upper one; both
    share the x axis and use the same data-per-inch scale.
    """
    kw = dict(marker=[(-1, -slope), (1, slope)], markersize=size, linestyle="none",
              color=PALETTE["black"], mec=PALETTE["black"], mew=AXES_LINEWIDTH,
              clip_on=False, zorder=10)
    ax_low.plot([0], [1], transform=ax_low.transAxes, **kw)
    ax_high.plot([0], [0], transform=ax_high.transAxes, **kw)


def equalize_column_gaps(fig: Figure, axes: Sequence[Axes], *, gap_in: float | None = None,
                         left_in: float | None = None, right_in: float | None = None,
                         partners: dict | None = None, iterations: int = 4) -> float:
    """Re-space one row of panels so the white space between neighbours is identical.

    The white space is measured ink to ink: from the right edge of a panel's
    tight bounding box (tick labels, axis labels, in-axes legends and
    annotations included) to the left edge of the next panel's.  The leftmost
    and rightmost ink keep their current positions unless ``left_in`` /
    ``right_in`` (inches) are given.  With ``gap_in=None`` the panel widths are
    kept and the available white space is shared equally; with a fixed
    ``gap_in`` the widths are scaled by one common factor instead.
    ``partners`` maps a panel to further axes that share its x geometry (for
    example the upper half of a broken axis); they move with it.
    Returns the common gap in inches.
    """
    from matplotlib.transforms import Bbox
    axes = list(axes)
    partners = partners or {}
    if len(axes) < 2:
        return 0.0
    width_in = fig.get_figwidth()
    gap = 0.0
    for _ in range(max(1, iterations)):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        px = fig.dpi
        lefts, rights, widths, inks = [], [], [], []
        for ax in axes:
            members = [ax, *partners.get(ax, [])]
            ink = Bbox.union([m.get_tightbbox(renderer) for m in members])
            pos = ax.get_position()
            x0, x1 = pos.x0 * width_in, pos.x1 * width_in
            lefts.append(x0 - ink.x0 / px)
            rights.append(ink.x1 / px - x1)
            widths.append(x1 - x0)
            inks.append((ink.x0 / px, ink.x1 / px))
        lo = inks[0][0] if left_in is None else float(left_in)
        hi = inks[-1][1] if right_in is None else float(right_in)
        fixed = sum(lefts) + sum(rights)
        if gap_in is None:
            scale = 1.0
            gap = (hi - lo - fixed - sum(widths)) / (len(axes) - 1)
        else:
            gap = float(gap_in)
            scale = (hi - lo - fixed - (len(axes) - 1) * gap) / sum(widths)
        x = lo
        for ax, left, right, w in zip(axes, lefts, rights, widths):
            x0 = x + left
            w_new = w * scale
            for member in (ax, *partners.get(ax, [])):
                p = member.get_position()
                member.set_position([x0 / width_in, p.y0, w_new / width_in, p.height])
            x = x0 + w_new + right + gap
    return gap


def accuracy_axis(ax: Axes, which: str = "y", *, lo: float = -0.06, hi: float = 1.04,
                  major: Sequence[float] = (0.0, 0.5, 1.0),
                  minor: Sequence[float] = (0.25, 0.75), label: str | None = None) -> None:
    """Shared accuracy scale: starts just below 0 so floor-level points stay visible."""
    if which == "y":
        ax.set_ylim(lo, hi)
        ax.set_yticks(list(major))
        ax.set_yticks(list(minor), minor=True)
        if label is not None:
            ax.set_ylabel(label)
    else:
        ax.set_xlim(lo, hi)
        ax.set_xticks(list(major))
        ax.set_xticks(list(minor), minor=True)
        if label is not None:
            ax.set_xlabel(label)


def integer_axis(ax: Axes, which: str = "x", *, lo: int = 1, hi: int = 12, step: int = 2,
                 pad: float = 0.45, label: str | None = None) -> None:
    """Integer axis with labelled ticks every ``step`` and unlabelled minor ticks."""
    major = list(range(lo, hi + 1, step))
    minor = [v for v in range(lo, hi + 1) if v not in major]
    if which == "x":
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_xticks(major)
        ax.set_xticks(minor, minor=True)
        ax.xaxis.set_minor_formatter(NullFormatter())
        if label is not None:
            ax.set_xlabel(label)
    else:
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_yticks(major)
        ax.set_yticks(minor, minor=True)
        ax.yaxis.set_minor_formatter(NullFormatter())
        if label is not None:
            ax.set_ylabel(label)


def hide_y_axis(ax: Axes) -> None:
    """Forest-style panels: keep the category labels, drop the vertical spine."""
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", which="both", length=0, pad=3)


def save_publication_figure(
    fig: Figure,
    output_stem: Path,
    *,
    png_dpi: int = 600,
    close: bool = True,
) -> dict[str, Path]:
    """Save editable vector masters plus high-resolution raster derivatives."""
    _trim_spines(fig)
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "svg": output_stem.with_suffix(".svg"),
        "pdf": output_stem.with_suffix(".pdf"),
        "png": output_stem.with_suffix(".png"),
        "tiff": output_stem.with_suffix(".tiff"),
    }
    common = {"bbox_inches": "tight", "pad_inches": 0.03, "facecolor": "white"}
    fig.savefig(paths["svg"], **common)
    fig.savefig(paths["pdf"], **common)
    fig.savefig(paths["png"], dpi=png_dpi, **common)
    from PIL import Image
    with Image.open(paths["png"]) as im:
        im.convert("RGB").save(paths["tiff"], compression="tiff_lzw", dpi=(png_dpi, png_dpi))
    pdftops = shutil.which("pdftops")
    if pdftops:
        eps = output_stem.with_suffix(".eps")
        subprocess.run([pdftops, "-eps", "-level3", str(paths["pdf"]), str(eps)], check=True, timeout=60)
        paths["eps"] = eps
    if close:
        plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Shared presentation contract for every manuscript figure.
# ---------------------------------------------------------------------------
# Legends are frameless, tight, and identical everywhere.
LEGEND_KW = dict(
    fontsize=FS_LEGEND, frameon=False, handlelength=2.1, handletextpad=0.45,
    labelspacing=0.32, columnspacing=0.9, borderaxespad=0.25, borderpad=0.15,
)
# Marker-only legends (categorical panels) need no line-length allowance.
LEGEND_KW_POINTS = dict(LEGEND_KW, handlelength=0.9, handletextpad=0.40)

# In-panel annotation anchors: loc -> (x, y, ha, va) in axes fraction.
ANNOT_ANCHORS = {
    "upper left": (0.025, 0.975, "left", "top"),
    "upper right": (0.975, 0.975, "right", "top"),
    "lower left": (0.025, 0.025, "left", "bottom"),
    "lower right": (0.975, 0.025, "right", "bottom"),
    "upper center": (0.500, 0.975, "center", "top"),
    "lower center": (0.500, 0.025, "center", "bottom"),
    "center left": (0.025, 0.500, "left", "center"),
    "center right": (0.975, 0.500, "right", "center"),
}

MINUS = "\u2212"


def num(value: float, fmt: str = "{:.2f}") -> str:
    """Format a number with a typographic minus sign."""
    return fmt.format(float(value)).replace("-", MINUS)


def sci(value: float, digits: int = 1) -> str:
    r"""Scientific notation as mathtext, e.g. 5e-05 -> $5.0\times10^{-5}$."""
    value = float(value)
    if value == 0 or not np.isfinite(value):
        return num(value, "{:.3g}")
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0 ** exponent)
    return rf"${mantissa:.{digits}f}\times10^{{{exponent}}}$"


def fmt_p(p: float, *, prefix: str = "", digits: int = 3) -> str:
    """Journal-style P value: italic P, three decimals, '< 0.001' floor."""
    p = float(p)
    lead = f"{prefix} " if prefix else ""
    if not np.isfinite(p):
        return f"{lead}$P$ = n/a"
    if p < 0.001:
        return f"{lead}$P$ < 0.001"
    return f"{lead}$P$ = {p:.{digits}f}"


def fmt_ci(mean: float, lo: float, hi: float, fmt: str = "{:.2f}") -> str:
    """'0.72 [0.70, 0.74]' with typographic minus signs."""
    return f"{num(mean, fmt)} [{num(lo, fmt)}, {num(hi, fmt)}]"


def annotate(ax: Axes, text: str, *, loc: str = "upper right",
             color: str | None = None, fontsize: float = FS_ANNOT, **kwargs):
    """In-panel statistics with one anchor convention across every figure."""
    x, y, ha, va = ANNOT_ANCHORS[loc]
    return ax.text(
        kwargs.pop("x", x), kwargs.pop("y", y), text, transform=ax.transAxes,
        ha=kwargs.pop("ha", ha), va=kwargs.pop("va", va), fontsize=fontsize,
        color=PALETTE["black"] if color is None else color,
        linespacing=1.3, zorder=6, **kwargs,
    )


def legend(ax: Axes, *, loc: str, points: bool = False, handles=None, **kwargs):
    """Frameless legend with the shared geometry; ``points`` for marker-only keys."""
    base = dict(LEGEND_KW_POINTS if points else LEGEND_KW)
    base.update(kwargs)
    if handles is not None:
        return ax.legend(handles=handles, loc=loc, **base)
    return ax.legend(loc=loc, **base)


def categorical_xlim(ax: Axes, positions, *, pad: float = 0.62) -> None:
    """Keep categorical clusters off the spines."""
    positions = np.asarray(positions, dtype=float)
    step = float(np.min(np.diff(np.sort(positions)))) if positions.size > 1 else 1.0
    ax.set_xlim(positions.min() - pad * step, positions.max() + pad * step)


def categorical_ylim(ax: Axes, positions, *, pad: float = 0.62) -> None:
    """Vertical counterpart of :func:`categorical_xlim` for forest-style panels."""
    positions = np.asarray(positions, dtype=float)
    step = float(np.min(np.diff(np.sort(positions)))) if positions.size > 1 else 1.0
    lo, hi = positions.min() - pad * step, positions.max() + pad * step
    bottom, top = ax.get_ylim()
    ax.set_ylim((hi, lo) if bottom > top else (lo, hi))


def significance_bracket(ax: Axes, x1: float, x2: float, y: float, text: str, *,
                         h: float = 0.025, color: str | None = None,
                         fontsize: float = FS_ANNOT, lw: float = 0.7,
                         text_pad_pt: float = 0.8) -> None:
    """Classic bracket between two categories with the statistic above it.

    ``y`` and ``h`` are in data units of the y axis.
    """
    c = PALETTE["black"] if color is None else color
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=c, lw=lw,
            clip_on=False, zorder=6, solid_capstyle="butt")
    ax.annotate(text, xy=((x1 + x2) / 2, y + h), xytext=(0, text_pad_pt),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=fontsize, color=c, annotation_clip=False, zorder=6)


def end_labels(ax: Axes, items, *, x: float, dx_pt: float = 3.0, min_gap_frac: float = 0.075,
               fontsize: float = FS_INPANEL, ylim: tuple[float, float] | None = None) -> list:
    """Direct labels at the right end of curves, nudged apart when they collide.

    ``items`` is a sequence of ``(y, text, color)``; ``x`` is the data x at
    which the labels are anchored (normally the last x of the curves).
    ``min_gap_frac`` is the minimum vertical separation as a fraction of the
    y range.
    """
    lo, hi = ylim if ylim is not None else ax.get_ylim()
    gap = min_gap_frac * (hi - lo)
    order = sorted(range(len(items)), key=lambda i: items[i][0])
    ys = [float(items[i][0]) for i in order]
    for _ in range(60):  # simple relaxation
        moved = False
        for k in range(1, len(ys)):
            if ys[k] - ys[k - 1] < gap - 1e-12:
                mid = 0.5 * (ys[k] + ys[k - 1])
                ys[k - 1], ys[k] = mid - gap / 2, mid + gap / 2
                moved = True
        if not moved:
            break
    texts = []
    for k, i in enumerate(order):
        _, text, color = items[i]
        texts.append(ax.annotate(
            text, xy=(x, ys[k]), xytext=(dx_pt, 0), textcoords="offset points",
            ha="left", va="center", fontsize=fontsize, color=color,
            annotation_clip=False, zorder=7,
        ))
    return texts


def t_density(ax: Axes, position: float, mean: float, sem: float, df: int, *,
              color: str, height: float = 0.32, orientation: str = "h",
              alpha: float = 0.22, span_sd: float = 4.0, zorder: float = 1.5) -> None:
    """Half-violin of the t-based sampling distribution of a mean difference.

    Drawn above (``orientation='h'``) a row at ``position``.  Its 2.5-97.5 %
    quantiles coincide with the t-based 95 % CI stored in the frozen contrast
    tables, so the shape and the CI bar tell the same story.
    """
    from scipy import stats
    if not (np.isfinite(mean) and np.isfinite(sem) and sem > 0 and df >= 1):
        return
    grid = np.linspace(mean - span_sd * sem, mean + span_sd * sem, 241)
    dens = stats.t.pdf((grid - mean) / sem, df)
    dens = dens / dens.max() * height
    if orientation == "h":
        ax.fill_between(grid, position, position + dens, color=color, alpha=alpha,
                        lw=0, zorder=zorder)
        ax.plot(grid, position + dens, color=color, lw=0.6, alpha=0.9, zorder=zorder)
    else:
        ax.fill_betweenx(grid, position, position + dens, color=color, alpha=alpha,
                         lw=0, zorder=zorder)
        ax.plot(position + dens, grid, color=color, lw=0.6, alpha=0.9, zorder=zorder)


def group_brackets(ax: Axes, groups, *, gap_pt: float = 6.0, label_gap_pt: float = 3.5,
                   fontsize: float = FS_INPANEL, color: str | None = None):
    """Thin bracket plus one rotated group name beside a block of forest rows."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_px = max(ax.get_window_extent(renderer).width, 1.0)
    pt = fig.dpi / 72.0
    widths = [t.get_window_extent(renderer).width
              for t in ax.get_yticklabels() if t.get_text()]
    label_width = (max(widths) if widths else 0.0) / axes_px
    x = -(label_width + gap_pt * pt / axes_px)
    label_x = x - (label_gap_pt + fontsize * 0.62) * pt / axes_px
    trans = ax.get_yaxis_transform()
    for text, first, last in groups:
        lo, hi = sorted((float(first), float(last)))
        ax.plot([x, x], [lo - .28, hi + .28], transform=trans,
                color=PALETTE["neutral_mid"], lw=.7, clip_on=False, zorder=2)
        ax.text(label_x, (lo + hi) / 2, text, transform=trans, rotation=90,
                ha="center", va="center", fontsize=fontsize,
                color=PALETTE["black"] if color is None else color, clip_on=False)


def place_panel_labels(fig: Figure, items, *, dy_pt: float = 2.0,
                       dx_pt: float = 0.0, fontsize: float = PANEL_LABEL_SIZE,
                       align=()):
    """Flush-left panel letters aligned to each panel's own left-most ink.

    ``items`` is a sequence of ``(axes, letter)`` or ``(axes, letter, options)``.
    Options: ``dx_pt`` / ``dy_pt`` (extra offsets), ``y_frac`` (axes-fraction
    height instead of the axes top) and ``x_fig`` / ``y_fig`` (explicit figure
    fraction overrides).  ``align`` lists groups of letters that must share one x.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    pt = fig.dpi / 72.0
    texts, placed = [], []
    for item in items:
        ax, letter = item[0], item[1]
        opts = item[2] if len(item) > 2 else {}
        pos = ax.get_position()
        bbox = ax.get_tightbbox(renderer)
        x = bbox.x0 / width + (dx_pt + opts.get("dx_pt", 0.0)) * pt / width
        if "y_frac" in opts:
            y = pos.y0 + opts["y_frac"] * pos.height
        else:
            y = pos.y1
        y += (dy_pt + opts.get("dy_pt", 0.0)) * pt / height
        placed.append({
            "letter": panel_letter(letter), "x": opts.get("x_fig", x),
            "y": opts.get("y_fig", y), "col": pos.x0,
            "snap": "x_fig" not in opts,
        })

    gap = 7.0 * pt / width
    boxes = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        pos = ax.get_position()
        right = pos.x1
        for title in (ax.title, getattr(ax, "_left_title", None),
                      getattr(ax, "_right_title", None)):
            if title is not None and title.get_text():
                right = max(right, title.get_window_extent(renderer).x1 / width)
        boxes.append((pos, right))
    for item, entry in zip(placed, items):
        me = entry[0].get_position()
        floor = 0.0
        for pos, right in boxes:
            same_row = min(pos.y1, me.y1) - max(pos.y0, me.y0) > 0.25 * me.height
            if same_row and pos.x1 <= me.x0 + 1e-6 and right < me.x0:
                floor = max(floor, right + gap)
        item["floor"] = floor

    snap_tol = 0.012
    for item in placed:
        if not item["snap"]:
            item["x_col"] = max(item["x"], item["floor"])
            continue
        column = [o for o in placed
                  if o["snap"] and abs(o["col"] - item["col"]) < snap_tol]
        item["x_col"] = max(min(o["x"] for o in column),
                            max(o["floor"] for o in column))
    for group in align:
        members = [o for o in placed if o["letter"] in {panel_letter(g) for g in group}]
        if len(members) > 1:
            x_shared = max(min(o["x_col"] for o in members),
                           max(o["floor"] for o in members))
            for o in members:
                o["x_col"] = x_shared
    for item in placed:
        texts.append(fig.text(
            item.get("x_col", item["x"]), item["y"], item["letter"],
            fontsize=fontsize, fontweight="bold", color=PALETTE["black"],
            ha="left", va="bottom",
        ))
    return texts


# ---------------------------------------------------------------------------
# Composite primitives shared by every figure (array in, ink out).
# ---------------------------------------------------------------------------
def swarm(ax: Axes, position: float, values, *, spec: dict, width: float = 0.13,
          orientation: str = "v", s: float | None = None, alpha: float | None = None,
          zorder: float = 3, offsets=None):
    """Individual sources as a deterministic beeswarm around ``position``.

    Returns the sideways offsets so that paired lines can reuse them.
    """
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if offsets is None:
        offsets = np.zeros(vals.size)
        offsets[finite] = beeswarm_offsets(vals[finite], width=width)
    kw = scatter_kw(spec, s=s, alpha=alpha)
    if orientation == "v":
        ax.scatter(position + offsets[finite], vals[finite], zorder=zorder, **kw)
    else:
        ax.scatter(vals[finite], position + offsets[finite], zorder=zorder, **kw)
    return offsets


def summary_errorbar(ax: Axes, position: float, values=None, *, spec: dict,
                     mean: float | None = None, lo: float | None = None, hi: float | None = None,
                     orientation: str = "v", ms: float | None = None, zorder: float = 6,
                     label: str | None = None, capsize: float = CAPSIZE,
                     ecolor: str | None = None):
    """Mean and 95 % CI drawn with the condition's own marker (not a generic diamond)."""
    if mean is None:
        mean, lo, hi = mean_ci95(values)
    if not np.isfinite(mean):
        return mean, lo, hi
    err = None
    if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
        err = [[mean - lo], [hi - mean]]
    mk = marker_kw(spec, ms=ms)
    # Summary markers get a dark outline so they separate from the source cloud
    # in colour *and* in greyscale.
    if spec.get("filled", True):
        mk["mec"] = PALETTE["black"]
        mk["mew"] = 0.7
    else:
        mk["mew"] = 1.35
    col = spec["color"] if ecolor is None else ecolor
    if orientation == "v":
        ax.errorbar(position, mean, yerr=err, fmt="none", ecolor=col,
                    elinewidth=ERROR_LINEWIDTH + 0.2, capsize=capsize, capthick=ERROR_LINEWIDTH,
                    zorder=zorder - 0.1)
        ax.plot([position], [mean], ls="none", zorder=zorder, label=label, **mk)
    else:
        ax.errorbar(mean, position, xerr=err, fmt="none", ecolor=col,
                    elinewidth=ERROR_LINEWIDTH + 0.2, capsize=capsize, capthick=ERROR_LINEWIDTH,
                    zorder=zorder - 0.1)
        ax.plot([mean], [position], ls="none", zorder=zorder, label=label, **mk)
    return mean, lo, hi


def pair_lines(ax: Axes, xs, ys, *, color: str | None = None, lw: float = PAIR_LW,
               alpha: float = PAIR_ALPHA, zorder: float = 1.5, orientation: str = "v") -> None:
    """Faint within-source connectors (``xs``/``ys`` are sequences of arrays)."""
    xs = [np.asarray(x, dtype=float) for x in xs]
    ys = [np.asarray(y, dtype=float) for y in ys]
    c = PALETTE["neutral_mid"] if color is None else color
    for j in range(len(ys[0])):
        px = [x[j] for x in xs]
        py = [y[j] for y in ys]
        if np.isfinite(py).sum() >= 2:
            if orientation == "v":
                ax.plot(px, py, color=c, lw=lw, alpha=alpha, zorder=zorder)
            else:
                ax.plot(py, px, color=c, lw=lw, alpha=alpha, zorder=zorder)


def estimation_row(ax: Axes, y: float, diffs, *, spec: dict, mean: float, lo: float, hi: float,
                   sem: float | None = None, df: int | None = None, density_height: float = 0.30,
                   point_offset: float = -0.20, swarm_width: float = 0.07, ms: float | None = None):
    """One row of a horizontal estimation (raincloud) plot.

    t-based sampling density above the row, the individual paired source
    differences below it, and the mean with its 95 % CI on the row itself.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if sem is None and diffs.size > 1:
        sem = float(diffs.std(ddof=1) / np.sqrt(diffs.size))
    if df is None:
        df = max(int(diffs.size) - 1, 1)
    if density_height > 0 and sem is not None and np.isfinite(sem) and sem > 0:
        t_density(ax, y + 0.06, mean, sem, df, color=spec["color"], height=density_height)
    if diffs.size:
        swarm(ax, y + point_offset, diffs, spec=spec, width=swarm_width,
              orientation="h", s=SWARM_SIZE * 0.85)
    summary_errorbar(ax, y, spec=spec, mean=mean, lo=lo, hi=hi, orientation="h", ms=ms)


def stars(p: float, *, ns: str = "n.s.") -> str:
    """Significance mark for brackets: *** < 0.001 <= ** < 0.01 <= * < 0.05 <= n.s."""
    p = float(p)
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ns
