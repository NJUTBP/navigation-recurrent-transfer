#!/usr/bin/env python3
"""Figure 2: navigation pretraining and the representations of the selected checkpoints.

Reads figure_source_data/Figure2 (navigation_history.csv, the 20-step trajectory candidates,
rate_map_rank_{2..5}_2500.csv and representative_grid_scores.csv).  Panels: a path-integration
error eps_pos and b grid-like unit fraction f_G over training (all sources, mean +/- SEM,
validation-best checkpoints marked); c the representative decoded trajectory with per-step
decoding error and a 10-cm scale bar; d the grid-score distribution of every unit of the
representative source on a broken y axis (the G > 0.3 criterion is drawn only if it reproduces f_G
of navigation_history.csv); e the units ranked 2-5: spatial autocorrelograms (top, with G) and rate
maps (bottom).  The representative rollout is the candidate with the lowest mean decoding error in
trajectory_candidates_20step/trajectory_candidates_summary.csv (final error and trajectory id break
ties).
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import convolve, gaussian_filter

from .common import FigureContext, empty_panel, load_csv, manifest_for_figure, write_json
from .nature_style import (
    FIG_WIDTH, BAND_ALPHA, LINEWIDTH, PALETTE, FS_MIN, FS_INPANEL, FS_ANNOT,
    place_panel_labels, legend, apply_publication_style, save_publication_figure,
    style_axis, closed_frame, axis_break_marks, equalize_column_gaps,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

#: Epoch axis of A/B: labelled powers of four from 1 up to the first one that
#: reaches the last epoch (150 epochs -> 1, 4, 16, 64, 256), unlabelled minor
#: ticks at the powers of two in between.  The axis starts just below epoch 1
#: so the first-epoch values do not sit on the y axis.
_EPOCH_AXIS_START = 0.8
_RATE_MAP_DISPLAY_SIGMA_BINS = 0.75
_RATE_MAP_DISPLAY_TRUNCATE = 2.5
#: Grid-score criterion drawn in D.  It is only drawn when it reproduces the
#: frozen grid-like fraction f_G of the representative source (checked below).
GRID_THRESHOLD = 0.30
#: Spatial autocorrelograms in E follow Savelli et al. (2017) at map level.
#: Full lag range; grid scores shown remain the frozen values.
SHOW_AUTOCORRELOGRAMS = True
SAC_MIN_OVERLAP = 100
RATE_MAP_RANKS = (2, 3, 4, 5)
#: Physical scale of the E maps: 50 x 50 bins of the 2.2 m x 2.2 m arena
#: (Section 2.1; Supplementary Methods S3).  Used only for the coordinate glyph.
ARENA_SIDE_M = 2.2
#: Arm length (m) of the coordinate glyph in each row of E.  Autocorrelogram
#: lags span twice the arena, so 1 m there is drawn as long as 0.5 m in a rate map.
GLYPH_ARM_M = {"sac": 1.0, "map": 0.5}
RANK_COLORS = ["#1D4580", "#2E5FA8", "#5B84C4", "#8FA9D6"]


def _smooth_rate_map_for_display(mat: np.ndarray, sigma_bins: float = _RATE_MAP_DISPLAY_SIGMA_BINS) -> np.ndarray:
    """NaN-aware Gaussian smoothing for Panel E visualization."""
    arr = np.asarray(mat, dtype=float)
    if sigma_bins <= 0:
        return arr
    valid = np.isfinite(arr)
    if not np.any(valid):
        return arr
    values = np.where(valid, arr, 0.0)
    weights = valid.astype(float)
    num = gaussian_filter(values, sigma=sigma_bins, mode="nearest", truncate=_RATE_MAP_DISPLAY_TRUNCATE)
    den = gaussian_filter(weights, sigma=sigma_bins, mode="nearest", truncate=_RATE_MAP_DISPLAY_TRUNCATE)
    smoothed = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 1e-8)
    smoothed[~valid] = np.nan
    return np.clip(smoothed, 0.0, 1.0)


def _spatial_autocorrelogram(mat: np.ndarray, min_n: int = SAC_MIN_OVERLAP) -> np.ndarray:
    """Full-lag Pearson SAC after Savelli et al. (2017) map smoothing.

    Clipped 5x5 Gaussian, variance 2 bins^2; at least five finite bins
    per smoothing window. Outside-map bins are unavailable, not wrapped.
    DOI: 10.7554/eLife.21354. No grid score is recomputed here.
    """
    m = np.asarray(mat, dtype=float)
    valid = np.isfinite(m)
    yy, xx = np.mgrid[-2:3, -2:3]
    kernel = np.exp(-(xx * xx + yy * yy) / 4.0)
    num = convolve(np.where(valid, m, 0.0), kernel, mode="constant", cval=0)
    den = convolve(valid.astype(float), kernel, mode="constant", cval=0)
    count = convolve(valid.astype(int), np.ones((5, 5), dtype=int), mode="constant", cval=0)
    m = np.divide(num, den, out=np.full_like(m, np.nan), where=(den > 0) & (count >= 5))
    ny, nx = m.shape
    r = np.full((2 * ny - 1, 2 * nx - 1), np.nan)
    for dy in range(-ny + 1, ny):
        for dx in range(-nx + 1, nx):
            a = m[max(0, dy):min(ny, ny + dy), max(0, dx):min(nx, nx + dx)]
            b = m[max(0, -dy):min(ny, ny - dy), max(0, -dx):min(nx, nx - dx)]
            overlap = np.isfinite(a) & np.isfinite(b)
            if int(overlap.sum()) < min_n:
                continue
            a, b = a[overlap], b[overlap]
            a, b = a - a.mean(), b - b.mean()
            denominator = np.linalg.norm(a) * np.linalg.norm(b)
            if denominator > 0:
                r[dy + ny - 1, dx + nx - 1] = np.clip(np.dot(a, b) / denominator, -1, 1)
    return r


def _bare_image_axis(ax) -> None:
    ax.set_xticks([]); ax.set_yticks([])
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _coordinate_glyph(ax, *, arm_frac: float, xlabel: str, ylabel: str, scale_text: str,
                      gap_in: float = 0.17, color: str | None = None) -> None:
    """L-shaped x/y arrow pair left of an image row: orientation and physical scale.

    Both arms are ``arm_frac`` of the (square) image width, i.e. they represent
    ``scale_text`` in that row.  Drawn as artists of ``ax`` (outside the image,
    unclipped) so that the glyph counts as panel ink for label placement.
    """
    c = PALETTE["black"] if color is None else color
    w_in = ax.get_position().width * ax.figure.get_figwidth()
    ox, oy = -gap_in / w_in - arm_frac, 0.0
    arrow = dict(arrowstyle="-|>,head_length=0.55,head_width=0.24", mutation_scale=6,
                 lw=0.7, color=c, shrinkA=0, shrinkB=0)
    kw = dict(xycoords="axes fraction", textcoords="axes fraction", annotation_clip=False)
    ax.annotate("", xy=(ox + arm_frac, oy), xytext=(ox, oy), arrowprops=arrow, **kw)
    ax.annotate("", xy=(ox, oy + arm_frac), xytext=(ox, oy), arrowprops=arrow, **kw)
    tkw = dict(transform=ax.transAxes, fontsize=FS_MIN, color=c, clip_on=False)
    ax.text(ox + arm_frac + 0.02, oy, xlabel, ha="left", va="center", **tkw)
    ax.text(ox, oy + arm_frac + 0.025, ylabel, ha="center", va="bottom", **tkw)
    ax.text(ox + arm_frac / 2, oy - 0.035, scale_text, ha="center", va="top", **tkw)


def _epoch_ticks(max_epoch: float) -> list[int]:
    """Powers of four from 1 to the first one >= the last epoch (150 -> ..., 64, 256)."""
    ticks = [1]
    while ticks[-1] < max_epoch:
        ticks.append(ticks[-1] * 4)
    return ticks


def _history_panel(ax, hist: pd.DataFrame, value: str, ylabel: str, *, scale: float = 1.0,
                   logy: bool = False):
    if hist.empty or value not in hist:
        empty_panel(ax)
        return
    piv = hist.pivot_table(index="epoch", columns="source_seed", values=value).sort_index()
    for col in piv:
        ax.plot(piv.index, piv[col] * scale, color=PALETTE["neutral_mid"], alpha=0.45, lw=0.55, zorder=1)
    m = piv.mean(axis=1).to_numpy(float) * scale
    se = piv.std(axis=1, ddof=1).to_numpy(float) / np.sqrt(max(1, piv.shape[1])) * scale
    x = piv.index.to_numpy(float)
    ax.fill_between(x, m - se, m + se, color=PALETTE["navigation"], alpha=BAND_ALPHA + 0.06, lw=0, zorder=2)
    ax.plot(x, m, color=PALETTE["navigation"], lw=LINEWIDTH + 0.2, zorder=3)
    sel = hist[hist.selected_best_validation.astype(bool)].copy()
    if not sel.empty:
        ax.scatter(sel.epoch, sel[value] * scale, s=24, marker="o", facecolor="white",
                   edgecolor=PALETTE["navigation_dark"], linewidth=0.95, zorder=5)
    ax.set_xscale("log", base=2)
    max_epoch = float(np.nanmax(np.r_[x, hist.epoch.to_numpy(float)]))
    ticks = _epoch_ticks(max_epoch)
    ax.set_xticks(ticks, [str(v) for v in ticks])
    ax.set_xticks([2 * t for t in ticks[:-1]], minor=True)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(_EPOCH_AXIS_START, ticks[-1])
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Navigation epoch")
    ax.set_ylabel(ylabel)
    closed_frame(ax)


#: Broken y axis of D: used when the tallest bin is more than this many times
#: the second tallest (the near-zero spike would otherwise flatten the tail).
_BREAK_RATIO = 2.0
#: Count range shown above the break and the white gap between the segments.
_BREAK_TOP_SPAN = 80
_BREAK_GAP_IN = 0.07


def _ceil_to(value: float, step: float) -> float:
    return float(np.ceil(value / step) * step)


def _grid_score_panel(ax, ax_top, scores: pd.DataFrame, hist: pd.DataFrame, rep_seed: int, top_units):
    """Histogram of every unit's grid score.

    ``ax`` is the full panel rectangle on entry.  If the near-zero bin is a
    spike, the rectangle is split into a lower segment (0 to just above the
    second-tallest bin) and a short upper segment that holds the top of the
    spike.  Both segments use the same counts-per-inch, so bar heights stay
    comparable across the break.
    """
    if scores.empty or "grid_score" not in scores:
        empty_panel(ax, "Grid scores unavailable")
        ax_top.set_visible(False)
        return None
    g = scores.grid_score.to_numpy(float)
    g = g[np.isfinite(g)]
    bins = np.arange(-0.45, 1.46, 0.05)
    counts, edges = np.histogram(g, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    frac_above = float(np.mean(g > GRID_THRESHOLD))
    sel = hist[(hist.source_seed == rep_seed) & hist.selected_best_validation.astype(bool)]
    f_g = float(sel.grid_fraction.iloc[0]) if len(sel) else np.nan
    criterion_ok = np.isfinite(f_g) and abs(frac_above - f_g) < 0.002
    colors = [PALETTE["navigation"] if (criterion_ok and c > GRID_THRESHOLD) else PALETTE["neutral_light"]
              for c in centers]

    # x axis: labelled ends at multiples of 0.5 that enclose every unit's score.
    x_lo = float(np.floor(g.min() / 0.5) * 0.5)                  # -0.40 -> -0.5
    x_hi = _ceil_to(g.max(), 0.5)                                 #  1.35 ->  1.5
    x_ticks = np.arange(x_lo, x_hi + 1e-9, 0.5)

    peak = float(counts.max())
    second = float(np.sort(counts)[-2]) if counts.size > 1 else peak
    broken = second > 0 and peak > _BREAK_RATIO * second
    if broken:
        low_top = _ceil_to(second * 1.06, 10)                     # 306 units -> 330
        high_top = _ceil_to(peak, 20) + 20                        # 916 units -> 940
        high_bottom = high_top - _BREAK_TOP_SPAN                  # 860
    else:
        low_top = _ceil_to(peak * 1.08, 10)

    full = ax.get_position()
    fig = ax.figure
    if broken:
        gap = _BREAK_GAP_IN / fig.get_figheight()
        h_low = (full.height - gap) * low_top / (low_top + _BREAK_TOP_SPAN)
        ax.set_position([full.x0, full.y0, full.width, h_low])
        ax_top.set_position([full.x0, full.y0 + h_low + gap, full.width, full.height - h_low - gap])
    segments = [ax, ax_top] if broken else [ax]
    if not broken:
        ax_top.set_visible(False)

    for a in segments:
        a.bar(centers, counts, width=0.05, color=colors, edgecolor="white", linewidth=0.3, zorder=2)
        if criterion_ok:
            a.axvline(GRID_THRESHOLD, color=PALETTE["neutral_dark"], lw=0.8, ls=(0, (3, 2)), zorder=3)
        a.set_xlim(x_lo, x_hi)
        a.set_xticks(x_ticks)
        a.set_xticks(x_ticks[:-1] + 0.25, minor=True)
        closed_frame(a)

    step = 100 if low_top > 200 else 50
    ax.set_ylim(0, low_top)
    ax.set_yticks(np.arange(0, low_top + 1e-9, step))
    # The histogram remains a grid-score distribution, but the advisor asked
    # to remove the redundant x-axis title and use the compact count label.
    ax.set_xlabel("Grid score")
    ax.set_ylabel("Counts")
    if broken:
        ax_top.set_ylim(high_bottom, high_top)
        ax_top.set_yticks([t for t in np.arange(0, high_top + 1e-9, step) if t >= high_bottom])
        ax_top.spines["bottom"].set_visible(False)
        ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        axis_break_marks(ax, ax_top)
        # One y label for both segments, centred on the whole panel.
        ax.yaxis.label.set_y(0.5 * full.height / ax.get_position().height)
    return criterion_ok


def _trajectory_panel(ax, traj: pd.DataFrame):
    if traj.empty:
        empty_panel(ax, "Representative rollout unavailable")
        return
    if "time" in traj:
        traj = traj.sort_values("time", kind="stable")
    tx, ty = traj.true_x.to_numpy(float), traj.true_y.to_numpy(float)
    dx, dy = traj.decoded_x.to_numpy(float), traj.decoded_y.to_numpy(float)
    n_steps = len(tx)
    for a, b, c, d in zip(tx, ty, dx, dy):          # per-step decoding error
        ax.plot([a, c], [b, d], color=PALETTE["neutral_mid"], lw=0.55, zorder=1)
    # Every time step is a marker on both paths (20 steps in the frozen rollout).
    # The decoder output is held for several consecutive steps, so some decoded
    # markers coincide exactly; the grey error segments show which true step
    # each decoded marker belongs to.
    true_kw = dict(color=PALETTE["neutral_dark"], lw=1.3, marker="o", ms=2.9,
                   mfc=PALETTE["neutral_light"], mec=PALETTE["neutral_dark"], mew=0.6)
    dec_kw = dict(color=PALETTE["navigation"], lw=1.1, marker="o", ms=3.4,
                  mfc=PALETTE["navigation"], mec="white", mew=0.4)
    ax.plot(tx, ty, "-", zorder=2, solid_capstyle="round", **true_kw)
    ax.plot(dx, dy, "-", zorder=3, **dec_kw)
    ax.scatter(tx[0], ty[0], s=30, color=PALETTE["black"], zorder=5)
    ax.scatter(tx[-1], ty[-1], s=30, facecolor="white", edgecolor=PALETTE["black"], linewidth=0.9, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("S")   # final scale-bar alignment is applied after column spacing
    pad = 0.035
    ax.set_xlim(min(tx.min(), dx.min()) - pad, max(tx.max(), dx.max()) + pad)
    # Extra room below the path for the scale bar and the key.
    ax.set_ylim(min(ty.min(), dy.min()) - 0.085, max(ty.max(), dy.max()) + 0.03)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    # 10-cm scale bar, bottom left.
    x0 = min(tx.min(), dx.min()) - 0.01
    y0 = min(ty.min(), dy.min()) - 0.045
    scale_bar, = ax.plot([x0, x0 + 0.10], [y0, y0], color=PALETTE["black"], lw=1.6, solid_capstyle="butt")
    ax.text(x0 + 0.05, y0 - 0.008, "10 cm", ha="center", va="top", fontsize=FS_MIN)
    # Step numbers at the two ends make the rollout length explicit.
    step_kw = dict(textcoords="offset points", va="center", fontsize=FS_MIN,
                   color=PALETTE["neutral_dark"], zorder=6)
    ax.annotate("step 1", xy=(tx[0], ty[0]), xytext=(-25, -1.0), ha="left", **step_kw)
    ax.annotate(f"step {n_steps}", xy=(tx[-1], ty[-1]), xytext=(28, 0.0), ha="right", **step_kw)
    handles = [Line2D([0], [0], label="True", **true_kw),
               Line2D([0], [0], label="Decoded", **dec_kw)]
    legend(ax, loc="lower right", handles=handles, bbox_to_anchor=(-0.05, 0.1), handlelength=1.3)
    return scale_bar


def _select_representative_trajectory(data: Path, expected_source: int):
    """Load the lowest-error frozen validation trajectory without hard-coding its id.

    The summary table is part of the active Figure 2 source-data contract.  The
    selected file is searched first in the 100-candidate archive and then in
    the convenience copy at the Figure2 root.  Any missing or inconsistent
    summary/file pair is a hard error: silently falling back to an arbitrary
    rollout would make the representative panel non-reproducible.
    """
    summary_path = data / "trajectory_candidates_20step" / "trajectory_candidates_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing representative-trajectory summary: {summary_path}")
    summary = pd.read_csv(summary_path)
    required = {"trajectory_id", "n_steps", "mean_error_m", "final_error_m", "file"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(
            "Representative-trajectory summary is missing columns: "
            + ", ".join(sorted(missing))
        )
    if summary.empty:
        raise ValueError("Representative-trajectory summary is empty")
    for col in ("trajectory_id", "n_steps", "mean_error_m", "final_error_m"):
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    summary = summary.dropna(subset=["trajectory_id", "n_steps", "mean_error_m", "final_error_m"])
    if summary.empty:
        raise ValueError("Representative-trajectory summary has no numeric candidate rows")
    # Deterministic ordering makes the selected candidate reproducible even if
    # two trajectories have the same rounded summary error.
    row = summary.sort_values(
        ["mean_error_m", "final_error_m", "trajectory_id"], kind="stable"
    ).iloc[0]
    filename = str(row["file"])
    candidates = [
        data / "trajectory_candidates_20step" / filename,
        data / filename,
    ]
    trajectory_path = next((p for p in candidates if p.exists()), None)
    if trajectory_path is None:
        raise FileNotFoundError(
            f"Selected representative trajectory {filename!r} is absent from both "
            f"{candidates[0].parent} and {data}"
        )
    traj = load_csv(trajectory_path)
    if traj.empty:
        raise ValueError(f"Selected representative trajectory is empty: {trajectory_path}")
    if len(traj) != int(row["n_steps"]):
        raise ValueError(
            f"Trajectory length mismatch for {trajectory_path.name}: "
            f"summary={int(row['n_steps'])}, file={len(traj)}"
        )
    metadata_path = data / "representative_metadata.json"
    if metadata_path.exists():
        metadata = pd.read_json(metadata_path, typ="series")
        if "source_seed" in metadata and int(metadata["source_seed"]) != int(expected_source):
            raise ValueError(
                f"Representative metadata source {int(metadata['source_seed'])} "
                f"does not match FigureContext source {int(expected_source)}"
            )
    selection = {
        "summary_path": summary_path,
        "trajectory_path": trajectory_path,
        "trajectory_id": int(row["trajectory_id"]),
        "n_candidates": int(len(summary)),
        "n_steps": int(row["n_steps"]),
        "mean_error_m": float(row["mean_error_m"]),
        "final_error_m": float(row["final_error_m"]),
    }
    return traj, selection


def build(ctx: FigureContext):
    apply_publication_style()
    data = ctx.source_data_dir / "Figure2"
    out = ctx.figures_dir
    hist = load_csv(data / "navigation_history.csv")
    traj, trajectory_selection = _select_representative_trajectory(data, ctx.representative_source)
    rates = [load_csv(data / f"rate_map_rank_{i}_2500.csv") for i in RATE_MAP_RANKS]
    scores = load_csv(data / "representative_grid_scores.csv")
    rep_seed = ctx.representative_source

    W, H = FIG_WIDTH, 3.95
    fig = plt.figure(figsize=(W, H))
    add = lambda x, y, w, h: fig.add_axes([x / W, y / H, w / W, h / H])   # inches -> fraction

    panel_w, panel_h = 1.70, 1.28
    top_y, bottom_y = 2.42, 0.42
    ax_a = add(0.50, top_y, panel_w, panel_h)
    ax_b = add(2.78, top_y, panel_w, panel_h)
    ax_c = add(5.05, top_y, 1.72, 1.44)
    # E's two square rows have exactly the same total height as A and D.
    row_gap = 0.12
    s, gap = (panel_h - row_gap) / 2, 0.16
    cbar_x = 5.91   # shift all of E left by 0.40 in; its label stays aligned with B
    x0 = cbar_x - 0.09 - (4 * s + 3 * gap)
    e_bottom_y = bottom_y
    e_top_y = e_bottom_y + s + row_gap
    # D matches A's full rectangle, including both segments and the axis break.
    ax_d = add(0.50, bottom_y, panel_w, panel_h)
    ax_d_top = add(0.50, bottom_y, panel_w, panel_h)   # upper segment of D's broken y axis (split below)
    # E: autocorrelograms (top row; G is computed from them) above rate maps.
    sac_axes = [add(x0 + i * (s + gap), e_top_y, s, s) for i in range(4)] if SHOW_AUTOCORRELOGRAMS else []
    map_y = e_bottom_y if sac_axes else e_top_y
    map_axes = [add(x0 + i * (s + gap), map_y, s, s) for i in range(4)]

    # A, B: learning over navigation training.
    # Use the mathematical quantity on the y-axis; the caption defines it as
    # position-decoding error.
    _history_panel(ax_a, hist, "pi_error_m", "$\\varepsilon_{\\mathrm{pos}}$ (cm)",
                   scale=100, logy=True)
    ax_a.set_yticks([3, 10, 30, 100], ["3", "10", "30", "100"])
    ax_a.yaxis.set_minor_locator(plt.NullLocator())
    ax_a.set_ylim(2.6, 140)
    _history_panel(ax_b, hist, "grid_fraction", "$f_{\\mathrm{G}}$", scale=1.0)
    # A fraction on the shared 0-1 scale (starts just below 0 like the accuracy axes).
    ax_b.set_ylim(-0.06, 1.04)
    ax_b.set_yticks([0, 0.5, 1.0])
    ax_b.set_yticks([0.25, 0.75], minor=True)

    # D: population of grid scores behind the example maps.
    top_units = []
    for frame in rates:
        if not frame.empty:
            top_units.append((int(frame.unit.iloc[0]), float(frame.grid_score.iloc[0])))
    _grid_score_panel(ax_d, ax_d_top, scores, hist, rep_seed, top_units)

    # C: slightly enlarged rollout; align its visible scale bar after column spacing.
    trajectory_scale_bar = _trajectory_panel(ax_c, traj)

    # E: units ranked 2 through 5 -- spatial autocorrelogram (top; G above each
    # column is computed from it) and rate map (bottom).  Row identities are in
    # the caption; the glyph at the left of each row gives orientation + scale.
    mappable = sac_mappable = None
    n_map_bins = None
    for i, (ax, frame) in enumerate(zip(map_axes, rates)):
        sac_ax = sac_axes[i] if sac_axes else None
        if frame.empty:
            ax.axis("off")
            if sac_ax is not None:
                sac_ax.axis("off")
            continue
        mat = frame.pivot(index="y_bin", columns="x_bin", values="normalized_activation").to_numpy()
        n_map_bins = int(mat.shape[1])
        mappable = ax.imshow(_smooth_rate_map_for_display(mat), origin="lower", cmap="viridis",
                             vmin=0, vmax=1, interpolation="none", rasterized=True)
        _bare_image_axis(ax)
        top_ax = sac_ax if sac_ax is not None else ax
        g = float(frame.grid_score.iloc[0])
        top_ax.text(0.5, 1.03, f"$G$ = {g:.2f}", transform=top_ax.transAxes, ha="center",
                    va="bottom", fontsize=FS_MIN)
        if sac_ax is not None:
            sac = _spatial_autocorrelogram(mat)
            sac_mappable = sac_ax.imshow(sac, origin="lower", cmap="RdBu_r", vmin=-1.0, vmax=1.0,
                                         interpolation="none", rasterized=True)
            _bare_image_axis(sac_ax)
    if n_map_bins:
        bin_m = ARENA_SIDE_M / n_map_bins
        _coordinate_glyph(map_axes[0], arm_frac=GLYPH_ARM_M["map"] / (n_map_bins * bin_m),
                          xlabel="$x$", ylabel="$y$", scale_text=f"{GLYPH_ARM_M['map']:g} m")
        if sac_axes:
            n_lag = 2 * n_map_bins - 1                     # full-lag autocorrelogram
            _coordinate_glyph(sac_axes[0], arm_frac=GLYPH_ARM_M["sac"] / (n_lag * bin_m),
                              xlabel=r"$\Delta x$", ylabel=r"$\Delta y$",
                              scale_text=f"{GLYPH_ARM_M['sac']:g} m")

    # Colour bars at the right end of each row, without tick marks (a tick
    # next to "1" read as a minus sign).
    def _colorbar(source, row_y, ticks, labels, text):
        cax = add(cbar_x, row_y + 0.06, 0.06, s - 0.12)
        cb = fig.colorbar(source, cax=cax)
        cb.set_ticks(ticks, labels=labels)
        cb.ax.minorticks_off()
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=FS_MIN, length=0, width=0, pad=2.0)
        cb.set_label(text, fontsize=FS_MIN, labelpad=1.5)
    if sac_mappable is not None:
        _colorbar(sac_mappable, e_top_y, [-1, 0, 1], ["−1", "0", "1"], "Correlation")
    if mappable is not None:
        _colorbar(mappable, map_y, [0, 1], ["0", "1"], "Norm. activity")

    # Identical white space between A, B and C (ink to ink).
    # left_in pins the leftmost ink (A's two-line y label) inside the page so
    # the figure stays at FIG_WIDTH.
    equalize_column_gaps(fig, [ax_a, ax_b, ax_c], partners={}, left_in=0.07)

    # Column spacing can move A; keep both segments of D directly beneath it.
    a_box = ax_a.get_position()
    for ax in (ax_d, ax_d_top):
        box = ax.get_position()
        ax.set_position([a_box.x0, box.y0, a_box.width, box.height])

    # Align the visible scale bar, not the invisible axes box, with A/B's x-axes.
    # Measure after aspect/spacing changes; translating the original box keeps C's scale.
    if trajectory_scale_bar is not None:
        fig.canvas.draw()
        bar_xy = trajectory_scale_bar.get_xydata()[0]
        bar_y = fig.transFigure.inverted().transform(ax_c.transData.transform(bar_xy))[1]
        dy = ax_a.get_position().y0 - bar_y
        ax_c.set_position(ax_c.get_position(original=True).translated(0.0, dy))

    # C: letter level with A/B. D/E: shared label height above E's top row.
    e_label_ax = (sac_axes or map_axes)[0]
    de_label_y = (bottom_y + panel_h + 0.08) / H
    place_panel_labels(fig, [
        (ax_a, "A"), (ax_b, "B"), (ax_c, "C", {"y_fig": ax_a.get_position().y1 + 2.0 / 72 / H}),
        (ax_d, "D", {"y_fig": de_label_y}),
        (e_label_ax, "E", {"y_fig": de_label_y}),
    ], align=[("A", "D"), ("B", "E")])

    paths = save_publication_figure(fig, out / "Figure2")
    inputs = [
        data / "navigation_history.csv", trajectory_selection["summary_path"],
        trajectory_selection["trajectory_path"],
        *[data / f"rate_map_rank_{i}_2500.csv" for i in RATE_MAP_RANKS],
        data / "representative_grid_scores.csv",
    ]
    write_json(
        out / "manifests" / "Figure2.json",
        manifest_for_figure(
            ctx, "Figure2", Path(__file__), inputs, paths,
            notes=[
                "Figure1 is an external author-drawn model schematic.",
                "A, B: grey = individual navigation sources, blue = mean +/- SEM; hollow points mark minimum-validation-loss checkpoints used by all transfer analyses. A: position-decoding error eps_pos on a logarithmic axis. B: grid-like unit fraction f_G as a fraction (0-1). The log2 epoch axis runs from just below epoch 1 to 256 so that all 150 epochs lie inside the labelled range.",
                f"D: grid scores of all units of source {rep_seed} at its validation-best checkpoint (representative_grid_scores.csv). The dashed G = {GRID_THRESHOLD} criterion (blue bars = grid-like units) is drawn only when the fraction of units above it reproduces f_G of navigation_history.csv (|diff| < 0.002). The y axis is broken when the near-zero bin exceeds twice the second-tallest bin; both segments use the same counts per inch.",
                f"C: true (grey) and decoded (blue) positions of the lowest-mean-error member of the frozen validation-trajectory candidate table (trajectory id {trajectory_selection['trajectory_id']}, {trajectory_selection['n_candidates']} candidates, {trajectory_selection['n_steps']} steps); the decoder output is held over consecutive steps, so coincident decoded markers overlap exactly; filled circle = start (step 1), open circle = end; thin grey segments join simultaneous true/decoded positions; scale bar 10 cm.",
                "E: units ranked 2 through 5; top row = spatial autocorrelograms (G above each column), bottom row = rate maps; the coordinate glyph at the left of each row shows x/y (rate map, 0.5 m arms) and lag dx/dy (autocorrelogram, 1 m arms), assuming the 50 x 50 maps tile the 2.2 m arena. Rate maps use NaN-aware Gaussian smoothing (sigma = 0.75 bin) for display only. Autocorrelograms follow the map-level procedure of Savelli et al. (2017, doi:10.7554/eLife.21354): 5x5 Gaussian smoothing with variance 2 bins^2, at least 5 finite bins per smoothing window, Pearson correlation with overlap >= 100 bins, full lag range without circular cropping, colour scale [-1, 1]. Input zeros are treated as observed values; occupancy masks are unavailable. Displayed grid scores are the frozen values and are not recomputed.",
            ],
        ),
    )
    return paths
