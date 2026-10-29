"""Shared presentation layer for Figure 7 and Supplementary Figures S7, S8, S9, S11 and S12.

These builders are presentation-only.  They read exclusively from the formally
frozen tables under ``aggregate_full/figure_source_data`` and never touch the
experiment output tree or recompute inferential tests.

This module holds the *single* typographic and geometric contract used by the
five mechanism figures, so that panel letters, legends, in-panel statistics,
source clouds and mean curves are pixel-consistent with each other and with the
Figure 2-6 style defined in ``nature_style.py``.  Nothing here changes any data
interface: every function takes the same frozen columns the panels already use.
"""
from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

try:
    from .nature_style import (
        PALETTE, BAND_ALPHA, CAPSIZE, ERROR_LINEWIDTH, LINEWIDTH,
        deterministic_jitter, mean_ci95,
        # Shared presentation contract (single source of truth for every figure).
        FS_ANNOT, FS_LEGEND, FS_INPANEL, FS_TICK_DENSE,
        MS_MEAN, MS_SUMMARY, MS_SMALL,
        SOURCE_LW, SOURCE_ALPHA, PAIR_LW, PAIR_ALPHA, SWARM_SIZE, SWARM_ALPHA,
        LEGEND_KW, LEGEND_KW_POINTS, MINUS, num, sci, annotate, legend,
        categorical_xlim, categorical_ylim, group_brackets, place_panel_labels,
        line_kw, marker_kw, summary_errorbar, light_line_kw,
    )
except ImportError:
    from nature_style import (
        PALETTE, BAND_ALPHA, CAPSIZE, ERROR_LINEWIDTH, LINEWIDTH,
        deterministic_jitter, mean_ci95,
        FS_ANNOT, FS_LEGEND, FS_INPANEL, FS_TICK_DENSE,
        MS_MEAN, MS_SUMMARY, MS_SMALL,
        SOURCE_LW, SOURCE_ALPHA, PAIR_LW, PAIR_ALPHA, SWARM_SIZE, SWARM_ALPHA,
        LEGEND_KW, LEGEND_KW_POINTS, MINUS, num, sci, annotate, legend,
        categorical_xlim, categorical_ylim, group_brackets, place_panel_labels,
        line_kw, marker_kw, summary_errorbar, light_line_kw,
    )


try:
    from . import figure_paths
except ImportError:  # direct execution from hard8_phaseF/
    import figure_paths
AGGREGATE = figure_paths.AGGREGATE
SOURCE_ROOT = figure_paths.SOURCE_ROOT
OUT = figure_paths.OUT

NAV = PALETTE["navigation"]
MATCHED = PALETTE["matched"]   # L* 55, distinguishable from NAV in greyscale
REL = PALETTE["relation"]
REC = PALETTE["recurrent"]
NEG = PALETTE["negative"]
POS = PALETTE["positive"]
MID = PALETTE["neutral_mid"]
LIGHT = PALETTE["neutral_light"]
# Short vs long path splits and the three rho doses are separated in
# lightness (~20 L* apart) as well as in hue, marker and dash pattern.
SHORT = "#6C98D4"   # L* 62, light blue
LONG = "#A8403A"    # L* 42, brick
RHO_COLORS = {0.0: PALETTE["navigation_dark"], 0.05: "#8466B3", 0.10: "#E3927F"}   # L* 30 / 49 / 68

# ---------------------------------------------------------------------------
# Frozen source-data access (unchanged interface).
# ---------------------------------------------------------------------------
def figure_data(name: str) -> Path:
    path = SOURCE_ROOT / name
    if not path.is_dir():
        raise FileNotFoundError(f"Missing frozen figure source-data directory: {path}")
    complete = path / "PREPARE_COMPLETE.json"
    if not complete.exists():
        raise FileNotFoundError(f"Missing PREPARE_COMPLETE.json: {complete}")
    payload = read_json(complete)
    if payload.get("status") != "complete":
        raise RuntimeError(f"Frozen source-data is not complete: {complete}")
    return path


def read_csv(path: Path, required: tuple[str, ...] = ()) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty source table: {path}")
    df = pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path.name} missing columns: {missing}")
    return df


def read_json(path: Path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_sem_by(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    work = df[[x, y]].copy()
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work[np.isfinite(work[y])]
    rows = []
    for xv, q in work.groupby(x, sort=True):
        vals = q[y].to_numpy(float)
        mean = float(vals.mean())
        sem = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
        rows.append((xv, mean, sem, len(vals)))
    return pd.DataFrame(rows, columns=[x, "mean", "sem", "n"])


# ---------------------------------------------------------------------------
# Drawing primitives.
# ---------------------------------------------------------------------------
def source_trajectory_lines(ax, df: pd.DataFrame, *, x: str, y: str, group: str,
                            color: str, alpha: float = SOURCE_ALPHA,
                            lw: float = SOURCE_LW) -> None:
    """Per-source trajectories: visible as an ensemble, never as chart furniture."""
    for _, q in df.sort_values([group, x]).groupby(group):
        ax.plot(q[x].to_numpy(float), q[y].to_numpy(float), color=color,
                alpha=alpha, lw=lw, zorder=1, solid_capstyle="round")


def mean_curve(ax, df: pd.DataFrame, x: str, y: str, *, color: str | None = None, label: str | None = None,
               marker: str = "o", ls: str = "-", source_lines: bool = True,
               band: bool = True, markevery=None, ms: float = MS_MEAN,
               group: str = "source_seed", zorder: float = 3, spec: dict | None = None,
               light: bool = False):
    """Source ensemble + mean curve + SEM band, drawn identically in every panel.

    ``markevery`` thins markers on densely sampled curves so that a 160-point
    learning curve reads as a line rather than as a chain of markers.
    """
    if spec is None:
        spec = dict(color=color, marker=marker, ls=ls, filled=True, lw=LINEWIDTH, ms_scale=1.0)
    color = spec["color"]
    if source_lines and group in df.columns:
        source_trajectory_lines(ax, df, x=x, y=y, group=group, color=color)
    q = mean_sem_by(df, x, y)
    xv = q[x].to_numpy(float)
    mean = q["mean"].to_numpy(float)
    sem = q["sem"].to_numpy(float)
    if band and len(xv) > 1:
        ax.fill_between(xv, mean - sem, mean + sem, color=color,
                        alpha=BAND_ALPHA, lw=0, zorder=zorder - 1)
    # light=True: comparison training curve drawn as a light solid line.
    kw_fn = light_line_kw if light else line_kw
    ax.plot(xv, mean, **kw_fn(spec, ms=ms, markevery=markevery, label=label, zorder=zorder))
    return q


def source_swarm_with_ci(ax, x: float, values, *, color: str, marker: str = "o",
                         jitter_width: float = .055, face: str | None = None,
                         point_alpha: float = SWARM_ALPHA, zorder: int = 4,
                         label: str | None = None):
    """Every source as a point, plus the mean and its 95% CI."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    jitter = deterministic_jitter(len(vals), width=jitter_width)
    facecolor = color if face is None else face
    open_face = facecolor == "white"
    ax.scatter(
        np.full(len(vals), x) + jitter, vals, s=SWARM_SIZE, marker=marker,
        facecolor=facecolor, edgecolor=color if open_face else "white",
        linewidth=.8 if open_face else .45, alpha=point_alpha, zorder=zorder - 1,
    )
    spec = dict(color=color, marker=marker, filled=not open_face, ms_scale=1.0)
    return summary_errorbar(ax, x, vals, spec=spec, zorder=zorder + 1, label=label)


def deterministic_jitter_safe(n: int, *, width: float = .05) -> np.ndarray:
    """Deterministic jitter that also handles the n <= 1 edge case."""
    return deterministic_jitter(n, width=width) if n > 1 else np.zeros(max(n, 0))


def paired_source_lines(ax, xs, series, *, color: str = MID,
                        lw: float = PAIR_LW, alpha: float = PAIR_ALPHA) -> None:
    """Faint within-source connectors, matching the Figure 3/5/6 pairing idiom."""
    for values in series:
        values = np.asarray(values, dtype=float)
        if np.isfinite(values).sum() >= 2:
            ax.plot(xs, values, color=color, lw=lw, alpha=alpha, zorder=1)
