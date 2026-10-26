"""Common data-loading and plotting helpers for manuscript figures."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .nature_style import (
    BAND_ALPHA, CAPSIZE, ERROR_LINEWIDTH, LINEWIDTH, LINEWIDTH_SECONDARY,
    MARKERSIZE, PALETTE, POINT_ALPHA, deterministic_jitter, mean_ci95, mean_sem,
    reference_line, style_axis, SWARM_SIZE, MS_SUMMARY,
)

LABELS = {
    "transfer_freeze": "Navigation frozen",
    "ctl01_frozen": "Delayed reconstruction",
    "random_frozen": "Random recurrent",
    "transfer_finetune": "Navigation fine-tuned",
    "random_init": "Random trainable",
    "transfer_input_only": r"Navigation $\mathbf{W}_{\mathrm{in}}$ only",
    "transfer_recurrent_only": r"Navigation $\mathbf{W}_{\mathrm{rec}}$ only",
    "nav_frozen": "Navigation",
    "baseline_action_count_linear": "Action-count linear",
    "baseline_linear_accumulator": "2D accumulator",
    "nav_adapter": "Navigation + adapter",
    "nav_noadapter": "Navigation, readout only",
    "random_adapter": "Random + adapter",
    "navigation_frozen": "Navigation frozen",
    "matched_random_frozen": "Matched random frozen",
    "navigation_trainable": "Navigation trainable",
    "matched_random_trainable": "Matched random trainable",
    "navigation_hh_only_frozen": r"Navigation $\mathbf{W}_{\mathrm{rec}}$ only",
    "navigation_ih_only_frozen": r"Navigation $\mathbf{W}_{\mathrm{in}}$ only",
    "loop_holonomy": "Loop holonomy",
    "zero_padding_phase_difference": "Zero-padding phase",
    "path_identity_decoder_accuracy": "Path identity decoding",
    "residual_time_decoder_r2": "Residual time decoding",
    "exact_2d_gate": "Exact 2-D",
    "toroidal_base_plus_fibre_gate": "Base + fibre",
}

COLORS = {
    "transfer_freeze": PALETTE["navigation"],
    "ctl01_frozen": PALETTE["control"],
    "random_frozen": PALETTE["random"],
    "transfer_finetune": PALETTE["navigation_dark"],
    "random_init": PALETTE["random"],
    "transfer_input_only": PALETTE["input"],
    "transfer_recurrent_only": PALETTE["recurrent"],
    "nav_frozen": PALETTE["navigation"],
    "baseline_action_count_linear": PALETTE["input"],
    "baseline_linear_accumulator": PALETTE["accumulator"],
    "nav_adapter": PALETTE["navigation"],
    # Same navigation core without the adapter: preserve origin with an open
    # marker in Figure4 rather than changing its semantic color.
    "nav_noadapter": PALETTE["navigation"],
    "random_adapter": PALETTE["random"],
    "navigation_frozen": PALETTE["navigation"],
    "matched_random_frozen": PALETTE["random"],
    "navigation_trainable": PALETTE["relation"],
    "matched_random_trainable": PALETTE["neutral_mid"],
    "navigation_hh_only_frozen": PALETTE["recurrent"],
    "navigation_ih_only_frozen": PALETTE["input"],
    "aligned": PALETTE["navigation"],
    "shuffled": PALETTE["relation"],
    "mismatched_target": PALETTE["random"],
}

MARKERS = {
    "transfer_freeze": "o", "ctl01_frozen": "D", "random_frozen": "s",
    "transfer_finetune": "o", "random_init": "s", "transfer_input_only": "^",
    "transfer_recurrent_only": "D", "nav_frozen": "o",
    "baseline_action_count_linear": "v", "baseline_linear_accumulator": "P",
    "navigation_frozen": "o", "matched_random_frozen": "s",
    "navigation_trainable": "o", "matched_random_trainable": "s",
    "navigation_hh_only_frozen": "D", "navigation_ih_only_frozen": "^",
    "aligned": "o", "shuffled": "D", "mismatched_target": "s",
}


def label(value: object) -> str:
    text = str(value)
    if text in LABELS:
        return LABELS[text]
    if text.startswith("Ohh-"):
        try:
            parts = dict(p.split("-", 1) for p in text.split("__"))
            origin = "Navigation" if parts["Ohh"] == parts["Oih"] == "navigation" else (
                "Random" if parts["Ohh"] == parts["Oih"] == "matched_random" else "Mixed"
            )
            return f'{origin}: hh {parts["Thh"]}, in {parts["Tih"]}'
        except Exception:
            pass
    return text.replace("_", " ")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


@dataclass(frozen=True)
class FigureContext:
    project_root: Path
    plan_path: Path
    aggregate_dir: Path
    run_dir: Path
    source_data_dir: Path
    figures_dir: Path
    plan: dict

    @property
    def source_pairs(self):
        return self.plan["source_pairs"]

    @property
    def n_sources(self) -> int:
        return len(self.source_pairs)

    @property
    def representative_source(self) -> int:
        return int(self.source_pairs[0]["source_seed"])

    def navigation_source_dir(self, seed: int) -> Path:
        """Locate the source directory that owns the canonical navigation run.

        Final from-scratch runs keep navigation and downstream artifacts in one
        tree. Historical best-validation reruns keep navigation under the
        original ``sources/`` tree while downstream/topology artifacts live
        under ``rerun_best_validation/sources/``. Search explicitly for the
        navigation history instead of assuming --project is the repository root.
        """
        name = f"nav_seed_{int(seed)}"
        candidates = [
            self.run_dir / "sources" / name,
            self.plan_path.parent / "sources" / name,
            self.project_root / "sources" / name,
        ]
        for parent in (self.aggregate_dir, *self.aggregate_dir.parents):
            candidates.append(parent / "sources" / name)

        seen: set[Path] = set()
        for source in candidates:
            source = source.resolve()
            if source in seen:
                continue
            seen.add(source)
            if (source / "artifacts" / "navigation" / "history.json").exists():
                return source

        return (self.plan_path.parent / "sources" / name).resolve()

    def analysis_source_dir(self, seed: int) -> Path:
        """Return the source directory that owns the current analysis artifacts.

        This transparently supports both layouts:

        1. final from-scratch run: ``<run>/sources/nav_seed_*``;
        2. historical hybrid best-validation rerun:
           ``<run>/rerun_best_validation/sources/nav_seed_*``.

        We search from the aggregate directory upward for the nearest ``sources``
        tree containing the Schøyen topology artifact.  This makes the explicit
        plotting package useful for the user's already-completed hybrid cohort
        without weakening the canonical one-tree final reproduction.
        """
        name = f"nav_seed_{int(seed)}"
        seen: set[Path] = set()
        for parent in (self.aggregate_dir, *self.aggregate_dir.parents):
            source = parent / "sources" / name
            if source in seen:
                continue
            seen.add(source)
            if (source / "artifacts" / "torus_topology").exists():
                return source
        return self.run_dir / "sources" / name

    @property
    def final_checkpoint_policy(self) -> str | None:
        """Checkpoint label to write into figure provenance.

        Old frozen ``plan.json`` files can still record the superseded balanced
        policy even when the aggregate directory is the later
        ``rerun_best_validation`` cohort.  In that recognized historical layout
        the figure provenance must reflect the actual data contract.
        """
        configured = self.plan.get("config", {}).get("checkpoint_policy")
        if configured == "best_validation":
            return configured
        if "rerun_best_validation" in self.aggregate_dir.parts:
            return "best_validation"
        return configured


def make_context(project_root: Path, plan_path: Path, aggregate_dir: Path) -> FigureContext:
    project_root = Path(project_root).resolve()
    plan_path = Path(plan_path).resolve()
    aggregate_dir = Path(aggregate_dir).resolve()
    plan = read_json(plan_path)

    # The frozen plan itself lives in the run directory. Historical plotting
    # wrappers sometimes pass that same run directory as --project, so
    # appending plan["run_root"] again creates a duplicated path.
    plan_parent = plan_path.parent.resolve()
    configured_run_root = plan.get("run_root")
    configured_candidate = (
        (project_root / configured_run_root).resolve()
        if configured_run_root else None
    )
    if (plan_parent / "sources").exists():
        run_dir = plan_parent
    elif configured_candidate is not None and configured_candidate.exists():
        run_dir = configured_candidate
    else:
        run_dir = plan_parent

    source_data_dir = aggregate_dir / "figure_source_data"
    figures_dir = aggregate_dir / "figures"
    source_data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return FigureContext(project_root, plan_path, aggregate_dir, run_dir, source_data_dir, figures_dir, plan)


def load_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def empty_panel(ax, text: str = "No evaluable observations") -> None:
    ax.axis("off")
    ax.text(
        0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes,
        fontsize=7.0, color=PALETTE["neutral_dark"],
    )


def plot_source_curve(
    ax,
    frame: pd.DataFrame,
    *,
    group_col: str = "arm",
    x_col: str = "x",
    value_col: str = "value",
    order: Sequence[str] | None = None,
    labels: Mapping[str, str] | None = None,
    colors: Mapping[str, str] | None = None,
    markers: Mapping[str, str] | None = None,
    linestyles: Mapping[str, str] | None = None,
    band: bool = True,
):
    """Plot source-level mean learning curves with SEM bands.

    The data contract is unchanged: each row remains one source-level value at
    one x value.  Styling intentionally distinguishes conditions without using
    heavy strokes or saturated colors.
    """
    if frame.empty:
        empty_panel(ax)
        return
    categories = list(order) if order is not None else list(dict.fromkeys(frame[group_col].astype(str)))
    for category in categories:
        g = frame[frame[group_col].astype(str) == str(category)]
        if g.empty:
            continue
        rows = []
        for x, q in g.groupby(x_col):
            vals = q[value_col].to_numpy(float)
            m, se = mean_sem(vals)
            rows.append((float(x), m, se))
        rows.sort(key=lambda z: z[0])
        arr = np.asarray(rows, dtype=float)
        color = (colors or COLORS).get(str(category), PALETTE["neutral_dark"])
        marker = (markers or MARKERS).get(str(category), "o")
        text = (labels or LABELS).get(str(category), label(category))
        linestyle = (linestyles or {}).get(str(category), "-")
        ax.plot(
            arr[:, 0], arr[:, 1], marker=marker, ms=MARKERSIZE,
            lw=LINEWIDTH, color=color, ls=linestyle, label=text,
            markeredgecolor="white", markeredgewidth=0.5,
        )
        if band and len(arr) > 1:
            ax.fill_between(
                arr[:, 0], arr[:, 1] - arr[:, 2], arr[:, 1] + arr[:, 2],
                color=color, alpha=BAND_ALPHA, linewidth=0, zorder=0,
            )
    style_axis(ax)


def plot_source_points(
    ax,
    frame: pd.DataFrame,
    *,
    category_col: str = "arm",
    value_col: str = "value",
    order: Sequence[str] | None = None,
    reference: float | None = None,
    ylabel: str = "",
    paired_lines: bool = False,
    rotate: float = 0.0,
):
    if frame.empty:
        empty_panel(ax)
        return
    cats = list(order) if order is not None else list(dict.fromkeys(frame[category_col].astype(str)))
    if paired_lines and "source_seed" in frame:
        for _, g in frame.groupby("source_seed"):
            vals = []
            for c in cats:
                q = g[g[category_col].astype(str) == str(c)][value_col]
                vals.append(float(q.mean()) if len(q) else np.nan)
            if np.isfinite(vals).sum() >= 2:
                ax.plot(
                    np.arange(len(cats)), vals, color=PALETTE["neutral_mid"],
                    lw=0.55, alpha=0.28, zorder=1,
                )
    for i, c in enumerate(cats):
        vals = frame[frame[category_col].astype(str) == str(c)][value_col].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        color = COLORS.get(str(c), PALETTE["navigation"])
        ax.scatter(
            i + deterministic_jitter(len(vals), width=0.085), vals,
            s=SWARM_SIZE, color=color, alpha=POINT_ALPHA, linewidth=0, zorder=2,
        )
        if len(vals):
            m, lo, hi = mean_ci95(vals)
            ax.errorbar(
                i, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=MS_SUMMARY,
                mfc=color, mec=PALETTE["black"], mew=0.6, color=color,
                capsize=CAPSIZE, elinewidth=ERROR_LINEWIDTH, zorder=4,
            )
    if reference is not None:
        reference_line(ax, reference)
    ax.set_xticks(
        range(len(cats)), [label(c) for c in cats], rotation=rotate,
        ha="right" if rotate else "center",
    )
    if ylabel:
        ax.set_ylabel(ylabel)
    style_axis(ax)


def plot_forest(
    ax,
    contrasts: pd.DataFrame,
    paired: pd.DataFrame,
    *,
    scale: float = 1.0,
    xlabel: str = "Paired difference",
):
    if contrasts.empty:
        empty_panel(ax)
        return
    contrasts = contrasts.reset_index(drop=True)
    for i, row in contrasts.iterrows():
        c = row["contrast"]
        vals = paired[paired["contrast"] == c]["difference"].to_numpy(float) * scale if not paired.empty else np.array([])
        vals = vals[np.isfinite(vals)]
        ax.scatter(
            vals, np.full(len(vals), i) + deterministic_jitter(len(vals), width=0.085),
            s=SWARM_SIZE, color=PALETTE["navigation"], alpha=POINT_ALPHA,
            linewidth=0, zorder=2,
        )
        mean = float(row.get("mean", np.nan)) * scale
        lo = float(row.get("ci_low", np.nan)) * scale
        hi = float(row.get("ci_high", np.nan)) * scale
        if np.isfinite(mean):
            ax.errorbar(
                mean, i, xerr=[[mean - lo], [hi - mean]] if np.isfinite(lo + hi) else None,
                fmt="o", ms=MS_SUMMARY, color=PALETTE["navigation"],
                mfc=PALETTE["navigation"], mec=PALETTE["black"], mew=0.6,
                capsize=CAPSIZE, elinewidth=ERROR_LINEWIDTH, zorder=4,
            )
    names = []
    for _, row in contrasts.iterrows():
        left = label(row.get("left", "")); right = label(row.get("right", ""))
        names.append(f"{left} − {right}" if left or right else label(row["contrast"]))
    ax.set_yticks(range(len(contrasts)), names, fontsize=6.5)
    ax.invert_yaxis()
    reference_line(ax, 0, orientation="v")
    ax.set_xlabel(xlabel)
    style_axis(ax)


def manifest_for_figure(context: FigureContext, figure: str, script: Path, inputs: Iterable[Path], outputs: Mapping[str, Path], notes: Sequence[str] | None = None) -> dict:
    items={}
    for p in inputs:
        p=Path(p)
        if p.exists() and p.is_file():
            try: key=str(p.resolve().relative_to(context.project_root))
            except ValueError: key=str(p.resolve())
            items[key]=sha256(p)
    outmap={}
    for k,p in outputs.items():
        p=Path(p)
        try: outmap[k]=str(p.resolve().relative_to(context.project_root))
        except ValueError: outmap[k]=str(p.resolve())
    try:
        script_name = str(script.resolve().relative_to(context.project_root))
    except ValueError:
        script_name = str(script.resolve())
    return {
        "figure": figure,
        "run_id": context.plan["config"]["run_id"],
        "script": script_name,
        "script_sha256": sha256(script),
        "inputs": items,
        "outputs": outmap,
        "n_navigation_sources_planned": context.n_sources,
        "checkpoint_policy": context.final_checkpoint_policy,
        "style_contract": {
            "backend": "Python/matplotlib",
            "final_width_in": 6.85,  # v2: 174 mm (FIG_WIDTH in nature_style.py)
            "editable_svg_text": True,
            "png_dpi": 600,
        },
        "notes": list(notes or []),
    }


def plot_named_source_points(
    ax,
    frame: pd.DataFrame,
    specs: Sequence[tuple[str, str, str]],
    *,
    reference: float | None = None,
    ylabel: str = "",
    paired_lines: bool = False,
    rotate: float = 0.0,
):
    """Compact source-level point plot with explicit display labels/colors."""
    if frame.empty:
        empty_panel(ax); return
    keys = [s[0] for s in specs]
    if paired_lines and "source_seed" in frame:
        for _, g in frame.groupby("source_seed"):
            vals = []
            for key in keys:
                q = g[g.arm.astype(str) == str(key)].value
                vals.append(float(q.mean()) if len(q) else np.nan)
            if np.isfinite(vals).sum() >= 2:
                ax.plot(
                    np.arange(len(keys)), vals, color=PALETTE["neutral_mid"],
                    lw=0.55, alpha=0.28, zorder=1,
                )
    for i, (key, text, color) in enumerate(specs):
        vals = frame[frame.arm.astype(str) == str(key)].value.to_numpy(float)
        vals = vals[np.isfinite(vals)]
        ax.scatter(
            i + deterministic_jitter(len(vals), width=0.075), vals,
            s=SWARM_SIZE, color=color, alpha=POINT_ALPHA, lw=0, zorder=2,
        )
        if len(vals):
            m, lo, hi = mean_ci95(vals)
            ax.errorbar(
                i, m, yerr=[[m - lo], [hi - m]], fmt="o", ms=MS_SUMMARY,
                color=color, mfc=color, mec=PALETTE["black"], mew=0.6,
                capsize=CAPSIZE, elinewidth=ERROR_LINEWIDTH, zorder=4,
            )
    if reference is not None:
        reference_line(ax, reference)
    ax.set_xticks(
        range(len(specs)), [s[1] for s in specs], rotation=rotate,
        ha="right" if rotate else "center",
    )
    if ylabel:
        ax.set_ylabel(ylabel)
    style_axis(ax)


def plot_paired_named(
    ax,
    paired: pd.DataFrame,
    specs: Sequence[tuple[str, str, str]],
    *,
    xlabel: str,
    xlim: tuple[float, float] | None = None,
    shade_positive: bool = False,
):
    """Horizontal paired effects with source points and mean 95% CI."""
    if paired.empty or not specs:
        empty_panel(ax); return
    n = len(specs)
    all_vals = paired[paired.contrast.astype(str).isin([s[0] for s in specs])].difference.to_numpy(float)
    all_vals = all_vals[np.isfinite(all_vals)]
    if shade_positive and len(all_vals):
        upper = max(float(np.nanmax(all_vals)) * 1.12, 0.01)
        ax.axvspan(0, upper, color=PALETTE["navigation"], alpha=0.035, lw=0, zorder=0)
    for i, (contrast, text, color) in enumerate(specs):
        y = n - 1 - i
        vals = paired[paired.contrast.astype(str) == str(contrast)].difference.to_numpy(float)
        vals = vals[np.isfinite(vals)]
        ax.scatter(
            vals, np.full(len(vals), y) + deterministic_jitter(len(vals), width=0.075),
            s=SWARM_SIZE, color=color, alpha=POINT_ALPHA, lw=0, zorder=2,
        )
        if len(vals):
            m, lo, hi = mean_ci95(vals)
            ax.errorbar(
                m, y, xerr=[[m - lo], [hi - m]], fmt="o", ms=MS_SUMMARY,
                color=color, mfc=color, mec=PALETTE["black"], mew=0.6,
                capsize=CAPSIZE, elinewidth=ERROR_LINEWIDTH, zorder=4,
            )
    reference_line(ax, 0, orientation="v")
    ax.set_yticks(range(n), [s[1] for s in specs][::-1], fontsize=6.5)
    ax.set_xlabel(xlabel)
    ax.set_ylim(-0.55, n - 0.45)
    if xlim is not None:
        ax.set_xlim(*xlim)
    style_axis(ax)

