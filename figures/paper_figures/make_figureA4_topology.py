#!/usr/bin/env python3
"""FigureA4 — Euclidean Schøyen automatic torus topology."""
from __future__ import annotations

from pathlib import Path
import os
import json
import hashlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import FigureContext, empty_panel, load_csv, manifest_for_figure, write_json
from .nature_style import (
    LINEWIDTH_SECONDARY, MARKERSIZE, PALETTE, place_panel_labels, panel_letter, legend, annotate,
    apply_publication_style, reference_line, save_publication_figure, style_axis,
)


def _euclidean_min(frame: pd.DataFrame) -> pd.Series:
    """Euclidean-only robust minimum persistence gap used throughout Figure A4."""
    needed = ["euclidean_H1_2_to_3", "euclidean_H2_1_to_2"]
    if not set(needed).issubset(frame.columns):
        return pd.Series(np.nan, index=frame.index, dtype=float)
    x = frame[needed].apply(pd.to_numeric, errors="coerce")
    return x.min(axis=1)


def _euclidean_pass(frame: pd.DataFrame, h1_thr: float, h2_thr: float) -> pd.Series:
    h1 = pd.to_numeric(frame.get("euclidean_H1_2_to_3"), errors="coerce")
    h2 = pd.to_numeric(frame.get("euclidean_H2_1_to_2"), errors="coerce")
    return h1.ge(h1_thr) & h2.ge(h2_thr)


def _finite_lifetimes(dgm: np.ndarray) -> np.ndarray:
    q = np.asarray(dgm, dtype=float)
    if q.ndim != 2 or q.shape[1] < 2 or len(q) == 0:
        return np.empty(0, dtype=float)
    finite = np.isfinite(q[:, 1])
    z = q[finite, 1] - q[finite, 0]
    z = z[np.isfinite(z) & (z > 0)]
    return np.sort(z)[::-1]


def _gap_scores_from_embedding(points: np.ndarray) -> tuple[float, float, int]:
    """Low-cost visualization-only PH diagnostic for a UMAP restart.

    This never replaces the formal PCA6 topology analysis.  It is used only to
    prefer a UMAP restart whose *visualization embedding* retains the expected
    toroidal signature instead of cutting a periodic direction open.
    """
    try:
        from ripser import ripser
    except Exception:
        return float("nan"), float("nan"), -1
    n_perm = min(300, len(points))
    r = ripser(points, maxdim=2, coeff=47, n_perm=n_perm, metric="euclidean")
    dgms = r.get("dgms", [])
    if len(dgms) < 3:
        return float("nan"), float("nan"), -1
    h0_inf = int(np.sum(~np.isfinite(np.asarray(dgms[0])[:, 1]))) if len(dgms[0]) else 0
    h1 = _finite_lifetimes(dgms[1]); h2 = _finite_lifetimes(dgms[2])
    h1_gap = float(h1[1] / max(h1[2], 1e-12)) if len(h1) >= 3 else float("nan")
    h2_gap = float(h2[0] / max(h2[1], 1e-12)) if len(h2) >= 2 else float("nan")
    return h1_gap, h2_gap, h0_inf


def _normalize_schoyen(pointcloud: np.ndarray) -> np.ndarray:
    """Match the normalization used in the Schøyen analysis notebook."""
    z = np.asarray(pointcloud, dtype=float)
    z = z - np.mean(z, axis=0, keepdims=True)
    scale = float(np.max(np.abs(z)))
    return z / max(scale, 1e-12)


def _artifact_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    stat = path.stat()
    h.update(str(path.resolve()).encode())
    h.update(str(stat.st_size).encode())
    h.update(str(stat.st_mtime_ns).encode())
    return h.hexdigest()


def _schoyen_visual_embedding(ctx: FigureContext, data_dir: Path):
    """Recompute a Schøyen-style visualization from the saved PCA6 population.

    The scientific topology decision remains the already-frozen PCA6/Ripser
    result.  This helper changes visualization only.

    Compared with the old FigureA4C, it follows the published Schøyen workflow:
      PCA6 population -> UMAP3 (Euclidean, 4000 neighbours, min_dist=.8,
      spectral init) -> use the existing Gardner/Schøyen fuzzy-downsample
      indices -> center/normalize -> color by PC1.

    The original repository does not set random_state for UMAP.  To avoid a
    single unlucky cut of a periodic direction, we run a small *predeclared*
    set of restarts and choose the one that best preserves topology in the
    visualization embedding, breaking ties by kNN trustworthiness.  This
    selection affects the picture only and is recorded in a CSV sidecar.
    """
    seed = ctx.representative_source
    artifact = ctx.analysis_source_dir(seed) / "artifacts" / "torus_topology" / "torus_analysis_arrays.npz"

    cache = data_dir / "representative_schoyen_umap3.csv"
    metrics_path = data_dir / "representative_schoyen_umap_restarts.csv"
    meta_path = data_dir / "representative_schoyen_umap_meta.json"

    # Local figure-polish bundles may contain the already selected UMAP3 cache
    # but not the original per-source torus_analysis_arrays.npz. In that case
    # use the frozen cached visualization directly rather than falling back to
    # the older representative_umap.csv.
    if not artifact.exists():
        if cache.exists():
            try:
                cached_frame = pd.read_csv(cache)
                cached_metrics = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
                return cached_frame, cached_metrics, None
            except Exception:
                pass
        return None

    fingerprint = _artifact_fingerprint(artifact)

    # Visualization cache contract.  The v7 cache was keyed only by the input
    # artifact and could therefore silently reuse a cache created with a
    # different number of restarts.  Keep the visualization parameters in the
    # cache key and allow an explicit forced recomputation on an LSF compute
    # node.
    n_restarts = max(1, int(os.environ.get("FIGA4_UMAP_RESTARTS", "8")))
    base_seed = int(os.environ.get("FIGA4_UMAP_BASE_SEED", "0"))
    force_recompute = str(os.environ.get("FIGA4_FORCE_RECOMPUTE", "0")).strip().lower() in {"1", "true", "yes", "on"}
    requested_parameters = {
        "n_components": 3,
        "min_dist": 0.8,
        "metric": "euclidean",
        "init": "spectral",
        "n_restarts": int(n_restarts),
        "base_seed": int(base_seed),
    }
    if (not force_recompute) and cache.exists() and metrics_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cached = dict(meta.get("parameters", {}))
            cache_ok = (
                meta.get("artifact_fingerprint") == fingerprint
                and int(meta.get("n_restarts", -1)) == int(n_restarts)
                and int(meta.get("base_seed", -1)) == int(base_seed)
                and int(cached.get("n_components", -1)) == 3
                and float(cached.get("min_dist", float("nan"))) == 0.8
                and str(cached.get("metric", "")).lower() == "euclidean"
                and str(cached.get("init", "")).lower() == "spectral"
            )
            if cache_ok:
                return pd.read_csv(cache), pd.read_csv(metrics_path), artifact
        except Exception:
            pass

    with np.load(artifact, allow_pickle=False) as a:
        required = {"pca6_spatial_population", "downsample_indices"}
        if not required.issubset(a.files):
            return None
        X6 = np.asarray(a["pca6_spatial_population"], dtype=np.float64)
        inds = np.asarray(a["downsample_indices"], dtype=np.int64).reshape(-1)
    inds = inds[(inds >= 0) & (inds < len(X6))]
    if X6.ndim != 2 or X6.shape[1] < 3 or len(inds) < 100:
        return None

    try:
        import umap
        from sklearn.manifold import trustworthiness
    except Exception as exc:
        raise RuntimeError("FigureA4 Schøyen visualization requires umap-learn and scikit-learn") from exc

    nn = min(4000, max(2, len(X6) - 1))
    rows = []
    embeddings = []
    Xds = X6[inds]
    for offset in range(n_restarts):
        rs = base_seed + offset
        emb = umap.UMAP(
            n_components=3,
            min_dist=0.8,
            n_neighbors=nn,
            metric="euclidean",
            init="spectral",
            random_state=rs,
        ).fit_transform(X6)
        uds = np.asarray(emb[inds], dtype=np.float64)
        try:
            tw = float(trustworthiness(Xds, uds, n_neighbors=min(50, max(5, len(uds)//20)), metric="euclidean"))
        except Exception:
            tw = float("nan")
        h1_gap, h2_gap, h0_inf = _gap_scores_from_embedding(uds)
        topo_score = min(h1_gap, h2_gap) if np.isfinite(h1_gap) and np.isfinite(h2_gap) else float("nan")
        passes = bool(h0_inf == 1 and np.isfinite(h1_gap) and np.isfinite(h2_gap) and h1_gap >= 1.5 and h2_gap >= 1.5)
        rows.append(dict(random_state=rs, trustworthiness=tw, H0_infinite_components=h0_inf,
                         H1_second_to_third=h1_gap, H2_first_to_second=h2_gap,
                         embedding_min_gap=topo_score, embedding_passes_121=passes))
        embeddings.append(np.asarray(emb, dtype=np.float64))

    m = pd.DataFrame(rows)
    # Publication-safe selection: first prefer embeddings preserving the 1,2,1
    # signature; among them maximize min gap; trustworthiness is the tie-break.
    rank = []
    for i, row in m.iterrows():
        p = 1 if bool(row["embedding_passes_121"]) else 0
        score = float(row["embedding_min_gap"]) if np.isfinite(row["embedding_min_gap"]) else -np.inf
        tw = float(row["trustworthiness"]) if np.isfinite(row["trustworthiness"]) else -np.inf
        rank.append((p, score, tw, -int(row["random_state"]), i))
    best_i = max(rank)[-1]
    m["selected"] = False
    m.loc[best_i, "selected"] = True

    emb = _normalize_schoyen(embeddings[best_i])[inds]
    pc1 = X6[inds, 0]
    frame = pd.DataFrame(dict(umap1=emb[:,0], umap2=emb[:,1], umap3=emb[:,2], pc1=pc1,
                              original_index=inds))
    frame.to_csv(cache, index=False)
    m.to_csv(metrics_path, index=False)
    meta = {
        "artifact": str(artifact),
        "artifact_fingerprint": fingerprint,
        "selected_random_state": int(m.loc[best_i, "random_state"]),
        "n_restarts": int(n_restarts),
        "base_seed": int(base_seed),
        "force_recompute_requested": bool(force_recompute),
        "visualization_only": True,
        "parameters": {**requested_parameters, "n_neighbors": int(nn)},
        "downsampled_points": int(len(inds)),
        "color": "PCA6 component 1",
        "note": "Topology inference is unchanged and remains based on the frozen PCA6 persistent-homology analysis.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return frame, m, artifact



def _parse_view_pairs(text: str):
    """Parse 'elev,azim;elev,azim;...' into a list of float pairs."""
    out = []
    for item in str(text).split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [p.strip() for p in item.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Invalid FigureA4C view spec: {item!r}")
        out.append((float(parts[0]), float(parts[1])))
    return out


def _camera_config(data_dir: Path):
    """Resolve the publication camera configuration for FigureA4C.

    Priority:
      1) explicit environment variables;
      2) figureA4C_view.json saved by the local interactive picker;
      3) the original four Schøyen-style views.

    Camera choice changes only the rendering of the already-frozen UMAP3 point
    cloud. It never changes the embedding or any topology statistic.
    """
    view_file_env = os.environ.get("FIGA4_C_VIEW_FILE", "").strip()
    view_file = Path(view_file_env).expanduser() if view_file_env else data_dir / "figureA4C_view.json"

    cfg = {}
    if view_file.exists():
        try:
            cfg = json.loads(view_file.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}

    default_views = [(0.0, 0.0), (0.0, 90.0), (45.0, 0.0), (45.0, 90.0)]
    views = default_views
    if isinstance(cfg.get("views"), list) and cfg["views"]:
        parsed = []
        for v in cfg["views"]:
            if isinstance(v, dict) and "elev" in v and "azim" in v:
                parsed.append((float(v["elev"]), float(v["azim"])))
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                parsed.append((float(v[0]), float(v[1])))
        if parsed:
            views = parsed
    elif "elev" in cfg and "azim" in cfg:
        views = [(float(cfg["elev"]), float(cfg["azim"]))]

    env_views = os.environ.get("FIGA4_C_VIEWS", "").strip()
    if env_views:
        views = _parse_view_pairs(env_views)
    else:
        elev_env = os.environ.get("FIGA4_C_ELEV", "").strip()
        azim_env = os.environ.get("FIGA4_C_AZIM", "").strip()
        if elev_env and azim_env:
            views = [(float(elev_env), float(azim_env))]

    layout = str(os.environ.get("FIGA4_C_LAYOUT", cfg.get("layout", "quad"))).strip().lower()
    if layout not in {"single", "quad"}:
        layout = "quad"

    if layout == "single":
        views = [views[0] if views else (20.0, -60.0)]
    else:
        if len(views) == 1:
            e, a = views[0]
            views = [(e, a), (e, a + 90.0), (e + 45.0, a), (e + 45.0, a + 90.0)]
        while len(views) < 4:
            views.append(default_views[len(views)])
        views = views[:4]

    return {
        "layout": layout,
        "views": [(float(e), float(a)) for e, a in views],
        "view_file": str(view_file) if view_file.exists() else None,
    }


def _equal_cube_limits(frame: pd.DataFrame, pad: float = 0.04):
    """Equal-scale 3-D limits that contain every displayed point.

    The previous renderer hard-coded +/-0.65 on all axes. That clips part of
    the current selected UMAP3 point cloud and can make a closed manifold look
    artificially open.
    """
    xyz = frame[["umap1", "umap2", "umap3"]].to_numpy(float)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    if len(xyz) == 0:
        return ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    lo = np.nanmin(xyz, axis=0)
    hi = np.nanmax(xyz, axis=0)
    center = (lo + hi) / 2.0
    radius = 0.5 * float(np.max(hi - lo))
    radius = max(radius * (1.0 + float(pad)), 1e-6)
    return tuple((float(c - radius), float(c + radius)) for c in center)


def _plot_schoyen_views(fig, subplotspec, frame: pd.DataFrame, camera_cfg=None):
    """Render the frozen Schøyen-style UMAP3 from user-controlled cameras."""
    camera_cfg = camera_cfg or {
        "layout": "quad",
        "views": [(0, 0), (0, 90), (45, 0), (45, 90)],
    }
    layout = camera_cfg["layout"]
    views = camera_cfg["views"]

    if layout == "single":
        gs = subplotspec.subgridspec(1, 1)
    else:
        gs = subplotspec.subgridspec(2, 2, wspace=0.00, hspace=0.00)

    vals = frame["pc1"].to_numpy(float)
    norm = plt.Normalize(np.nanmin(vals), np.nanmax(vals))
    cmap = plt.get_cmap("viridis")
    limits = _equal_cube_limits(
        frame, pad=float(os.environ.get("FIGA4_C_PAD", "0.04"))
    )
    point_size = float(
        os.environ.get("FIGA4_C_POINT_SIZE", "2.2" if layout == "single" else "1.8")
    )
    alpha = float(
        os.environ.get("FIGA4_C_ALPHA", "0.60" if layout == "single" else "0.55")
    )

    axes = []
    for i, (elev, azim) in enumerate(views):
        if layout == "single":
            ax = fig.add_subplot(gs[0, 0], projection="3d")
        else:
            ax = fig.add_subplot(gs[i // 2, i % 2], projection="3d")
        axes.append(ax)

        ax.scatter(
            frame.umap1, frame.umap2, frame.umap3,
            c=cmap(norm(vals)),
            s=point_size,
            alpha=alpha,
            linewidth=0,
            depthshade=False,
            rasterized=True,
        )
        ax.view_init(elev=float(elev), azim=float(azim))
        ax.set_xlim3d(*limits[0])
        ax.set_ylim3d(*limits[1])
        ax.set_zlim3d(*limits[2])
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(
        sm, ax=axes, orientation="horizontal",
        fraction=.035 if layout == "quad" else .028,
        pad=.005 if layout == "quad" else .015,
        aspect=32,
    )
    cb.set_ticks([float(np.nanmin(vals)), float(np.nanmax(vals))])
    cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=6.0, length=1.3, pad=1)
    cb.outline.set_visible(False)
    cb.set_label("PC1", fontsize=6.0, labelpad=-1)
    return axes


def _add_aligned_panel_label(fig, reference_ax, slot, label: str, *, reference_x: float = -.20) -> None:
    """Align a panel label horizontally with the label on a reference axis.

    3-D Matplotlib axes have different internal margins from ordinary 2-D axes,
    so placing panel letters with ``Axes3D.text2D`` can visibly shift them.
    FigureA4 uses this figure-coordinate helper for panel C so its letter is
    horizontally aligned with panel A.
    """
    x_display = reference_ax.transAxes.transform((reference_x, 0.0))
    x_figure = fig.transFigure.inverted().transform(x_display)[0]
    bbox = slot.get_position(fig)
    fig.text(
        x_figure, bbox.y1 + .012, panel_letter(label),
        fontsize=10.3, fontweight="bold", color=PALETTE["black"],
        ha="left", va="bottom", clip_on=False,
    )


def _cohomology_display_specs(frame: pd.DataFrame):
    """Return two scientifically interpretable color fields for panel C.

    Prefer the old FigureA4C convention (physical x/y) when the cohomological
    artifact could be mapped back to spatial bins.  Otherwise use the two
    topology-derived periodic coordinates themselves.
    """
    q = frame.copy()
    sx = pd.to_numeric(q.get("spatial_x"), errors="coerce")
    sy = pd.to_numeric(q.get("spatial_y"), errors="coerce")
    if sx is not None and sy is not None and np.isfinite(sx).sum() and np.isfinite(sy).sum():
        q["__display_1"] = sx
        q["__display_2"] = sy
        return q, [
                    (r"Spatial $x$", "__display_1", "viridis", r"spatial $x$"),
                    (r"Spatial $y$", "__display_2", "plasma", r"spatial $y$"),
                ]

    q["__display_1"] = np.cos(2.0 * np.pi * pd.to_numeric(q["theta1_turns"], errors="coerce"))
    q["__display_2"] = np.cos(2.0 * np.pi * pd.to_numeric(q["theta2_turns"], errors="coerce"))
    return q, [
        (r"Circular coordinate $\theta_1$", "__display_1", "twilight", r"$\cos(\theta_1)$"),
        (r"Circular coordinate $\theta_2$", "__display_2", "twilight_shifted", r"$\cos(\theta_2)$"),
    ]


def _plot_canonical_torus_pair(fig, subplotspec, frame: pd.DataFrame):
    """Draw two complementary views of the same topology-derived torus."""
    q, specs = _cohomology_display_specs(frame)
    cgs = subplotspec.subgridspec(1, 2, wspace=.035)

    elev1 = float(os.environ.get("FIGA4_COHOMOLOGY_ELEV1", "24"))
    azim1 = float(os.environ.get("FIGA4_COHOMOLOGY_AZIM1", "-55"))
    elev2 = float(os.environ.get("FIGA4_COHOMOLOGY_ELEV2", "24"))
    azim2 = float(os.environ.get("FIGA4_COHOMOLOGY_AZIM2", "-55"))
    cameras = [(elev1, azim1), (elev2, azim2)]

    point_size = float(os.environ.get("FIGA4_COHOMOLOGY_POINT_SIZE", "2.5"))
    alpha = float(os.environ.get("FIGA4_COHOMOLOGY_ALPHA", "0.64"))

    axes = []
    for i, ((title, color_key, cmap, color_label), (elev, azim)) in enumerate(zip(specs, cameras)):
        ax = fig.add_subplot(cgs[0, i], projection="3d")
        axes.append(ax)

        vals = pd.to_numeric(q[color_key], errors="coerce").to_numpy(float)
        good = (
            np.isfinite(q.torus_x.to_numpy(float))
            & np.isfinite(q.torus_y.to_numpy(float))
            & np.isfinite(q.torus_z.to_numpy(float))
            & np.isfinite(vals)
        )
        sc = ax.scatter(
            q.loc[good, "torus_x"], q.loc[good, "torus_y"], q.loc[good, "torus_z"],
            c=vals[good], cmap=cmap,
            s=point_size, alpha=alpha, linewidth=0,
            depthshade=False, rasterized=True,
        )
        ax.view_init(elev=elev, azim=azim)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass
        ax.set_axis_off()
        try:
            ax.set_box_aspect((1, 1, .58))
        except Exception:
            pass
        ax.set_title(title, loc="left", pad=-1, fontsize=7.0)

        cb = fig.colorbar(
            sc, ax=ax, orientation="horizontal",
            fraction=.035, pad=.005, aspect=24,
        )
        finite_vals = vals[np.isfinite(vals)]
        if len(finite_vals):
            vmin = float(np.nanmin(finite_vals))
            vmax = float(np.nanmax(finite_vals))
            cb.set_ticks([vmin, vmax])
            if color_key.startswith("__display") and title.startswith("Spatial"):
                cb.set_ticklabels(["low", "high"])
            else:
                cb.set_ticklabels(["-1", "1"])
        cb.ax.tick_params(labelsize=6.0, length=1.2, pad=1)
        cb.outline.set_visible(False)
        cb.set_label(color_label, fontsize=6.0, labelpad=-1)

    return axes


def build(ctx: FigureContext):
    apply_publication_style()
    data = ctx.source_data_dir / "FigureA4"
    out = ctx.figures_dir
    summary = load_csv(data / "torus_auto_summary.csv")
    metrics = load_csv(data / "representative_topology_metrics.csv")
    umap = load_csv(data / "representative_umap.csv")
    cands = load_csv(data / "representative_candidates.csv")
    coho = load_csv(data / "representative_cohomology_torus.csv")
    camera_cfg = None

    topo_cfg = ctx.plan["config"].get("schoyen_topology", {})
    h1_thr = float(topo_cfg.get("h1_gap_threshold", 1.5))
    h2_thr = float(topo_cfg.get("h2_gap_threshold", 1.5))
    # The manuscript run uses the same numerical threshold for H1 and H2.
    # Use the stricter value for the minimum-gap reference line.
    thr = max(h1_thr, h2_thr)

    fig = plt.figure(figsize=(7.25, 5.45))
    outer = fig.add_gridspec(
        2, 1, height_ratios=[.92, 1.08], left=.075, right=.992,
        top=.94, bottom=.105, hspace=.56,
    )
    top = outer[0].subgridspec(1, 2, width_ratios=[.43, .57], wspace=.34)
    bottom = outer[1].subgridspec(1, 2, width_ratios=[1.88, 1.00], wspace=.38)
    ax_a = fig.add_subplot(top[0, 0])
    ax_b = fig.add_subplot(top[0, 1])
    c_slot = bottom[0, 0]
    ax_d = fig.add_subplot(bottom[0, 1])

    # A — Euclidean-only cohort evidence.
    required = {"euclidean_H1_2_to_3", "euclidean_H2_1_to_2"}
    if summary.empty or not required.issubset(summary.columns):
        empty_panel(ax_a, "Cohort-level Euclidean topology scores unavailable")
    else:
        seed_col = "seed" if "seed" in summary.columns else "source_seed"
        z = summary.copy()
        z["euclidean_robust_min_gap"] = _euclidean_min(z)
        z = z.dropna(subset=["euclidean_robust_min_gap"])
        if seed_col in z.columns:
            z = z.sort_values(seed_col)
        if z.empty:
            empty_panel(ax_a, "Cohort-level Euclidean topology scores unavailable")
        else:
            x = np.arange(len(z))
            vals = z.euclidean_robust_min_gap.to_numpy(float)
            passed = _euclidean_pass(z, h1_thr, h2_thr).to_numpy(bool)
            colors = [PALETTE["navigation"] if p else PALETTE["neutral_mid"] for p in passed]
            ax_a.bar(x, vals, color=colors, alpha=.92, width=.72, linewidth=0)
            reference_line(ax_a, thr)
            # v2.5: sources numbered s = 1..10 in seed order (as in Figure 6A).
            labels = [str(i + 1) for i in range(len(z))]
            ax_a.set_xticks(x, labels)
            ax_a.set_xlabel("Navigation source $s$")
            ax_a.set_ylabel("Minimum Euclidean\npersistence-gap ratio")
            style_axis(ax_a, max_y_ticks=4)
            # Centred above the mid-height bars, the one gap that is free in
            # every source; typeset like every other in-panel statistic.
            #annotate(ax_a, f"Criterion met: {int(np.sum(passed))}/{len(z)}",
                     #loc="upper center")
    # v2: panel title removed (content belongs in the caption): "Torus-like criterion across sources"

    # B — representative Euclidean persistence only.
    if metrics.empty or not {"metric", "arm", "x", "value"}.issubset(metrics.columns):
        empty_panel(ax_b)
    else:
        palette = {
            "euclidean_H1": PALETTE["navigation"],
            "euclidean_H2": PALETTE["relation"],
        }
        labels = {
            "euclidean_H1": r"Euclidean $H_1$",
            "euclidean_H2": r"Euclidean $H_2$",
        }
        q = metrics[(metrics.metric == "lifetime") & metrics.arm.astype(str).isin(palette)].copy()
        for arm in ["euclidean_H1", "euclidean_H2"]:
            g = q[q.arm.astype(str) == arm].sort_values("x")
            if g.empty:
                continue
            ax_b.plot(
                g.x, g.value, marker="o", ms=MARKERSIZE - 1.0,
                markeredgecolor="white", markeredgewidth=.3,
                lw=LINEWIDTH_SECONDARY, color=palette[arm],
                ls="-", label=labels[arm],
            )
        ax_b.set_xlabel("Persistence rank")
        ax_b.set_ylabel("Persistence lifetime")
        # The rank axis is an integer index: keep integer ticks on it.
        ranks = sorted({int(v) for v in q.x.dropna().to_numpy(float)})
        if ranks:
            step = max(1, int(np.ceil(len(ranks) / 5)))
            ax_b.set_xticks(ranks[::step])
        legend(ax_b, loc="upper right", ncol=1)
        style_axis(ax_b, max_y_ticks=4, max_x_ticks=5)
    # v2: panel title removed; the source id stays visible in the panel.
    rep_s = sorted(int(p["source_seed"]) for p in ctx.source_pairs).index(int(ctx.representative_source)) + 1
    ax_b.text(0.5, 0.98, f"source $s$ = {rep_s}", transform=ax_b.transAxes, ha="center", va="top", fontsize=6.3)

    # C — topology-derived circular-coordinate rendering.
    #
    # A/B/D remain the frozen topology statistics.  C is a representative
    # visualization only: the donut geometry is the standard canonical torus
    # parameterization of the two cohomological circle coordinates.
    c_title = "Topology-derived canonical torus"
    coho_required = {
        "theta1_turns", "theta2_turns",
        "torus_x", "torus_y", "torus_z", "pc1",
    }
    if (not coho.empty) and coho_required.issubset(coho.columns):
        q = coho.copy()
        for col in coho_required:
            q[col] = pd.to_numeric(q[col], errors="coerce")
        q = q.dropna(subset=list(coho_required))
        if q.empty:
            ax_c = fig.add_subplot(c_slot)
            empty_panel(ax_c, "Cohomological torus coordinates unavailable")
            c_axes = [ax_c]
        else:
            c_axes = _plot_canonical_torus_pair(fig, c_slot, q)
    else:
        # Backward-compatible fallback only. If the cohomological artifact is
        # absent, retain the previous UMAP3 visualization.
        c_title = "Schøyen-style PCA6 → UMAP3 (fallback)"
        camera_cfg = _camera_config(data)
        sch = _schoyen_visual_embedding(ctx, data)
        if sch is None:
            if umap.empty or not {"umap1", "umap2", "umap3"}.issubset(umap.columns):
                ax_c = fig.add_subplot(c_slot)
                empty_panel(ax_c, "Cohomological torus artifact not yet computed")
                c_axes = [ax_c]
            else:
                fallback = umap.copy()
                fallback["pc1"] = pd.to_numeric(fallback.get("spatial_x"), errors="coerce")
                fallback[["umap1", "umap2", "umap3"]] = _normalize_schoyen(
                    fallback[["umap1", "umap2", "umap3"]].to_numpy(float)
                )
                c_axes = _plot_schoyen_views(
                    fig, c_slot, fallback.dropna(subset=["pc1"]), camera_cfg
                )
        else:
            sch_frame, sch_metrics, sch_artifact = sch
            c_axes = _plot_schoyen_views(fig, c_slot, sch_frame, camera_cfg)

    # Use figure coordinates for the C label.  This avoids the different
    # internal margins of Axes3D and aligns C horizontally with panel A.
    c_bbox = c_slot.get_position(fig)
    fig.text(
        c_bbox.x0, c_bbox.y1 + .012, c_title,
        fontsize=7.0, color=PALETTE["black"],
        ha="left", va="bottom", clip_on=False,
    )

    # D — Euclidean-only candidate comparison.
    required_c = {"euclidean_H1_2_to_3", "euclidean_H2_1_to_2"}
    if cands.empty or not required_c.issubset(cands.columns):
        empty_panel(ax_d, "Candidate Euclidean scores unavailable")
    else:
        q = cands.copy()
        q["euclidean_robust_min_gap"] = _euclidean_min(q)
        q = q.dropna(subset=["euclidean_robust_min_gap"])
        labels = q["clusters"].astype(str).tolist()
        x = np.arange(len(q))
        vals = q["euclidean_robust_min_gap"].to_numpy(float)
        passed = _euclidean_pass(q, h1_thr, h2_thr).to_numpy(bool)
        colors = [PALETTE["navigation"] if p else PALETTE["neutral_mid"] for p in passed]
        ax_d.bar(x, vals, color=colors, alpha=.92, width=.72, linewidth=0)
        reference_line(ax_d, thr)
        ax_d.set_xticks(x, [s.replace("_", "+") for s in labels], rotation=0, ha="center", fontsize=6.2)
        ax_d.set_xlabel("Candidate ensemble")
        ax_d.set_ylabel("Minimum Euclidean persistence-gap ratio")
        style_axis(ax_d, max_y_ticks=4)
    # v2: panel title removed (content belongs in the caption): "Representative candidate ensembles"

    # A, B and D are flush with their own panels; C is a 3-D group whose slot has
    # no tick labels, so it inherits panel A's letter position.
    letters = place_panel_labels(fig, [(ax_a, "A"), (ax_b, "B"), (ax_d, "D")])
    fig.text(
        letters[0].get_position()[0], c_slot.get_position(fig).y1 + .012, panel_letter("C"),
        fontsize=10.3, fontweight="bold", color=PALETTE["black"],
        ha="left", va="bottom", clip_on=False,
    )

    paths = save_publication_figure(fig, out / "FigureA4")
    inputs = [
        data / "torus_auto_summary.csv", data / "representative_topology_metrics.csv",
        data / "representative_umap.csv", data / "representative_schoyen_umap3.csv",
        data / "representative_schoyen_umap_restarts.csv", data / "representative_candidates.csv",
        data / "representative_cohomology_torus.csv",
        data / "representative_cohomology_torus_meta.json",
    ]
    if camera_cfg is not None and camera_cfg.get("view_file"):
        inputs.append(Path(camera_cfg["view_file"]))
    write_json(
        out / "manifests" / "FigureA4.json",
        manifest_for_figure(
            ctx, "FigureA4", Path(__file__), inputs, paths,
            notes=[
                "All topology decisions displayed in Figure A4 use the Euclidean persistence criteria only.",
                "Panels A and D use the minimum of the Euclidean H1 and H2 persistence gaps.",
                "Panel B displays Euclidean H1 and H2 persistence lifetimes separately.",
                "Panel C shows two complementary views/colorings of the same canonical torus derived from the two longest Euclidean H1 persistent-cohomology circular coordinates; it is visualization-only and does not alter the formal topology decision.",
                "When available, the two Panel C views are colored by physical spatial x and y (matching the earlier FigureA4C convention); otherwise they are colored by cos(theta1) and cos(theta2).",
                "The canonical donut geometry is imposed by the standard torus parameterization and must not be interpreted as the raw Euclidean geometry of the neural point cloud.",
                "Circular coordinates are identifiable only up to phase offset, sign, permutation, and integral basis changes.",
                "No legacy high-grid-score module is used for cohort topology selection; the cohomology calculation here is only a representative-source visualization.",
            ],
        ),
    )
    return paths
