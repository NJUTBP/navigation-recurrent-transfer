"""Experiment 1 — Navigation pre-training.

Upgrades vs original (learning from reference code)
----------------------------------------------------
1. **Optimizer uses weight_decay=0** — W_hh regularisation is now handled
   inside the loss function via model.recurrent_l2(), not via Adam.
   Reference applies L2 only to recurrent weights.
2. **Default loss_type="ce"** — cross-entropy produces sharper grid patterns.
3. **Online data generation option** — config ``data.online=true`` enables
   infinite trajectory streaming (reference style), giving the model
   fresh trajectories every step instead of re-using a fixed dataset.
4. **DoG place cells** — enabled by default via config, matching reference.
5. **Cosine LR schedule** — optional, for long training runs.
6. **use_bias=False** forwarded to model.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from gpm_cmap.data.nav_env.dataset import NavDatasetConfig, NavTrajectoryDataset, NavOnlineDataset
from gpm_cmap.data.nav_env.open_field import OpenFieldConfig
from gpm_cmap.data.nav_env.place_cells import PlaceCellEncoder
from gpm_cmap.models.grid_rnn import GridNavigationModel
from gpm_cmap.tasks.navigation import (
    evaluate_navigation,
    evaluate_navigation_loss,
    train_navigation,
)
from gpm_cmap.utils import (
    configure_runtime,
    load_config,
    make_torch_generator,
    make_worker_init_fn,
    prepare_output_dir,
    save_config_snapshot,
    save_run_provenance,
    set_seed,
    write_json,
)
from gpm_cmap.utils.checkpointing import load_checkpoint, save_checkpoint, sha256_file
from gpm_cmap.utils.logging_utils import get_logger


# -----------------------------------------------------------------------
#  Data loaders
# -----------------------------------------------------------------------

def make_loaders(cfg):
    open_field = OpenFieldConfig(
        arena_size=cfg["data"]["arena_size"],
        dt=cfg["data"].get("dt", 0.05),
        speed_scale=cfg["data"].get("speed_scale", 0.13),
        sigma_theta=cfg["data"].get("sigma_theta", 5.0),
        max_speed=cfg["data"].get("max_speed", 0.5),
        border_region=cfg["data"].get("border_region", 0.03),
        boundary_mode=cfg["data"].get("boundary_mode", "avoid"),
    )
    ds_cfg = NavDatasetConfig(
        num_trajectories=cfg["data"]["num_trajectories"],
        seq_len=cfg["data"]["seq_len"],
        open_field=open_field,
        place_cells=cfg["data"]["place_cells"],
        place_sigma=cfg["data"]["place_sigma"],
        arena_size=cfg["data"]["arena_size"],
        seed=cfg["seed"],
        trajectory_seed=cfg["seed"],
        place_seed=cfg["seed"],
        use_dog=cfg["data"].get("use_dog", True),
        surround_scale=cfg["data"].get("surround_scale", 2.0),
        surround_scale_mode=cfg["data"].get("surround_scale_mode", "variance"),
    )

    # [NEW] online mode: fresh trajectories every step
    use_online = cfg["data"].get("online", False)
    if use_online:
        train_ds = NavOnlineDataset(
            ds_cfg,
            steps_per_epoch=cfg["data"].get("steps_per_epoch", 1000),
            batch_size=cfg["data"]["batch_size"],  # [FIX] use config batch_size
        )
        train_loader = DataLoader(train_ds, batch_size=None)
    else:
        full_ds = NavTrajectoryDataset(ds_cfg)
        n_train = int(0.8 * len(full_ds))
        n_val = len(full_ds) - n_train
        train_ds, _ = random_split(
            full_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(cfg["seed"]),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=int(cfg["data"].get("num_workers", 0)),
            generator=make_torch_generator(cfg["seed"]),
            worker_init_fn=make_worker_init_fn(cfg["seed"]),
        )

    # validation set is always static for reproducibility
    val_ds = NavTrajectoryDataset(NavDatasetConfig(
        num_trajectories=cfg["analysis"].get(
            "val_num_trajectories",
            max(64, cfg["data"]["num_trajectories"] // 5),
        ),
        seq_len=cfg["data"]["seq_len"],
        open_field=open_field,
        place_cells=cfg["data"]["place_cells"],
        place_sigma=cfg["data"]["place_sigma"],
        arena_size=cfg["data"]["arena_size"],
        seed=cfg["seed"],
        trajectory_seed=cfg["seed"] + 9999,
        place_seed=cfg["seed"],
        use_dog=cfg["data"].get("use_dog", True),
        surround_scale=cfg["data"].get("surround_scale", 2.0),
        surround_scale_mode=cfg["data"].get("surround_scale_mode", "variance"),
    ))
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        worker_init_fn=make_worker_init_fn(cfg["seed"] + 9999),
    )
    place_encoder = val_ds.place_encoder

    return train_loader, val_loader, place_encoder


def make_topology_loader(cfg):
    """Create a longer, reset-sparse evaluation set for manifold analysis.

    Training/validation trajectories are only 20 steps in the supplied
    configs.  Flattening thousands of such short, independently initialised
    snippets can make the manifold reflect the place-code initialiser and
    trajectory resets rather than recurrent dynamics.  The topology loader is
    therefore generated separately and is used only for the selected final
    checkpoint.
    """
    torus_cfg = _torus_defaults(cfg.get("analysis", {}).get("torus", {}))
    if not bool(torus_cfg.get("enabled", True)):
        return None, None

    open_field = OpenFieldConfig(
        arena_size=cfg["data"]["arena_size"],
        dt=cfg["data"].get("dt", 0.05),
        speed_scale=cfg["data"].get("speed_scale", 0.13),
        sigma_theta=cfg["data"].get("sigma_theta", 5.0),
        max_speed=cfg["data"].get("max_speed", 0.5),
        border_region=cfg["data"].get("border_region", 0.03),
        boundary_mode=cfg["data"].get("boundary_mode", "avoid"),
    )
    dataset = NavTrajectoryDataset(NavDatasetConfig(
        num_trajectories=int(torus_cfg["eval_num_trajectories"]),
        seq_len=int(torus_cfg["eval_seq_len"]),
        open_field=open_field,
        place_cells=cfg["data"]["place_cells"],
        place_sigma=cfg["data"]["place_sigma"],
        arena_size=cfg["data"]["arena_size"],
        seed=cfg["seed"],
        trajectory_seed=cfg["seed"] + 20_001,
        place_seed=cfg["seed"],
        use_dog=cfg["data"].get("use_dog", True),
        surround_scale=cfg["data"].get("surround_scale", 2.0),
        surround_scale_mode=cfg["data"].get("surround_scale_mode", "variance"),
    ))
    loader = DataLoader(
        dataset,
        batch_size=min(
            int(torus_cfg["eval_batch_size"]),
            int(torus_cfg["eval_num_trajectories"]),
        ),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        worker_init_fn=make_worker_init_fn(cfg["seed"] + 20_001),
    )
    return loader, dataset.place_encoder


# -----------------------------------------------------------------------
#  Plotting helpers
# -----------------------------------------------------------------------

def plot_grid_summary(eval_res, output_dir: Path, max_units_to_plot: int = 8) -> None:
    grid_scores = [float(item["grid_score"]) for item in eval_res["grid_results"]]
    plt.figure(figsize=(6, 4))
    plt.hist(grid_scores, bins=20)
    plt.title("Grid score distribution")
    plt.xlabel("grid score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_dir / "grid_score_hist.png", dpi=150)
    plt.close()

    top_units = sorted(eval_res["grid_results"], key=lambda x: float(x["grid_score"]),
                       reverse=True)[:max_units_to_plot]
    ncols = 4
    nrows = max(1, int(np.ceil(len(top_units) / ncols)))
    plt.figure(figsize=(3 * ncols, 3 * nrows))
    for idx, item in enumerate(top_units, start=1):
        plt.subplot(nrows, ncols, idx)
        plt.imshow(item["rate_map"], origin="lower")
        plt.title(f"u={item['unit']} gs={item['grid_score']:.2f}")
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "top_rate_maps.png", dpi=150)
    plt.close()


def plot_path_integration_scatter(true_pos, decoded_pos, output_path, max_pts=2000):
    n = min(max_pts, len(true_pos))
    idx = np.random.default_rng(0).choice(len(true_pos), n, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for dim, label in enumerate(["x", "y"]):
        ax = axes[dim]
        ax.scatter(true_pos[idx, dim], decoded_pos[idx, dim], s=2, alpha=0.3)
        lim = max(abs(true_pos[:, dim]).max(), abs(decoded_pos[:, dim]).max()) * 1.1
        ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=0.8)
        ax.set_xlabel(f"True {label}"); ax.set_ylabel(f"Decoded {label}")
        ax.set_title(f"Path integration: {label}"); ax.set_aspect("equal")
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()


def plot_path_integration_error_over_time(
    true_pos,
    decoded_pos,
    time_indices,
    output_path,
    *,
    sample_dt: float,
):
    """Plot long-rollout decoding error as a function of recurrent time.

    The supplied training configs unroll only 20 steps.  A clean torus in a
    short, repeatedly re-initialised validation set can therefore coexist with
    unstable autonomous dynamics.  This diagnostic makes long-horizon drift
    explicit rather than attributing every weak manifold to UMAP/PCA choices.
    """
    true_pos = np.asarray(true_pos, dtype=float)
    decoded_pos = np.asarray(decoded_pos, dtype=float)
    time_indices = np.asarray(time_indices).reshape(-1)
    if true_pos.shape != decoded_pos.shape or true_pos.shape[0] != time_indices.size:
        raise ValueError("Position arrays and time_indices must have matching samples")

    errors = np.linalg.norm(decoded_pos - true_pos, axis=-1)
    valid = np.isfinite(errors) & np.isfinite(time_indices)
    errors = errors[valid]
    steps = time_indices[valid].astype(int)
    unique_steps = np.unique(steps)
    means = np.asarray([np.mean(errors[steps == step]) for step in unique_steps])
    medians = np.asarray([np.median(errors[steps == step]) for step in unique_steps])
    q25 = np.asarray([np.quantile(errors[steps == step], 0.25) for step in unique_steps])
    q75 = np.asarray([np.quantile(errors[steps == step], 0.75) for step in unique_steps])
    seconds = unique_steps.astype(float) * float(sample_dt)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(seconds, means, label="mean error")
    ax.plot(seconds, medians, label="median error")
    ax.fill_between(seconds, q25, q75, alpha=0.2, label="interquartile range")
    ax.set_xlabel("time since recurrent initialisation (s)")
    ax.set_ylabel("position decoding error (m)")
    ax.set_title("Long-rollout path-integration stability")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_path_integration_drift(
    true_pos,
    decoded_pos,
    time_indices,
    *,
    sample_dt: float,
) -> dict:
    """Return compact long-rollout drift diagnostics for JSON output."""
    true_pos = np.asarray(true_pos, dtype=float)
    decoded_pos = np.asarray(decoded_pos, dtype=float)
    time_indices = np.asarray(time_indices).reshape(-1)
    errors = np.linalg.norm(decoded_pos - true_pos, axis=-1)
    valid = np.isfinite(errors) & np.isfinite(time_indices)
    errors = errors[valid]
    steps = time_indices[valid].astype(int)
    if errors.size == 0:
        return {}

    unique_steps = np.unique(steps)
    mean_by_step = np.asarray([
        np.mean(errors[steps == step]) for step in unique_steps
    ])
    seconds = unique_steps.astype(float) * float(sample_dt)
    if unique_steps.size >= 2:
        slope = float(np.polyfit(seconds, mean_by_step, deg=1)[0])
    else:
        slope = float("nan")

    cutoff = max(1, int(np.ceil(unique_steps.size * 0.25)))
    early_steps = unique_steps[:cutoff]
    late_steps = unique_steps[-cutoff:]
    early_mask = np.isin(steps, early_steps)
    late_mask = np.isin(steps, late_steps)
    return {
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
        "early_quarter_mean_error_m": float(np.mean(errors[early_mask])),
        "late_quarter_mean_error_m": float(np.mean(errors[late_mask])),
        "late_to_early_error_ratio": float(
            np.mean(errors[late_mask]) / max(np.mean(errors[early_mask]), 1e-12)
        ),
        "linear_error_slope_m_per_s": slope,
        "rollout_duration_s": float(seconds[-1]) if seconds.size else 0.0,
    }


def plot_umap_hidden(projection, positions, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    n = min(5000, len(projection))
    idx = np.random.default_rng(0).choice(len(projection), n, replace=False)
    for dim, (cmap, label) in enumerate([("viridis", "x-position"), ("magma", "y-position")]):
        ax = axes[dim]
        sc = ax.scatter(projection[idx, 0], projection[idx, 1],
                        c=positions[idx, dim], cmap=cmap, s=3, alpha=0.5)
        plt.colorbar(sc, ax=ax, label=label)
        ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
        ax.set_title(f"Hidden states by {label}")
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()


# -----------------------------------------------------------------------
#  Post-training topology analysis
# -----------------------------------------------------------------------

def plot_umap_torus_3d(projection, colour, output_path):
    """Plot the same 3-D UMAP point cloud from three camera angles."""
    projection = np.asarray(projection)
    colour = np.asarray(colour)
    if projection.ndim != 2 or projection.shape[1] != 3:
        raise ValueError(f"Expected a [N, 3] UMAP projection, got {projection.shape}")
    views = [(20, 35), (20, 155), (70, 35)]
    fig = plt.figure(figsize=(15, 5))
    for idx, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        sc = ax.scatter(
            projection[:, 0], projection[:, 1], projection[:, 2],
            c=colour, cmap="viridis", s=2, alpha=0.55, linewidths=0,
        )
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_zlabel("UMAP-3")
        ax.set_title(f"view {idx}")
    fig.colorbar(sc, ax=fig.axes, fraction=0.018, pad=0.02, label="PC1")
    fig.suptitle("PCA(6) → 3-D UMAP of one grid module")
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.05, top=0.88, wspace=0.08)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_umap_position_colours(projection, positions, output_path):
    """Show how physical x/y position maps onto the 3-D population manifold."""
    projection = np.asarray(projection)
    positions = np.asarray(positions)
    fig = plt.figure(figsize=(11, 5))
    for idx, (dim, label, cmap) in enumerate(
        [(0, "x-position", "viridis"), (1, "y-position", "magma")], start=1
    ):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        sc = ax.scatter(
            projection[:, 0], projection[:, 1], projection[:, 2],
            c=positions[:, dim], cmap=cmap, s=2, alpha=0.55, linewidths=0,
        )
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_zlabel("UMAP-3")
        ax.set_title(label)
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.08)
    fig.suptitle("Physical position on the inferred population manifold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_persistence_barcodes(diagrams, null_max_lifetime, output_path):
    """Draw compact H0/H1/H2 barcode panels with shuffle thresholds."""
    n_dims = min(3, len(diagrams))
    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 4), squeeze=False)
    for dim in range(n_dims):
        ax = axes[0, dim]
        dgm = np.asarray(diagrams[dim], dtype=float)
        if dgm.size:
            finite_death = dgm[np.isfinite(dgm[:, 1]), 1]
            max_death = float(finite_death.max()) if finite_death.size else 1.0
            display = dgm.copy()
            display[~np.isfinite(display[:, 1]), 1] = max_death * 1.05
            order = np.argsort(display[:, 1] - display[:, 0])[::-1]
            display = display[order][:40]
            for rank, (birth, death) in enumerate(display):
                ax.hlines(rank, birth, death, linewidth=1.2)
            ax.set_ylim(-1, len(display) + 1)
        threshold = null_max_lifetime.get(dim, float("nan"))
        threshold_text = (
            f"shuffle max lifetime={threshold:.3f}"
            if np.isfinite(threshold) else "shuffle control not run"
        )
        ax.set_title(f"H{dim}\n{threshold_text}")
        ax.set_xlabel("filtration radius")
        ax.set_ylabel("feature rank")
        ax.grid(alpha=0.2)
    fig.suptitle("Persistent cohomology of the six-PC fuzzy representation")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_cohomological_coordinates(positions, coordinates_turns, output_path):
    """Map the two decoded H1 circular coordinates back to physical space."""
    positions = np.asarray(positions)
    coordinates_turns = np.asarray(coordinates_turns)
    if coordinates_turns.ndim != 2 or coordinates_turns.shape[1] != 2:
        raise ValueError(
            f"Expected decoded coordinates [N, 2], got {coordinates_turns.shape}"
        )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for dim, ax in enumerate(axes):
        valid = np.isfinite(coordinates_turns[:, dim]) & np.isfinite(positions).all(axis=1)
        sc = ax.scatter(
            positions[valid, 0],
            positions[valid, 1],
            c=np.cos(2.0 * np.pi * coordinates_turns[valid, dim]),
            cmap="viridis",
            s=8,
            alpha=0.7,
            linewidths=0,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("x-position")
        ax.set_ylabel("y-position")
        ax.set_title(f"cos(decoded circular coordinate {dim + 1})")
        fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)
    fig.suptitle("Cohomological decoding on persistence landmarks")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pca_dimension_sweep(rows, output_path):
    """Plot topology diagnostics across retained PCA dimensionalities."""
    dims = np.asarray([row["n_pca"] for row in rows], dtype=int)
    gap = np.asarray([row["h1_second_to_third_gap"] for row in rows], dtype=float)
    h2 = np.asarray([row["h2_first"] for row in rows], dtype=float)
    evr = np.asarray([row["cumulative_explained_variance"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(dims, evr, marker="o")
    axes[0].set_xlabel("number of PCs")
    axes[0].set_ylabel("cumulative explained variance")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(alpha=0.25)

    finite_gap = np.where(np.isfinite(gap), gap, np.nan)
    axes[1].plot(dims, finite_gap, marker="o")
    axes[1].set_xlabel("number of PCs")
    axes[1].set_ylabel("H1 second / third lifetime")
    axes[1].grid(alpha=0.25)

    axes[2].plot(dims, h2, marker="o")
    axes[2].set_xlabel("number of PCs")
    axes[2].set_ylabel("longest H2 lifetime")
    axes[2].grid(alpha=0.25)
    fig.suptitle("PCA-dimensionality sensitivity of toroidal persistence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_pca_scree(explained_variance_ratio, n_retained, output_path):
    """Plot per-component and cumulative variance before choosing UMAP input."""
    evr = np.asarray(explained_variance_ratio, dtype=float)
    components = np.arange(1, evr.size + 1)
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.bar(components, evr, alpha=0.7, label="per-PC variance")
    ax1.set_xlabel("principal component")
    ax1.set_ylabel("explained variance ratio")
    ax1.set_xticks(components)
    ax1.axvline(float(n_retained) + 0.5, linestyle="--", linewidth=1.2)
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(components, np.cumsum(evr), marker="o", linewidth=1.2)
    ax2.set_ylabel("cumulative explained variance")
    ax2.set_ylim(0, 1.02)
    fig.suptitle(f"PCA scree; first {int(n_retained)} PCs passed to UMAP/topology")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _stride_from_interval(config, stride_key, interval_key):
    explicit = config.get(stride_key)
    if explicit is not None:
        return max(1, int(explicit))
    dt = float(config.get("sample_dt", 0.02))
    interval = float(config.get(interval_key, dt))
    if dt <= 0 or interval <= 0:
        return 1
    return max(1, int(np.floor(interval / dt + 0.5)))


def _torus_defaults(torus_config):
    defaults = {
        "enabled": True,
        "eval_num_trajectories": 64,
        "eval_seq_len": 500,
        "eval_batch_size": 16,
        "n_pca": 6,
        "pca_diagnostic_components": 15,
        "sample_dt": 0.02,
        # The paper visualised 250-ms-spaced samples and used approximately
        # 50-ms population vectors for topology.  The stride is derived from dt
        # unless explicitly overridden with *_stride_steps.
        "visual_sample_interval_s": 0.25,
        "topology_sample_interval_s": 0.05,
        "visual_stride_steps": None,
        "topology_stride_steps": None,
        "burn_in_steps": 50,
        # 2.5 cm/s at dt=20 ms, matching the paper's awake-state filter.
        "min_step_displacement": 0.0005,
        "grid_score_key": "grid_score_conservative",
        "grid_score_threshold": 0.3,
        "spacing_tolerance": 0.25,
        "orientation_tolerance_deg": 15.0,
        "exclude_head_direction": True,
        "head_direction_threshold": 0.3,
        "min_module_units": 60,
        "allow_underpowered_persistence": False,
        "max_visual_points": 30000,
        "umap_n_components": 3,
        "umap_n_neighbors": 5000,
        "umap_min_dist": 0.8,
        "umap_metric": "cosine",
        "umap_init": "spectral",
        "max_active_points": 15000,
        "n_landmarks": 1200,
        "density_neighbors": 1500,
        "persistence_neighbors": 800,
        "density_omega": 1.0,
        "activity_mode": "auto",
        "coeff": 47,
        "n_shuffles": 0,
        "shuffle_within_trajectories": True,
        "decode_cocycles": True,
        "pca_sweep_dims": [],
        "pca_sweep_landmarks": 600,
        "pca_sweep_density_neighbors": 800,
        "pca_sweep_persistence_neighbors": 500,
        # Standalone analyze_torus can split complete trajectories so that
        # grid-module selection and persistent cohomology are evaluated on
        # disjoint data. Integrated Exp1 analysis remains backwards-compatible
        # unless explicit split indices are supplied.
        "independent_module_selection": False,
        "module_selection_fraction": 0.5,
        "n_jobs": -1,
    }
    if torus_config:
        defaults.update(torus_config)
    return defaults


def run_topology_analysis(
    hidden,
    positions,
    output_dir,
    logger,
    seed=42,
    *,
    headings=None,
    grid_results=None,
    trajectory_ids=None,
    time_indices=None,
    step_displacements=None,
    torus_config=None,
    module_selection_input_indices=None,
    topology_evaluation_input_indices=None,
):
    """Run a Gardner-style module-specific torus analysis.

    The former implementation calculated six PCs but then ignored them:
    both UMAP and Ripser received the raw hidden matrix.  This version selects
    one coherent grid module, excludes strongly directional units, z-scores
    each unit, sends six PCs to 3-D UMAP for visualisation, and computes
    persistent cohomology on the six-PC fuzzy graph rather than on the UMAP
    embedding.
    """
    from gpm_cmap.analysis.gardner_torus import (
        json_safe_persistence_summary,
        pca_dimension_sweep,
        persistent_cohomology_gardner,
        prepare_umap_input,
        select_grid_module,
        umap_from_pcs,
    )

    cfg = _torus_defaults(torus_config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def _finish_early(payload):
        write_json(payload, output_dir / "torus_analysis_summary.json")
        return payload

    hidden = np.asarray(hidden)
    positions = np.asarray(positions)
    if hidden.ndim != 2 or positions.shape != (hidden.shape[0], 2):
        raise ValueError(
            f"Expected hidden [N,H] and positions [N,2], got {hidden.shape} and {positions.shape}"
        )

    n_samples = hidden.shape[0]

    def _metadata_vector(values, name):
        if values is None:
            return None
        arr = np.asarray(values).reshape(-1)
        if arr.shape[0] != n_samples:
            raise ValueError(
                f"{name} has {arr.shape[0]} samples but hidden has {n_samples}"
            )
        return arr

    headings = _metadata_vector(headings, "headings")
    trajectory_ids = _metadata_vector(trajectory_ids, "trajectory_ids")
    time_indices = _metadata_vector(time_indices, "time_indices")
    step_displacements = _metadata_vector(
        step_displacements, "step_displacements"
    )

    analysis_mask = np.isfinite(hidden).all(axis=1) & np.isfinite(positions).all(axis=1)
    if headings is not None:
        analysis_mask &= np.isfinite(headings)
    if time_indices is not None and int(cfg["burn_in_steps"]) > 0:
        analysis_mask &= time_indices >= int(cfg["burn_in_steps"])
    if step_displacements is not None and float(cfg["min_step_displacement"]) > 0:
        analysis_mask &= step_displacements > float(cfg["min_step_displacement"])
    if int(analysis_mask.sum()) < 10:
        return _finish_early({
            "status": "insufficient_samples",
            "n_samples": int(analysis_mask.sum()),
            "is_torus_like": None,
            "parameters": cfg,
        })

    def _index_mask(indices, name):
        if indices is None:
            return analysis_mask.copy()
        arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            raise ValueError(f"{name} is empty")
        if arr.min() < 0 or arr.max() >= n_samples:
            raise IndexError(f"{name} contains an out-of-range sample index")
        mask = np.zeros(n_samples, dtype=bool)
        mask[np.unique(arr)] = True
        mask &= analysis_mask
        return mask

    module_selection_mask = _index_mask(
        module_selection_input_indices, "module_selection_input_indices"
    )
    topology_evaluation_mask = _index_mask(
        topology_evaluation_input_indices, "topology_evaluation_input_indices"
    )
    split_requested = (
        module_selection_input_indices is not None
        or topology_evaluation_input_indices is not None
    )
    split_overlap = int(
        np.sum(module_selection_mask & topology_evaluation_mask)
    )
    if split_requested and split_overlap:
        raise ValueError(
            "Module-selection and topology-evaluation samples must be disjoint; "
            f"found {split_overlap} overlapping filtered samples"
        )
    if int(module_selection_mask.sum()) < 10:
        return _finish_early({
            "status": "insufficient_module_selection_samples",
            "n_samples": int(module_selection_mask.sum()),
            "is_torus_like": None,
            "parameters": cfg,
        })
    if int(topology_evaluation_mask.sum()) < 10:
        return _finish_early({
            "status": "insufficient_topology_evaluation_samples",
            "n_samples": int(topology_evaluation_mask.sum()),
            "is_torus_like": None,
            "parameters": cfg,
        })

    visual_stride = _stride_from_interval(
        cfg, "visual_stride_steps", "visual_sample_interval_s"
    )
    topology_stride = _stride_from_interval(
        cfg, "topology_stride_steps", "topology_sample_interval_s"
    )
    visual_mask = topology_evaluation_mask.copy()
    topology_mask = topology_evaluation_mask.copy()
    if time_indices is not None:
        phase = time_indices - int(cfg["burn_in_steps"])
        visual_mask &= np.mod(phase, visual_stride) == 0
        topology_mask &= np.mod(phase, topology_stride) == 0
    # Metadata-free legacy arrays cannot be downsampled without risking cuts
    # across unknown trajectory boundaries, so both masks remain unchanged.
    if int(visual_mask.sum()) < 10:
        visual_mask = topology_evaluation_mask.copy()
        visual_stride = 1
    if int(topology_mask.sum()) < 10:
        topology_mask = topology_evaluation_mask.copy()
        topology_stride = 1

    hidden_selection = hidden[module_selection_mask]
    headings_selection = (
        None if headings is None else headings[module_selection_mask]
    )
    hidden_reference = hidden[topology_evaluation_mask]

    result = {
        "status": "started",
        "n_samples_before_filter": int(hidden.shape[0]),
        "n_samples_after_filter": int(analysis_mask.sum()),
        "selection_evaluation_split": {
            "requested": bool(split_requested),
            "independent": bool(split_requested and split_overlap == 0),
            "n_module_selection_samples": int(module_selection_mask.sum()),
            "n_topology_evaluation_samples": int(topology_evaluation_mask.sum()),
            "n_overlap_samples": int(split_overlap),
        },
        "sampling": {
            "sample_dt": float(cfg["sample_dt"]),
            "visual_stride_steps": int(visual_stride),
            "topology_stride_steps": int(topology_stride),
            "n_visual_vectors_before_umap_subsample": int(visual_mask.sum()),
            "n_topology_vectors_before_active_selection": int(topology_mask.sum()),
            "trajectory_metadata_available": trajectory_ids is not None,
        },
        "parameters": cfg,
    }
    if trajectory_ids is not None:
        selection_trajectories = np.unique(trajectory_ids[module_selection_mask])
        topology_trajectories, trajectory_counts = np.unique(
            trajectory_ids[topology_evaluation_mask], return_counts=True
        )
        result["sampling"].update({
            "n_trajectories": int(topology_trajectories.size),
            "n_module_selection_trajectories": int(selection_trajectories.size),
            "n_topology_evaluation_trajectories": int(topology_trajectories.size),
            "trajectory_split_disjoint": bool(
                np.intersect1d(selection_trajectories, topology_trajectories).size == 0
                if split_requested else False
            ),
            "median_vectors_per_trajectory_after_filter": float(
                np.median(trajectory_counts)
            ),
            "min_vectors_per_trajectory_after_filter": int(trajectory_counts.min()),
            "max_vectors_per_trajectory_after_filter": int(trajectory_counts.max()),
        })
    if time_indices is not None:
        finite_times = time_indices[topology_evaluation_mask]
        result["sampling"].update({
            "max_time_index_after_filter": int(np.max(finite_times)),
            "max_time_s_after_filter": float(
                np.max(finite_times) * float(cfg["sample_dt"])
            ),
        })

    if grid_results is None:
        logger.warning("No grid_results supplied; refusing to mix unclassified hidden units.")
        result.update({
            "status": "missing_grid_results",
            "is_torus_like": None,
        })
        return _finish_early(result)

    selection = select_grid_module(
        grid_results,
        activations=hidden_selection,
        headings=headings_selection,
        score_key=cfg["grid_score_key"],
        score_threshold=float(cfg["grid_score_threshold"]),
        spacing_tolerance=float(cfg["spacing_tolerance"]),
        orientation_tolerance_deg=float(cfg["orientation_tolerance_deg"]),
        exclude_head_direction=bool(cfg["exclude_head_direction"]),
        head_direction_threshold=float(cfg["head_direction_threshold"]),
        min_module_units=int(cfg["min_module_units"]),
    )
    result["module_selection"] = selection.to_dict()
    units = selection.unit_indices
    logger.info(
        "Torus module selection: %d/%d candidates retained; spacing=%.3f bins; orientation=%.1f°; %s",
        selection.n_units,
        selection.n_candidates,
        selection.median_spacing,
        selection.mean_orientation_deg,
        selection.note,
    )
    if units.size < 2:
        result.update({
            "status": "no_coherent_grid_module",
            "is_torus_like": None,
        })
        return _finish_early(result)

    module_hidden_visual = hidden[visual_mask][:, units]
    module_hidden_topology = hidden[topology_mask][:, units]
    positions_visual = positions[visual_mask]
    positions_topology = positions[topology_mask]
    topology_trajectory_ids = (
        trajectory_ids[topology_mask]
        if trajectory_ids is not None
        and bool(cfg["shuffle_within_trajectories"])
        else None
    )

    # -------------------- PCA -> 3-D UMAP (visualisation only) --------------------
    diagnostic_components = max(
        int(cfg["n_pca"]), int(cfg["pca_diagnostic_components"])
    )
    vis = prepare_umap_input(
        module_hidden_visual,
        n_pca=diagnostic_components,
        max_points=int(cfg["max_visual_points"]),
        seed=seed,
        # Match the paper's order: standardise the complete movement-filtered
        # module trace, then temporally downsample for the UMAP figure.
        standardization_reference=hidden_reference[:, units],
    )
    n_umap_pcs = min(int(cfg["n_pca"]), vis["projection"].shape[1])
    if n_umap_pcs < 2:
        result.update({
            "status": "insufficient_pca_rank",
            "is_torus_like": None,
        })
        return _finish_early(result)
    umap_pcs = vis["projection"][:, :n_umap_pcs]
    vis_positions = positions_visual[vis["input_indices"]]
    result["pca_explained_variance"] = vis["explained_variance_ratio"].tolist()
    result["pca_components_used"] = int(n_umap_pcs)
    logger.info(
        "Module PCA explained variance (diagnostic top-%d): %s",
        len(vis["explained_variance_ratio"]),
        ", ".join(f"{v:.3f}" for v in vis["explained_variance_ratio"]),
    )
    plot_pca_scree(
        vis["explained_variance_ratio"],
        n_umap_pcs,
        output_dir / "pca_scree.png",
    )

    umap_projection_arr = None
    umap_plot_done = False
    try:
        umap_res = umap_from_pcs(
            umap_pcs,
            n_components=int(cfg["umap_n_components"]),
            n_neighbors=int(cfg["umap_n_neighbors"]),
            min_dist=float(cfg["umap_min_dist"]),
            metric=str(cfg["umap_metric"]),
            init=str(cfg["umap_init"]),
            seed=seed,
        )
        umap_projection_arr = umap_res["projection"]
        if umap_projection_arr.shape[1] == 3:
            plot_umap_torus_3d(
                umap_projection_arr,
                umap_pcs[:, 0],
                output_dir / "umap_torus_3d.png",
            )
            plot_umap_position_colours(
                umap_projection_arr,
                vis_positions,
                output_dir / "umap_torus_position.png",
            )
            umap_plot_done = True
        result["umap"] = {
            "done": True,
            "n_components": int(umap_res["n_components"]),
            "n_neighbors": int(umap_res["n_neighbors"]),
            "min_dist": float(umap_res["min_dist"]),
            "metric": umap_res["metric"],
            "init": umap_res["init"],
        }
        logger.info("PCA -> 3-D UMAP projection saved.")
    except ImportError:
        logger.warning("umap-learn not installed — PCA completed, UMAP skipped.")
        result["umap"] = {"done": False, "reason": "umap-learn not installed"}

    arrays = {
        "selected_unit_indices": units,
        # The shared-manifold source fit must use exactly the held-out
        # topology-evaluation trajectories, not the module-selection set.
        "analysis_input_indices": np.flatnonzero(topology_evaluation_mask),
        "module_selection_input_indices": np.flatnonzero(module_selection_mask),
        "topology_evaluation_input_indices": np.flatnonzero(topology_evaluation_mask),
        "visual_input_indices": np.flatnonzero(visual_mask),
        "topology_input_indices": np.flatnonzero(topology_mask),
        "umap_input_indices_within_visual": vis["input_indices"],
        "umap_input_indices_original": np.flatnonzero(visual_mask)[vis["input_indices"]],
        "umap_pca_projection": umap_pcs,
        "umap_pca_diagnostic_projection": vis["projection"],
        "umap_positions": vis_positions,
    }
    if trajectory_ids is not None:
        arrays["umap_trajectory_ids"] = trajectory_ids[visual_mask][vis["input_indices"]]
    if time_indices is not None:
        arrays["umap_time_indices"] = time_indices[visual_mask][vis["input_indices"]]
    if umap_projection_arr is not None:
        arrays["umap_projection"] = umap_projection_arr

    effective_module_units = int(np.sum(vis["retained_unit_mask"]))
    result["effective_nonconstant_module_units"] = effective_module_units
    underpowered = (
        not selection.sufficient_for_topology
        or effective_module_units < int(cfg["min_module_units"])
    )
    if underpowered:
        result["underpowered"] = True
        result["underpowered_reason"] = (
            f"Only {effective_module_units} non-constant units remained; "
            f"configured minimum is {int(cfg['min_module_units'])}."
        )
        logger.warning(result["underpowered_reason"])
    else:
        result["underpowered"] = False

    # -------------------- Persistent cohomology (not on UMAP) --------------------
    persistence_attempted = (
        not underpowered or bool(cfg["allow_underpowered_persistence"])
    )
    if not persistence_attempted:
        result["persistent_cohomology"] = {
            "done": False,
            "reason": "underpowered grid module",
        }
        result["betti_numbers"] = None
        result["is_torus_like"] = None
    else:
        try:
            ph_res = persistent_cohomology_gardner(
                module_hidden_topology,
                n_pca=int(cfg["n_pca"]),
                max_active_points=int(cfg["max_active_points"]),
                n_landmarks=int(cfg["n_landmarks"]),
                density_neighbors=int(cfg["density_neighbors"]),
                persistence_neighbors=int(cfg["persistence_neighbors"]),
                density_omega=float(cfg["density_omega"]),
                activity_mode=str(cfg["activity_mode"]),
                metric="cosine",
                max_dim=2,
                coeff=int(cfg["coeff"]),
                n_shuffles=int(cfg["n_shuffles"]),
                seed=seed,
                n_jobs=int(cfg["n_jobs"]),
                do_cocycles=bool(cfg["decode_cocycles"]),
                trajectory_ids=topology_trajectory_ids,
            )
            ph_summary = json_safe_persistence_summary(ph_res)
            result["persistent_cohomology"] = ph_summary
            result["betti_numbers"] = ph_summary["significant_counts"]
            result["is_torus_like"] = ph_summary["is_torus_like"]

            active_local = ph_res["preprocessing"]["active_input_indices"]
            landmark_local = ph_res["preprocessing"]["landmark_input_indices"]
            topology_original = np.flatnonzero(topology_mask)
            arrays["ph_active_input_indices_within_topology"] = active_local
            arrays["ph_active_input_indices_original"] = topology_original[active_local]
            arrays["ph_landmark_input_indices_within_topology"] = landmark_local
            arrays["ph_landmark_input_indices_original"] = topology_original[landmark_local]
            arrays["ph_landmark_pcs"] = ph_res["preprocessing"]["landmark_pcs"]
            for dim, dgm in enumerate(ph_res["diagrams"]):
                arrays[f"persistence_H{dim}"] = dgm

            decoded = ph_res.get("decoded_landmark_coordinates")
            decoded_plot_done = False
            if decoded is not None:
                landmark_positions = positions_topology[landmark_local]
                arrays["decoded_torus_coordinates_turns"] = decoded[
                    "coordinates_turns"
                ]
                arrays["decoded_torus_coordinates_radians"] = decoded[
                    "coordinates_radians"
                ]
                arrays["decoded_torus_landmark_positions"] = landmark_positions
                plot_cohomological_coordinates(
                    landmark_positions,
                    decoded["coordinates_turns"],
                    output_dir / "cohomological_coordinates.png",
                )
                decoded_plot_done = True
            result["cohomological_coordinates_plot"] = decoded_plot_done
            plot_persistence_barcodes(
                ph_res["diagrams"],
                ph_res["null_max_lifetime"],
                output_dir / "persistence_barcodes.png",
            )
            logger.info(
                "Persistent cohomology: significant=%s | torus=%s | H1 gap=%.3f",
                ph_summary["significant_counts"],
                ph_summary["is_torus_like"],
                ph_summary["h1_second_to_third_gap"]
                if ph_summary["h1_second_to_third_gap"] is not None
                else float("nan"),
            )
            if int(cfg["n_shuffles"]) < 1000:
                logger.warning(
                    "Only %d shuffle controls requested. The reference analysis "
                    "used 1000; treat this run as exploratory rather than "
                    "publication-grade.",
                    int(cfg["n_shuffles"]),
                )
        except ImportError:
            logger.warning("ripser not installed — persistent cohomology skipped.")
            result["persistent_cohomology"] = {
                "done": False,
                "reason": "ripser not installed",
            }
            result["betti_numbers"] = None
            result["is_torus_like"] = None

    # Optional sensitivity analysis: the paper empirically justified six PCs.
    # This is disabled by default because repeated H2 computations are costly.
    sweep_dims = cfg.get("pca_sweep_dims", [])
    if persistence_attempted and sweep_dims:
        try:
            sweep = pca_dimension_sweep(
                module_hidden_topology,
                dimensions=sweep_dims,
                max_active_points=int(cfg["max_active_points"]),
                n_landmarks=int(cfg["pca_sweep_landmarks"]),
                density_neighbors=int(cfg["pca_sweep_density_neighbors"]),
                persistence_neighbors=int(cfg["pca_sweep_persistence_neighbors"]),
                density_omega=float(cfg["density_omega"]),
                activity_mode=str(cfg["activity_mode"]),
                metric="cosine",
                max_dim=2,
                coeff=int(cfg["coeff"]),
                seed=seed,
                n_jobs=int(cfg["n_jobs"]),
            )
            result["pca_dimension_sweep"] = sweep["rows"]
            plot_pca_dimension_sweep(
                sweep["rows"], output_dir / "pca_dimension_sweep.png"
            )
        except ImportError:
            result["pca_dimension_sweep"] = None
        except ValueError as exc:
            logger.warning("PCA dimension sweep skipped: %s", exc)
            result["pca_dimension_sweep"] = None

    np.savez_compressed(output_dir / "torus_analysis_arrays.npz", **arrays)
    result["status"] = "completed_underpowered" if underpowered else "completed"
    result["artifacts"] = {
        "arrays": "torus_analysis_arrays.npz",
        "pca_scree": "pca_scree.png",
        "umap_3d": "umap_torus_3d.png" if umap_plot_done else None,
        "umap_position": "umap_torus_position.png" if umap_plot_done else None,
        "persistence_barcodes": (
            "persistence_barcodes.png"
            if result.get("persistent_cohomology", {}).get("lifetimes") is not None
            else None
        ),
        "cohomological_coordinates": (
            "cohomological_coordinates.png"
            if result.get("cohomological_coordinates_plot") else None
        ),
        "pca_dimension_sweep": (
            "pca_dimension_sweep.png"
            if result.get("pca_dimension_sweep") is not None else None
        ),
    }
    write_json(result, output_dir / "torus_analysis_summary.json")
    return result


# -----------------------------------------------------------------------
#  Main
# -----------------------------------------------------------------------

def _full_navigation_evaluation(
    cfg,
    model,
    val_loader,
    place_encoder,
    device,
    *,
    grid_filter=None,
):
    grid_filter = grid_filter or {}
    return evaluate_navigation(
        model,
        val_loader,
        device,
        arena_size=cfg["data"]["arena_size"],
        grid_bins=cfg["analysis"]["grid_bins"],
        grid_smooth_sigma=cfg["analysis"].get("grid_smooth_sigma", 0.0),
        place_encoder=place_encoder,
        include_google_ref_score=cfg["analysis"].get("include_google_ref_score", False),
        include_hidden_diagnostics=cfg["analysis"].get("include_hidden_diagnostics", False),
        grid_burn_in_steps=int(grid_filter.get("burn_in_steps", 0)),
        grid_min_step_displacement=float(
            grid_filter.get("min_step_displacement", 0.0)
        ),
    )


def _scalar_navigation_summary(stats: dict) -> dict:
    """Extract JSON-safe scientific metrics from a full navigation evaluation."""
    keys = [
        "loss",
        "place_target_entropy",
        "uniform_loss",
        "loss_excess_over_floor",
        "frac_loss_captured",
        "mean_grid_score",
        "max_grid_score",
        "frac_grid_above_03",
        "mean_grid_score_conservative",
        "max_grid_score_conservative",
        "frac_grid_above_03_conservative",
        "mean_grid_score_google",
        "max_grid_score_google",
        "frac_grid_above_03_google",
        "path_integration_error",
        "path_integration_error_relative",
        "hidden_mean",
        "hidden_std",
        "hidden_active_fraction",
        "dead_unit_fraction",
        "near_dead_unit_fraction",
        "hidden_participation_ratio",
        "hidden_offdiag_corr_mse",
    ]
    return {key: stats.get(key) for key in keys if key in stats}


def _analyse_navigation_checkpoint(
    *,
    tag: str,
    checkpoint_path: Path,
    cfg: dict,
    model,
    val_loader,
    place_encoder,
    topology_loader,
    topology_place_encoder,
    device,
    output_dir: Path,
    logger,
    run_topology: bool,
) -> dict:
    """Generate all arrays, figures and metrics from one explicit checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    stats = _full_navigation_evaluation(cfg, model, val_loader, place_encoder, device)

    analysis_dir = output_dir / "checkpoint_analysis" / tag
    analysis_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "positions": stats["positions"],
        "hidden": stats["hidden"],
        "headings": stats["headings"],
        "trajectory_ids": stats["trajectory_ids"],
        "time_indices": stats["time_indices"],
        "step_displacements": stats["step_displacements"],
    }
    if "decoded_positions" in stats:
        arrays["decoded_positions"] = stats["decoded_positions"]
    np.savez(analysis_dir / "nav_eval_arrays.npz", **arrays)
    plot_grid_summary(
        stats,
        analysis_dir,
        max_units_to_plot=cfg["analysis"]["max_units_to_plot"],
    )
    if "decoded_positions" in stats:
        plot_path_integration_scatter(
            stats["positions"],
            stats["decoded_positions"],
            analysis_dir / "path_integration_scatter.png",
        )

    topology = {}
    topology_arrays_path = None
    topology_eval_metrics = None
    topology_drift_metrics = None
    topology_error_curve_path = None
    if run_topology:
        raw_torus_cfg = dict(cfg.get("analysis", {}).get("torus", {}))
        raw_torus_cfg.setdefault("sample_dt", float(cfg["data"].get("dt", 0.02)))
        torus_cfg = _torus_defaults(raw_torus_cfg)
        topology_stats = stats
        if topology_loader is not None:
            logger.info(
                "Running dedicated long-trajectory topology evaluation "
                "(%d trajectories × %d steps).",
                int(torus_cfg["eval_num_trajectories"]),
                int(torus_cfg["eval_seq_len"]),
            )
            topology_stats = _full_navigation_evaluation(
                cfg,
                model,
                topology_loader,
                topology_place_encoder or place_encoder,
                device,
                grid_filter=torus_cfg,
            )
            topology_arrays = {
                "positions": topology_stats["positions"],
                "hidden": topology_stats["hidden"],
                "headings": topology_stats["headings"],
                "trajectory_ids": topology_stats["trajectory_ids"],
                "time_indices": topology_stats["time_indices"],
                "step_displacements": topology_stats["step_displacements"],
            }
            if "decoded_positions" in topology_stats:
                topology_arrays["decoded_positions"] = topology_stats[
                    "decoded_positions"
                ]
            topology_arrays_path = analysis_dir / "topology_eval_arrays.npz"
            np.savez_compressed(topology_arrays_path, **topology_arrays)

        topology_eval_metrics = _scalar_navigation_summary(topology_stats)
        if "decoded_positions" in topology_stats:
            topology_error_curve_path = (
                analysis_dir / "topology_path_integration_error_over_time.png"
            )
            plot_path_integration_error_over_time(
                topology_stats["positions"],
                topology_stats["decoded_positions"],
                topology_stats["time_indices"],
                topology_error_curve_path,
                sample_dt=float(torus_cfg["sample_dt"]),
            )
            topology_drift_metrics = summarize_path_integration_drift(
                topology_stats["positions"],
                topology_stats["decoded_positions"],
                topology_stats["time_indices"],
                sample_dt=float(torus_cfg["sample_dt"]),
            )

        topology = run_topology_analysis(
            topology_stats["hidden"],
            topology_stats["positions"],
            analysis_dir,
            logger,
            seed=cfg["seed"],
            headings=topology_stats["headings"],
            grid_results=topology_stats["grid_results"],
            trajectory_ids=topology_stats["trajectory_ids"],
            time_indices=topology_stats["time_indices"],
            step_displacements=topology_stats["step_displacements"],
            torus_config=torus_cfg,
        )

    summary = {
        "checkpoint": str(checkpoint_path.relative_to(output_dir)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "selection_metadata": {
            key: value
            for key, value in checkpoint.items()
            if key not in {"model_state", "config"}
        },
        "metrics": _scalar_navigation_summary(stats),
        "topology_evaluation_metrics": topology_eval_metrics,
        "topology_long_rollout_drift": topology_drift_metrics,
        "topology": topology,
        "artifacts": {
            "arrays": str((analysis_dir / "nav_eval_arrays.npz").relative_to(output_dir)),
            "grid_score_hist": str((analysis_dir / "grid_score_hist.png").relative_to(output_dir)),
            "top_rate_maps": str((analysis_dir / "top_rate_maps.png").relative_to(output_dir)),
            "path_integration_scatter": (
                str((analysis_dir / "path_integration_scatter.png").relative_to(output_dir))
                if "decoded_positions" in stats else None
            ),
            "topology_arrays": (
                str(topology_arrays_path.relative_to(output_dir))
                if topology_arrays_path is not None else None
            ),
            "topology_path_integration_error_over_time": (
                str(topology_error_curve_path.relative_to(output_dir))
                if topology_error_curve_path is not None else None
            ),
        },
    }
    write_json(summary, analysis_dir / "summary.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    set_seed(
        cfg["seed"],
        deterministic_algorithms=bool(
            cfg.get("runtime", {}).get("deterministic_algorithms", False)
        ),
    )
    device = configure_runtime(cfg)
    output_dir = prepare_output_dir(cfg["output_dir"])
    logger = get_logger("exp1_nav_pretrain", output_dir / "train.log")
    save_config_snapshot(cfg, output_dir)
    save_run_provenance(output_dir, "exp1_nav_pretrain", cfg=cfg)
    logger.info("Using device: %s | cpu_threads=%d", device, torch.get_num_threads())

    train_loader, val_loader, place_encoder = make_loaders(cfg)
    model = GridNavigationModel(
        velocity_dim=cfg["model"]["velocity_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        place_dim=cfg["data"]["place_cells"],
        activation=cfg["model"]["activation"],
        nonnegative=cfg["model"]["nonnegative"],
        cell_type=cfg["model"].get("cell_type", "rnn"),
        dropout=cfg["model"].get("dropout", 0.0),
        use_bias=cfg["model"].get("use_bias", False),
        init_relu=cfg["model"].get("init_relu", False),
    ).to(device)
    logger.info(
        "Model: cell=%s hidden=%d bias=%s nonneg=%s",
        cfg["model"].get("cell_type", "rnn"),
        cfg["model"]["hidden_dim"],
        cfg["model"].get("use_bias", False),
        cfg["model"]["nonnegative"],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    scheduler = None
    if cfg["train"].get("use_cosine_schedule", False):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg["train"]["epochs"],
        )

    best_val_loss = float("inf")
    best_balanced_frac = -1.0
    best_grid_frac = -1.0
    history = []
    whh_wd = cfg["train"].get("whh_weight_decay", 1e-4)
    analysis_every = max(1, int(cfg["analysis"].get("every_n_epochs", 1)))

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        train_stats = train_navigation(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=cfg["train"]["grad_clip"],
            loss_type=cfg["train"].get("loss_type", "ce"),
            whh_weight_decay=whh_wd,
            hidden_decorrelation_weight=cfg["train"].get(
                "hidden_decorrelation_weight", 0.0
            ),
        )
        if scheduler is not None:
            scheduler.step()

        validation = evaluate_navigation_loss(model, val_loader, device)
        entry = {
            "epoch": epoch,
            "train_total_loss": train_stats["total_loss"],
            "train_task_loss": train_stats["task_loss"],
            "train_recurrent_regularizer": train_stats["recurrent_regularizer"],
            "train_decorrelation_regularizer": train_stats["decorrelation_regularizer"],
            "train_gradient_norm": train_stats["gradient_norm"],
            "validation_task_loss": validation["loss"],
            "validation_loss_excess_over_floor": validation["loss_excess_over_floor"],
            "validation_frac_loss_captured": validation["frac_loss_captured"],
            "full_analysis_performed": False,
        }

        if validation["loss"] < best_val_loss:
            best_val_loss = float(validation["loss"])
            save_checkpoint({
                "model_state": model.state_dict(),
                "config": cfg,
                "epoch": epoch,
                "best_validation_task_loss": best_val_loss,
            }, output_dir / "best.pt")

        perform_analysis = epoch % analysis_every == 0 or epoch == cfg["train"]["epochs"]
        if perform_analysis:
            full = _full_navigation_evaluation(
                cfg, model, val_loader, place_encoder, device
            )
            pi_err = float(full.get("path_integration_error_relative", -1.0))
            frac_03 = float(full.get("frac_grid_above_03", 0.0))
            entry.update({
                "full_analysis_performed": True,
                "mean_grid_score": full.get("mean_grid_score"),
                "max_grid_score": full.get("max_grid_score"),
                "frac_grid_above_03": frac_03,
                "frac_grid_above_03_conservative": full.get(
                    "frac_grid_above_03_conservative"
                ),
                "frac_grid_above_03_google": full.get("frac_grid_above_03_google"),
                "path_integration_error_relative": pi_err,
                "hidden_participation_ratio": full.get("hidden_participation_ratio"),
                "hidden_offdiag_corr_mse": full.get("hidden_offdiag_corr_mse"),
            })
            if frac_03 > best_grid_frac:
                best_grid_frac = frac_03
                save_checkpoint({
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "frac_grid_above_03": frac_03,
                    "path_integration_error_relative": pi_err,
                }, output_dir / "best_grid.pt")

            pi_gate = cfg["analysis"].get("checkpoint_max_pi_err", 0.08)
            if pi_err >= 0 and pi_err <= pi_gate and frac_03 > best_balanced_frac:
                best_balanced_frac = frac_03
                save_checkpoint({
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "epoch": epoch,
                    "frac_grid_above_03": frac_03,
                    "path_integration_error_relative": pi_err,
                }, output_dir / "best_grid_balanced.pt")

            logger.info(
                "epoch=%d train_task=%.4f train_total=%.4f val_task=%.4f "
                "grid=%.4f frac>0.3=%.1f%% PI_err=%.4f",
                epoch,
                train_stats["task_loss"],
                train_stats["total_loss"],
                validation["loss"],
                full["mean_grid_score"],
                frac_03 * 100,
                pi_err,
            )
        else:
            logger.info(
                "epoch=%d train_task=%.4f train_total=%.4f val_task=%.4f "
                "(grid analysis deferred; cadence=%d)",
                epoch,
                train_stats["task_loss"],
                train_stats["total_loss"],
                validation["loss"],
                analysis_every,
            )
        history.append(entry)

    write_json(history, output_dir / "history.json")

    requested_name = cfg["analysis"].get(
        "post_training_checkpoint", "best_grid_balanced.pt"
    )
    candidate_names = [requested_name, "best_grid.pt", "best.pt"]
    selected_path = next(
        (output_dir / name for name in candidate_names if (output_dir / name).exists()),
        None,
    )
    if selected_path is None:
        raise RuntimeError("Navigation training produced no checkpoint")
    logger.info("Post-training topology checkpoint: %s", selected_path.name)

    topology_loader, topology_place_encoder = make_topology_loader(cfg)

    checkpoint_paths = {
        "best_validation": output_dir / "best.pt",
        "best_grid": output_dir / "best_grid.pt",
        "best_grid_balanced": output_dir / "best_grid_balanced.pt",
    }
    checkpoint_summaries = {}
    for tag, checkpoint_path in checkpoint_paths.items():
        if not checkpoint_path.exists():
            continue
        checkpoint_summaries[tag] = _analyse_navigation_checkpoint(
            tag=tag,
            checkpoint_path=checkpoint_path,
            cfg=cfg,
            model=model,
            val_loader=val_loader,
            place_encoder=place_encoder,
            topology_loader=topology_loader,
            topology_place_encoder=topology_place_encoder,
            device=device,
            output_dir=output_dir,
            logger=logger,
            run_topology=checkpoint_path.resolve() == selected_path.resolve(),
        )

    selected_tag = next(
        tag
        for tag, checkpoint_path in checkpoint_paths.items()
        if checkpoint_path.exists() and checkpoint_path.resolve() == selected_path.resolve()
    )
    summary = {
        "best_validation_task_loss": best_val_loss,
        "analysis_every_n_epochs": analysis_every,
        "post_training_checkpoint": str(selected_path.relative_to(output_dir)),
        "post_training_checkpoint_tag": selected_tag,
        "checkpoint_analyses": checkpoint_summaries,
        "interpretation_note": (
            "Every metric and figure is stored under checkpoint_analysis/<tag> and "
            "bound to the checkpoint SHA-256 recorded there."
        ),
    }
    write_json(summary, output_dir / "summary.json")
    write_json({
        "experiment": "exp1_nav_pretrain",
        "history": "history.json",
        "summary": "summary.json",
        "selection_policy": {
            "best.pt": "minimum lightweight validation task loss",
            "best_grid.pt": "maximum grid fraction at configured analysis cadence",
            "best_grid_balanced.pt": "maximum grid fraction subject to PI-error gate",
        },
        "checkpoints": [
            {
                "path": str(path.relative_to(output_dir)),
                "sha256": sha256_file(path),
            }
            for path in checkpoint_paths.values()
            if path.exists()
        ],
    }, output_dir / "artifact_manifest.json")

    logger.info("Finished training. Best validation task loss: %.4f", best_val_loss)
    selected_metrics = checkpoint_summaries[selected_tag]["metrics"]
    logger.info("=== Success checklist for %s ===", selected_tag)
    logger.info(
        "  [%s] >= 10%% units grid score > 0.3  (actual: %.1f%%)",
        "OK" if selected_metrics.get("frac_grid_above_03", 0) >= 0.10 else "XX",
        selected_metrics.get("frac_grid_above_03", 0) * 100,
    )
    logger.info(
        "  [%s] PI error < 15%% arena  (actual: %.1f%%)",
        "OK"
        if selected_metrics.get("path_integration_error_relative", 1) < 0.15
        else "XX",
        selected_metrics.get("path_integration_error_relative", -1) * 100,
    )


if __name__ == "__main__":
    main()
