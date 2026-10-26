"""Automatic Schøyen-style torus discovery from a trained navigation checkpoint.

This analysis replaces the legacy grid-score-selected topology experiment.
It starts from the VALIDATION checkpoint ``best.pt`` and performs, end to end:

1. Fresh true-position rollouts for rotational cell-type clustering.
2. 32x32 recurrent-unit ratemaps, silent/bias filtering, rotational
   autocorrelation, UMAP2 + DBSCAN.
3. Automatic morphology ranking of band-like clusters.
4. A small, pre-defined candidate search over the strongest band clusters.
5. Independent 64x64 true-position ratemaps for topology.
6. Spatial-bin population vectors -> PCA6 -> Gardner/Schøyen fuzzy
   downsampling -> Ripser.
7. Persistent-homology evaluation with BOTH Euclidean and cosine metrics.
8. Automatic selection of the cluster ensemble with the strongest robust
   (1,2,1) signature across both metrics.

The goal is representation discovery: find a stable two-dimensional toroidal
population manifold in the RNN hidden layer. The analysis does not make a
causal claim about band cells versus grid cells.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.signal
import torch
import umap
from ripser import ripser
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from umap.umap_ import compute_membership_strengths, smooth_knn_dist

from gpm_cmap.data.nav_env.open_field import OpenFieldConfig, OpenFieldSampler
from gpm_cmap.data.nav_env.place_cells import PlaceCellEncoder
from gpm_cmap.models.grid_rnn import GridNavigationModel
from gpm_cmap.utils.checkpointing import load_checkpoint, sha256_file
from gpm_cmap.utils.config import load_config, save_config


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _as_int_list(text: str | None) -> List[int]:
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _finite_lifetimes(dgm: np.ndarray) -> np.ndarray:
    q = np.asarray(dgm, dtype=float)
    if q.size == 0:
        return np.empty(0, dtype=float)
    ok = np.isfinite(q[:, 0]) & np.isfinite(q[:, 1])
    life = q[ok, 1] - q[ok, 0]
    life = life[np.isfinite(life) & (life >= 0)]
    return np.sort(life)[::-1]


def _rank_ratio(values: np.ndarray, numerator_rank: int, denominator_rank: int) -> float | None:
    if len(values) < max(numerator_rank, denominator_rank):
        return None
    den = float(values[denominator_rank - 1])
    if den <= 0:
        return None
    return float(values[numerator_rank - 1] / den)


def _geometric_mean_positive(values: Iterable[float]) -> float:
    vals = np.asarray([float(v) for v in values if v is not None and float(v) > 0], dtype=float)
    if len(vals) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(vals))))


def _cluster_key(labels: Sequence[int]) -> str:
    return "_".join(str(int(x)) for x in labels)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {
    "cluster_trajectories": 10000,
    "cluster_seq_len": 20,
    "cluster_start_step": 10,
    "cluster_resolution": 32,
    "tda_trajectories": 5000,
    "tda_seq_len": 20,
    "tda_start_step": 0,
    "tda_resolution": 64,
    "rollout_batch_size": 200,
    "silent_variance": 1e-12,
    "bias_percentile": 99.75,
    "rotation_resolution": 360,
    "umap_neighbors": 20,
    "umap_min_dist": 0.05,
    "dbscan_eps": 0.5,
    "dbscan_min_samples": 30,
    "min_cluster_size": 64,
    "top_band_clusters": 4,
    "candidate_union_depth": 3,
    "fallback_pairwise": True,
    "pca_dim": 6,
    "downsample_neighbors": 1000,
    "downsample_points": 3000,
    "ripser_n_perm": 500,
    "ripser_coeff": 47,
    "h1_gap_threshold": 1.5,
    "h2_gap_threshold": 1.5,
    "random_state": 42,
    "max_missing_fraction_cluster": 0.005,
    "max_missing_fraction_tda": 0.02,
    "save_cluster_montages": True,
    "max_montage_units": 100,
}


def _analysis_cfg(cfg: Mapping[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = dict(DEFAULTS)
    user = ((cfg.get("analysis") or {}).get("schoyen_torus") or {})
    out.update(user)
    if args.smoke:
        out.update(
            cluster_trajectories=400,
            tda_trajectories=300,
            rollout_batch_size=100,
            rotation_resolution=90,
            min_cluster_size=8,
            top_band_clusters=3,
            candidate_union_depth=2,
            downsample_neighbors=100,
            downsample_points=250,
            ripser_n_perm=100,
            umap_neighbors=10,
            dbscan_min_samples=5,
            max_missing_fraction_cluster=0.25,
            max_missing_fraction_tda=0.25,
        )
    for key in [
        "cluster_trajectories", "tda_trajectories", "rollout_batch_size",
        "min_cluster_size", "top_band_clusters", "candidate_union_depth",
        "downsample_points", "ripser_n_perm",
    ]:
        override = getattr(args, key, None)
        if override is not None:
            out[key] = override
    return out


# ---------------------------------------------------------------------------
# Model + fresh rollouts
# ---------------------------------------------------------------------------

def _build_model(cfg: Mapping[str, Any], checkpoint: Path, device: torch.device) -> GridNavigationModel:
    model = GridNavigationModel(
        velocity_dim=int(cfg["model"]["velocity_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        place_dim=int(cfg["data"]["place_cells"]),
        activation=str(cfg["model"]["activation"]),
        nonnegative=bool(cfg["model"]["nonnegative"]),
        cell_type=str(cfg["model"].get("cell_type", "rnn")),
        dropout=float(cfg["model"].get("dropout", 0.0)),
        use_bias=bool(cfg["model"].get("use_bias", False)),
        init_relu=bool(cfg["model"].get("init_relu", False)),
    ).to(device)
    payload = load_checkpoint(checkpoint, map_location=device)
    state = payload.get("model_state") or payload.get("model_state_dict")
    if state is None:
        state = payload
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _build_sampler_and_encoder(cfg: Mapping[str, Any], seed: int) -> Tuple[OpenFieldConfig, PlaceCellEncoder]:
    d = cfg["data"]
    of_cfg = OpenFieldConfig(
        arena_size=float(d["arena_size"]),
        dt=float(d.get("dt", 0.05)),
        speed_scale=float(d.get("speed_scale", 0.13)),
        sigma_theta=float(d.get("sigma_theta", 5.0)),
        min_speed=float(d.get("min_speed", 0.0)),
        max_speed=float(d.get("max_speed", 0.5)),
        border_region=float(d.get("border_region", 0.03)),
        boundary_mode=str(d.get("boundary_mode", "avoid")),
    )
    encoder = PlaceCellEncoder(
        n_cells=int(d["place_cells"]),
        arena_size=float(d["arena_size"]),
        sigma=float(d["place_sigma"]),
        seed=int(cfg.get("seed", seed)),
        use_dog=bool(d.get("use_dog", True)),
        surround_scale=float(d.get("surround_scale", 2.0)),
        surround_scale_mode=str(d.get("surround_scale_mode", "variance")),
        periodic=str(d.get("boundary_mode", "avoid")) == "periodic",
    )
    return of_cfg, encoder


class RateMapAccumulator:
    def __init__(self, resolution: int, hidden_dim: int, arena_size: float, device: torch.device):
        self.resolution = int(resolution)
        self.hidden_dim = int(hidden_dim)
        self.arena_size = float(arena_size)
        self.half = self.arena_size / 2.0
        self.sums = torch.zeros(
            (self.resolution * self.resolution, self.hidden_dim),
            dtype=torch.float32,
            device=device,
        )
        self.counts = torch.zeros(
            self.resolution * self.resolution,
            dtype=torch.float32,
            device=device,
        )

    def add(self, positions: torch.Tensor, hidden: torch.Tensor) -> None:
        pos = positions.reshape(-1, 2)
        h = hidden.reshape(-1, hidden.shape[-1])
        scaled = (pos + self.half) / self.arena_size
        ix = torch.floor(scaled[:, 0] * self.resolution).long().clamp(0, self.resolution - 1)
        iy = torch.floor(scaled[:, 1] * self.resolution).long().clamp(0, self.resolution - 1)
        flat = iy * self.resolution + ix
        self.sums.index_add_(0, flat, h)
        self.counts.index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))

    def finish(self) -> Tuple[np.ndarray, np.ndarray]:
        sums = self.sums.detach().cpu()
        counts = self.counts.detach().cpu()
        means = sums / counts[:, None].clamp_min(1.0)
        means[counts == 0] = float("nan")
        maps = (
            means.reshape(self.resolution, self.resolution, self.hidden_dim)
            .permute(2, 0, 1)
            .numpy()
            .astype(np.float32)
        )
        occupancy = counts.reshape(self.resolution, self.resolution).numpy().astype(np.float32)
        return maps, occupancy


@torch.inference_mode()
def _generate_ratemaps(
    *,
    model: GridNavigationModel,
    place_encoder: PlaceCellEncoder,
    sampler: OpenFieldSampler,
    device: torch.device,
    hidden_dim: int,
    arena_size: float,
    num_trajectories: int,
    seq_len: int,
    start_step: int,
    resolution: int,
    batch_size: int,
    label: str,
) -> Dict[str, Any]:
    acc = RateMapAccumulator(resolution, hidden_dim, arena_size, device)
    err_sum = np.zeros(seq_len, dtype=np.float64)
    err_sq = np.zeros(seq_len, dtype=np.float64)
    err_n = np.zeros(seq_len, dtype=np.int64)
    done = 0
    t0 = time.time()
    while done < num_trajectories:
        cur = min(batch_size, num_trajectories - done)
        batch = sampler.sample_batch(batch_size=cur, seq_len=seq_len)
        positions = torch.from_numpy(batch["positions"]).to(device=device, dtype=torch.float32)
        velocities = torch.from_numpy(batch["velocities"]).to(device=device, dtype=torch.float32)
        init_pos = torch.from_numpy(batch["init_pos"]).to(device=device, dtype=torch.float32)
        init_place = place_encoder.encode(init_pos).to(device)
        outputs = model(velocities, init_place=init_place)
        hidden = outputs["hidden_seq"]
        acc.add(positions[:, start_step:, :], hidden[:, start_step:, :])
        decoded = place_encoder.decode(outputs["place_pred"])
        err = torch.linalg.vector_norm(decoded - positions, dim=-1).detach().cpu().numpy()
        err_sum += err.sum(axis=0)
        err_sq += (err * err).sum(axis=0)
        err_n += err.shape[0]
        done += cur
        print(
            f"[{label}] {done}/{num_trajectories} "
            f"({100.0 * done / num_trajectories:.1f}%) "
            f"elapsed={(time.time() - t0) / 60:.1f} min",
            flush=True,
        )
    ratemaps, occupancy = acc.finish()
    mean_err = err_sum / np.maximum(err_n, 1)
    var_err = err_sq / np.maximum(err_n, 1) - mean_err**2
    sd_err = np.sqrt(np.maximum(var_err, 0.0))
    coverage = float(np.mean(occupancy > 0))
    return {
        "ratemaps": ratemaps,
        "occupancy": occupancy,
        "coverage_fraction": coverage,
        "n_unvisited_bins": int(np.sum(occupancy == 0)),
        "mean_error_by_step": mean_err.astype(np.float32),
        "sd_error_by_step": sd_err.astype(np.float32),
        "mean_error_all": float(np.average(mean_err, weights=err_n)),
        "mean_error_final": float(mean_err[-1]),
    }


def _save_occupancy(occupancy: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(occupancy, origin="lower")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="visits")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Rotational clustering + morphology ranking
# ---------------------------------------------------------------------------

def _disk_mask(size: int) -> np.ndarray:
    y, x = np.ogrid[-size / 2 : size / 2, -size / 2 : size / 2]
    return x**2 + y**2 <= ((size - 1) / 2) ** 2


def _fill_residual_missing(maps: np.ndarray) -> np.ndarray:
    x = np.asarray(maps, dtype=np.float32).copy()
    means = np.nanmean(x, axis=(1, 2))
    for i in range(len(x)):
        bad = ~np.isfinite(x[i])
        if np.any(bad):
            x[i, bad] = means[i]
    return x


def _rotational_signature(rate_map: np.ndarray, n_angles: int) -> np.ndarray:
    m = np.asarray(rate_map, dtype=np.float64)
    autocorr = scipy.signal.correlate(m, m, mode="same")
    mask = _disk_mask(autocorr.shape[0])
    ref = autocorr[mask]
    if np.std(ref) <= 1e-15:
        return np.zeros(n_angles, dtype=np.float32)
    angles = np.linspace(-np.pi, np.pi, n_angles)
    out = np.empty(n_angles, dtype=np.float32)
    for j, angle in enumerate(angles):
        rot = scipy.ndimage.rotate(autocorr, np.degrees(angle), reshape=False, order=0)
        v = rot[mask]
        out[j] = 0.0 if np.std(v) <= 1e-15 else np.corrcoef(ref, v)[0, 1]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _band_anisotropy(rate_map: np.ndarray) -> Tuple[float, float]:
    m = np.asarray(rate_map, dtype=float)
    m = np.nan_to_num(m, nan=float(np.nanmean(m)))
    gy, gx = np.gradient(m)
    jxx = float(np.sum(gx * gx))
    jyy = float(np.sum(gy * gy))
    jxy = float(np.sum(gx * gy))
    tr = jxx + jyy
    if tr <= 1e-15:
        return 0.0, float("nan")
    delta = math.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy)
    anis = delta / tr
    grad = 0.5 * math.atan2(2.0 * jxy, jxx - jyy)
    band_angle = (math.degrees(grad) + 90.0) % 180.0
    return float(anis), float(band_angle)


def _axial_mean_deg(angles_deg: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    a = np.asarray(angles_deg, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(a) & np.isfinite(w)
    if not np.any(ok):
        return float("nan"), 0.0
    aa = np.deg2rad(2.0 * a[ok])
    ww = np.maximum(w[ok], 0.0)
    if np.sum(ww) <= 0:
        ww = np.ones_like(ww)
    z = np.sum(ww * np.exp(1j * aa)) / np.sum(ww)
    return float((np.degrees(np.angle(z)) / 2.0) % 180.0), float(abs(z))


def _save_montage(
    maps: np.ndarray,
    unit_indices: np.ndarray,
    path: Path,
    title: str,
    max_maps: int,
    seed: int,
) -> None:
    inds = np.asarray(unit_indices, dtype=int)
    if len(inds) == 0:
        return
    rng = np.random.default_rng(seed)
    if len(inds) > max_maps:
        inds = np.sort(rng.choice(inds, size=max_maps, replace=False))
    cols = 10
    rows = int(math.ceil(len(inds) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(1.3 * cols, 1.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, u in zip(axes, inds):
        ax.imshow(maps[u], origin="lower")
        ax.set_title(str(int(u)), fontsize=6)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@dataclass
class ClusterResult:
    accepted_indices: np.ndarray
    labels: np.ndarray
    embedding: np.ndarray
    signatures: np.ndarray
    rows: List[Dict[str, Any]]
    maps_filled: np.ndarray


def _cluster_units(
    ratemaps: np.ndarray,
    occupancy: np.ndarray,
    params: Mapping[str, Any],
    out: Path,
) -> ClusterResult:
    coverage = float(np.mean(occupancy > 0))
    missing = 1.0 - coverage
    if missing > float(params["max_missing_fraction_cluster"]):
        raise RuntimeError(
            f"Clustering occupancy missing {100*missing:.2f}% > allowed "
            f"{100*float(params['max_missing_fraction_cluster']):.2f}%"
        )
    maps_filled = _fill_residual_missing(ratemaps)
    spatial_var = np.var(maps_filled, axis=(1, 2))
    spatial_mean = np.mean(maps_filled, axis=(1, 2))
    silent = spatial_var <= float(params["silent_variance"])
    active_mean = spatial_mean[~silent]
    if len(active_mean) == 0:
        raise RuntimeError("All recurrent units are silent")
    cutoff = float(np.percentile(active_mean, float(params["bias_percentile"])))
    bias = spatial_mean > cutoff
    accepted_mask = ~(silent | bias)
    accepted_indices = np.flatnonzero(accepted_mask)
    accepted = maps_filled[accepted_indices]
    mu = accepted.mean(axis=(1, 2), keepdims=True)
    sd = accepted.std(axis=(1, 2), keepdims=True)
    accepted_z = (accepted - mu) / np.maximum(sd, 1e-12)

    n_angles = int(params["rotation_resolution"])
    signatures = np.empty((len(accepted_indices), n_angles), dtype=np.float32)
    for i, m in enumerate(accepted_z):
        signatures[i] = _rotational_signature(m, n_angles)
        if (i + 1) % 100 == 0 or i + 1 == len(accepted_z):
            print(f"[cluster] rotational signatures {i+1}/{len(accepted_z)}", flush=True)

    nn = min(int(params["umap_neighbors"]), max(2, len(signatures) - 1))
    embedding = umap.UMAP(
        metric="euclidean",
        n_neighbors=nn,
        min_dist=float(params["umap_min_dist"]),
        init="spectral",
        n_components=2,
        random_state=int(params["random_state"]),
    ).fit_transform(signatures)
    labels = DBSCAN(
        min_samples=int(params["dbscan_min_samples"]),
        eps=float(params["dbscan_eps"]),
    ).fit_predict(embedding)

    anis = np.empty(len(accepted_indices), dtype=float)
    orient = np.empty(len(accepted_indices), dtype=float)
    for i, m in enumerate(accepted_z):
        anis[i], orient[i] = _band_anisotropy(m)

    rows: List[Dict[str, Any]] = []
    cluster_dir = out / "cluster_montages"
    if bool(params["save_cluster_montages"]):
        cluster_dir.mkdir(parents=True, exist_ok=True)
    max_size = max([int(np.sum(labels == lab)) for lab in np.unique(labels) if lab != -1] or [1])
    for lab in np.unique(labels):
        pos = np.flatnonzero(labels == lab)
        units = accepted_indices[pos]
        ori_mean, ori_conc = _axial_mean_deg(orient[pos], anis[pos])
        med_anis = float(np.median(anis[pos]))
        mean_anis = float(np.mean(anis[pos]))
        size = int(len(units))
        size_factor = math.sqrt(size / max_size) if lab != -1 else 0.0
        band_score = med_anis * (0.85 + 0.15 * size_factor)
        row = {
            "cluster": int(lab),
            "size": size,
            "mean_activity": float(np.mean(spatial_mean[units])),
            "median_activity": float(np.median(spatial_mean[units])),
            "median_band_anisotropy": med_anis,
            "mean_band_anisotropy": mean_anis,
            "band_orientation_deg_diagnostic": ori_mean,
            "orientation_concentration_diagnostic": ori_conc,
            "band_morphology_score": float(band_score),
            "eligible_band_candidate": bool(
                lab != -1 and size >= int(params["min_cluster_size"])
            ),
        }
        rows.append(row)
        if lab != -1 and bool(params["save_cluster_montages"]):
            _save_montage(
                maps_filled,
                units,
                cluster_dir / f"cluster_{int(lab)}_n{size}.png",
                f"cluster {int(lab)} | n={size} | band score={band_score:.3f}",
                int(params["max_montage_units"]),
                int(params["random_state"]),
            )

    rows.sort(key=lambda z: z["cluster"])
    with (out / "cluster_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    np.savez_compressed(
        out / "rotational_clustering.npz",
        accepted_indices=accepted_indices,
        accepted_mask=accepted_mask,
        rotational_signatures=signatures,
        umap2=embedding,
        labels=labels,
        silent_mask=silent,
        bias_mask=bias,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    for lab in np.unique(labels):
        s = labels == lab
        ax.scatter(
            embedding[s, 0], embedding[s, 1],
            s=8 if lab != -1 else 3,
            alpha=0.75 if lab != -1 else 0.2,
            label=str(int(lab)),
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Rotational-autocorrelation UMAP + DBSCAN")
    ax.legend(title="cluster", fontsize=7, ncol=2, markerscale=2)
    fig.tight_layout()
    fig.savefig(out / "rotational_umap_clusters.png", dpi=220)
    plt.close(fig)

    eligible = [r for r in rows if r["eligible_band_candidate"]]
    if eligible:
        fig, ax = plt.subplots(figsize=(9, 5))
        for r in eligible:
            ax.scatter(
                r["band_orientation_deg_diagnostic"],
                r["median_band_anisotropy"],
                s=max(25, min(500, 1.4 * r["size"])),
                alpha=0.7,
            )
            ax.text(
                r["band_orientation_deg_diagnostic"],
                r["median_band_anisotropy"],
                str(r["cluster"]), fontsize=8,
            )
        ax.set_xlim(0, 180)
        ax.set_xlabel("Band orientation diagnostic (deg, modulo 180)")
        ax.set_ylabel("Median band anisotropy")
        ax.set_title("Automatic band-cluster morphology diagnostics")
        fig.tight_layout()
        fig.savefig(out / "cluster_band_diagnostics.png", dpi=220)
        plt.close(fig)

    _write_json(
        out / "cluster_metadata.json",
        {
            "position_source": "true",
            "coverage_fraction": coverage,
            "n_units_total": int(len(ratemaps)),
            "n_silent": int(silent.sum()),
            "bias_percentile": float(params["bias_percentile"]),
            "bias_cutoff": cutoff,
            "n_bias": int(bias.sum()),
            "n_accepted": int(len(accepted_indices)),
            "n_clusters_excluding_noise": int(np.sum(np.unique(labels) != -1)),
            "min_cluster_size_for_auto_selection": int(params["min_cluster_size"]),
            "band_score_definition": "median_anisotropy * (0.85 + 0.15*sqrt(cluster_size/max_cluster_size))",
        },
    )
    return ClusterResult(accepted_indices, labels, embedding, signatures, rows, maps_filled)


# ---------------------------------------------------------------------------
# Candidate generation and PH
# ---------------------------------------------------------------------------

def _candidate_cluster_sets(rows: Sequence[Mapping[str, Any]], params: Mapping[str, Any]) -> Tuple[List[List[int]], List[int]]:
    eligible = [r for r in rows if bool(r.get("eligible_band_candidate"))]
    eligible.sort(key=lambda r: (float(r["band_morphology_score"]), int(r["size"])), reverse=True)
    ranked = [int(r["cluster"]) for r in eligible[: int(params["top_band_clusters"])]]
    if not ranked:
        raise RuntimeError("No eligible band-like DBSCAN clusters were found")

    candidates: List[List[int]] = []
    # The two strongest single clusters.
    for lab in ranked[:2]:
        candidates.append([lab])
    # Cumulative unions, which captures the empirically useful strict->broad band continuum.
    depth = min(int(params["candidate_union_depth"]), len(ranked))
    for n in range(2, depth + 1):
        candidates.append(sorted(ranked[:n]))
    # Preserve order and uniqueness.
    unique: List[List[int]] = []
    seen = set()
    for c in candidates:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(list(key))
    return unique, ranked


def _fuzzy_downsample(data: np.ndarray, k: int, num_sample: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=np.float64)
    n = len(x)
    k = min(int(k), n)
    num_sample = min(int(num_sample), n)
    dmat = squareform(pdist(x, metric="cosine")).astype(np.float32)
    dmat = np.nan_to_num(dmat, nan=1.0, posinf=1.0, neginf=1.0)
    knn_indices = np.argsort(dmat, axis=1)[:, :k]
    knn_dists = dmat[np.arange(n)[:, None], knn_indices].copy().astype(np.float32)
    sigmas, rhos = smooth_knn_dist(knn_dists, float(k), local_connectivity=0.0)
    membership = compute_membership_strengths(knn_indices, knn_dists, sigmas, rhos)
    rows, cols, vals = membership[:3]
    result = coo_matrix((vals, (rows, cols)), shape=(n, n))
    result.eliminate_zeros()
    transpose = result.transpose()
    result = result + transpose - result.multiply(transpose)
    result.eliminate_zeros()
    fuzzy = result.toarray().astype(np.float32)
    F = np.sum(fuzzy, axis=1)
    available = np.ones(n, dtype=bool)
    all_inds = np.arange(n)
    selected = np.empty(num_sample, dtype=np.int64)
    trace = np.empty(num_sample, dtype=np.float64)
    i = int(np.argmax(F))
    selected[0] = i
    trace[0] = float(F[i])
    available[i] = False
    for j in range(1, num_sample):
        F += fuzzy[i]
        remaining = all_inds[available]
        local = int(np.argmax(F[available]))
        i = int(remaining[local])
        selected[j] = i
        trace[j] = float(F[i])
        available[i] = False
    return selected, trace


def _ph_summary(pointcloud: np.ndarray, metric: str, params: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[np.ndarray]]:
    result = ripser(
        pointcloud,
        n_perm=min(int(params["ripser_n_perm"]), len(pointcloud)),
        maxdim=2,
        coeff=int(params["ripser_coeff"]),
        metric=metric,
    )
    dgms = [np.asarray(x) for x in result["dgms"]]
    h1 = _finite_lifetimes(dgms[1])
    h2 = _finite_lifetimes(dgms[2])
    h1_gap = _rank_ratio(h1, 2, 3)
    h2_gap = _rank_ratio(h2, 1, 2)
    h1_balance = None
    if len(h1) >= 2 and h1[0] > 0:
        h1_balance = float(h1[1] / h1[0])
    summary = {
        "metric": metric,
        "H0_infinite_components": int(np.sum(~np.isfinite(dgms[0][:, 1]))) if len(dgms[0]) else 0,
        "H1_top10_lifetimes": [float(x) for x in h1[:10]],
        "H2_top10_lifetimes": [float(x) for x in h2[:10]],
        "H1_second_to_third": h1_gap,
        "H2_first_to_second": h2_gap,
        "H1_second_to_first": h1_balance,
        "n_perm": int(min(int(params["ripser_n_perm"]), len(pointcloud))),
        "coeff": int(params["ripser_coeff"]),
    }
    return summary, dgms


def _candidate_units(cluster: ClusterResult, labels: Sequence[int]) -> np.ndarray:
    return cluster.accepted_indices[np.isin(cluster.labels, np.asarray(labels, dtype=int))]


def _candidate_eval(
    *,
    labels: Sequence[int],
    cluster: ClusterResult,
    tda_maps: np.ndarray,
    tda_occupancy: np.ndarray,
    params: Mapping[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    units = _candidate_units(cluster, labels)
    if len(units) < int(params["pca_dim"]):
        raise RuntimeError(f"Candidate {labels} has only {len(units)} units")
    valid_bins = tda_occupancy.reshape(-1) > 0
    flat = tda_maps[units].reshape(len(units), -1).T
    finite = np.all(np.isfinite(flat), axis=1)
    valid = valid_bins & finite
    X = flat[valid]
    pca = PCA(n_components=int(params["pca_dim"]))
    X6 = pca.fit_transform(X)
    inds, trace = _fuzzy_downsample(
        X6,
        int(params["downsample_neighbors"]),
        int(params["downsample_points"]),
    )
    Xds = X6[inds]
    euc, euc_dgms = _ph_summary(Xds, "euclidean", params)
    cos, cos_dgms = _ph_summary(Xds, "cosine", params)

    gaps = [
        euc.get("H1_second_to_third"), euc.get("H2_first_to_second"),
        cos.get("H1_second_to_third"), cos.get("H2_first_to_second"),
    ]
    finite_gaps = [float(x) for x in gaps if x is not None and np.isfinite(x)]
    robust_score = min(finite_gaps) if len(finite_gaps) == 4 else 0.0
    geometric_score = _geometric_mean_positive(finite_gaps)
    h1_thr = float(params["h1_gap_threshold"])
    h2_thr = float(params["h2_gap_threshold"])
    passes = bool(
        euc["H0_infinite_components"] == 1
        and cos["H0_infinite_components"] == 1
        and (euc.get("H1_second_to_third") or 0.0) >= h1_thr
        and (cos.get("H1_second_to_third") or 0.0) >= h1_thr
        and (euc.get("H2_first_to_second") or 0.0) >= h2_thr
        and (cos.get("H2_first_to_second") or 0.0) >= h2_thr
    )

    np.save(out_dir / "selected_unit_indices.npy", units)
    np.save(out_dir / "pca6.npy", X6)
    np.save(out_dir / "downsample_indices.npy", inds)
    np.save(out_dir / "pca6_downsampled.npy", Xds)
    np.save(out_dir / "downsample_score_trace.npy", trace)
    np.savez_compressed(
        out_dir / "diagrams.npz",
        euclidean_H0=euc_dgms[0], euclidean_H1=euc_dgms[1], euclidean_H2=euc_dgms[2],
        cosine_H0=cos_dgms[0], cosine_H1=cos_dgms[1], cosine_H2=cos_dgms[2],
    )
    summary = {
        "clusters": [int(x) for x in labels],
        "n_units": int(len(units)),
        "spatial_population_shape": [int(X.shape[0]), int(X.shape[1])],
        "pca_explained_variance_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "euclidean": euc,
        "cosine": cos,
        "robust_min_gap_score": float(robust_score),
        "geometric_mean_gap_score": float(geometric_score),
        "passes_robust_121_signature": passes,
        "thresholds": {"H1_2_to_3": h1_thr, "H2_1_to_2": h2_thr},
    }
    _write_json(out_dir / "candidate_summary.json", summary)
    return summary


def _select_best_candidate(candidate_summaries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not candidate_summaries:
        raise RuntimeError("No topology candidates were evaluated")
    passes = [z for z in candidate_summaries if z.get("passes_robust_121_signature") is True]
    pool = passes if passes else list(candidate_summaries)
    return max(
        pool,
        key=lambda z: (
            float(z.get("robust_min_gap_score") or 0.0),
            float(z.get("geometric_mean_gap_score") or 0.0),
            int(z.get("n_units") or 0),
        ),
    )


def _plot_persistence(dgms: Sequence[np.ndarray], path: Path, title: str) -> None:
    vals: List[float] = []
    for d in dgms:
        q = np.asarray(d, dtype=float)
        if q.size:
            vals.extend(q[np.isfinite(q)].tolist())
    vmax = max(vals) if vals else 1.0
    vmax = max(vmax, 1e-8)
    lim = 1.08 * vmax
    fig, ax = plt.subplots(figsize=(6, 6))
    for dim, d in enumerate(dgms):
        q = np.asarray(d, dtype=float)
        if q.size == 0:
            continue
        finite = np.isfinite(q[:, 1])
        ax.scatter(q[finite, 0], q[finite, 1], s=18, label=f"H{dim}")
        if np.any(~finite):
            ax.scatter(q[~finite, 0], np.full(np.sum(~finite), 1.03*vmax), marker="^", s=28)
    ax.plot([0, lim], [0, lim], "--", linewidth=1)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Birth"); ax.set_ylabel("Death"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def _plot_rank(life: np.ndarray, path: Path, title: str) -> None:
    if len(life) == 0:
        return
    k = min(30, len(life))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(np.arange(1, k+1), life[:k], "o-")
    ax.set_xlabel("Lifetime rank"); ax.set_ylabel("death - birth"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def _finalize_selected(
    *,
    selected: Mapping[str, Any],
    cluster: ClusterResult,
    tda_maps: np.ndarray,
    tda_occupancy: np.ndarray,
    candidate_dir: Path,
    out: Path,
    params: Mapping[str, Any],
) -> Dict[str, Any]:
    labels = [int(x) for x in selected["clusters"]]
    units = _candidate_units(cluster, labels)
    X6 = np.load(candidate_dir / "pca6.npy")
    inds = np.load(candidate_dir / "downsample_indices.npy")
    Xds = np.load(candidate_dir / "pca6_downsampled.npy")
    with np.load(candidate_dir / "diagrams.npz") as a:
        euc_dgms = [a[f"euclidean_H{i}"] for i in range(3)]
        cos_dgms = [a[f"cosine_H{i}"] for i in range(3)]

    # 3-D UMAP is visualization only and is computed only for the final selected ensemble.
    nn = min(4000, max(2, len(X6)-1))
    u3 = umap.UMAP(
        n_components=3, min_dist=0.8, n_neighbors=nn,
        metric="euclidean", init="spectral",
        random_state=int(params["random_state"]),
    ).fit_transform(X6)

    valid_bins = tda_occupancy.reshape(-1) > 0
    flat = tda_maps[units].reshape(len(units), -1).T
    finite_rows = np.all(np.isfinite(flat), axis=1)
    valid = valid_bins & finite_rows
    yy, xx = np.meshgrid(
        np.arange(tda_occupancy.shape[0]), np.arange(tda_occupancy.shape[1]), indexing="ij"
    )
    spatial_bins = np.column_stack([xx.reshape(-1), yy.reshape(-1)])[valid]
    if len(spatial_bins) != len(X6):
        raise RuntimeError("Spatial-bin coordinates no longer match selected PCA population")

    fig = plt.figure(figsize=(12, 4))
    for j, (elev, azim) in enumerate([(20, 35), (20, 125), (75, 35)], 1):
        ax = fig.add_subplot(1, 3, j, projection="3d")
        ax.scatter(u3[:,0], u3[:,1], u3[:,2], c=X6[:,0], s=3, alpha=0.65)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    fig.suptitle(f"Auto-selected torus ensemble clusters={labels} | n={len(units)}")
    fig.tight_layout(); fig.savefig(out / "umap_torus_3d.png", dpi=220); plt.close(fig)

    _plot_persistence(euc_dgms, out / "persistence_euclidean.png", "Persistent homology: Euclidean")
    _plot_persistence(cos_dgms, out / "persistence_cosine.png", "Persistent homology: cosine")
    _plot_rank(_finite_lifetimes(euc_dgms[1]), out / "H1_lifetime_rank_euclidean.png", "H1 lifetime rank: Euclidean")
    _plot_rank(_finite_lifetimes(euc_dgms[2]), out / "H2_lifetime_rank_euclidean.png", "H2 lifetime rank: Euclidean")
    _plot_rank(_finite_lifetimes(cos_dgms[1]), out / "H1_lifetime_rank_cosine.png", "H1 lifetime rank: cosine")
    _plot_rank(_finite_lifetimes(cos_dgms[2]), out / "H2_lifetime_rank_cosine.png", "H2 lifetime rank: cosine")

    _save_montage(
        tda_maps, units, out / "selected_unit_ratemaps.png",
        f"Auto-selected recurrent units | clusters={labels} | n={len(units)}",
        int(params["max_montage_units"]), int(params["random_state"]),
    )

    # Three-family refinement, matching the Schøyen repository's diagnostic idea.
    selected_maps = tda_maps[units]
    corr_maps = np.empty_like(selected_maps, dtype=np.float32)
    for i, m in enumerate(selected_maps):
        filled = np.nan_to_num(m, nan=float(np.nanmean(m)))
        corr_maps[i] = scipy.signal.correlate(filled, filled, mode="same")
    corr_flat = corr_maps.reshape(len(corr_maps), -1).astype(np.float64)
    corr_flat -= corr_flat.mean(axis=1, keepdims=True)
    corr_flat /= np.maximum(corr_flat.std(axis=1, keepdims=True), 1e-12)
    corrcoef = (corr_flat @ corr_flat.T) / corr_flat.shape[1]
    klabels = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(corrcoef)
    kdir = out / "kmeans3_band_families"; kdir.mkdir(exist_ok=True)
    for lab in range(3):
        subgroup = units[klabels == lab]
        _save_montage(
            tda_maps, subgroup, kdir / f"subgroup_{lab}_n{len(subgroup)}.png",
            f"Band-family subgroup {lab} | n={len(subgroup)}",
            int(params["max_montage_units"]), int(params["random_state"]),
        )

    # Backward-compatible top-level arrays for plotting/aggregation.  These no longer
    # contain legacy cohomological phase coordinates.
    np.savez_compressed(
        out / "torus_analysis_arrays.npz",
        selected_unit_indices=units.astype(np.int64),
        selected_cluster_labels=np.asarray(labels, dtype=np.int64),
        pca6_spatial_population=X6.astype(np.float32),
        downsample_indices=inds.astype(np.int64),
        pca6_downsampled=Xds.astype(np.float32),
        umap3_embedding=u3.astype(np.float32),
        spatial_bin_xy=spatial_bins.astype(np.float32),
        euclidean_H0=euc_dgms[0], euclidean_H1=euc_dgms[1], euclidean_H2=euc_dgms[2],
        cosine_H0=cos_dgms[0], cosine_H1=cos_dgms[1], cosine_H2=cos_dgms[2],
        kmeans3_family_labels=klabels.astype(np.int64),
    )
    return {
        "selected_units": units,
        "selected_clusters": labels,
        "umap3_embedding": u3,
        "spatial_bins": spatial_bins,
    }


# ---------------------------------------------------------------------------
# End-to-end driver
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    params = _analysis_cfg(cfg, args)
    out = Path(args.output_dir or cfg.get("output_dir", "outputs/schoyen_torus"))
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config_used.yaml")
    _write_json(out / "analysis_parameters.json", params)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    source_seed = int(args.seed if args.seed is not None else cfg.get("seed", 0))
    device_text = args.device or cfg.get("device", "cuda")
    if str(device_text) == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(str(device_text))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    model = _build_model(cfg, checkpoint, device)
    of_cfg, encoder = _build_sampler_and_encoder(cfg, source_seed)
    hidden_dim = int(cfg["model"]["hidden_dim"])
    arena_size = float(cfg["data"]["arena_size"])

    provenance = {
        "analysis": "schoyen_auto_torus_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config": str(args.config),
        "source_seed": source_seed,
        "device": str(device),
        "torch_version": torch.__version__,
        "position_source": "true",
        "selection_goal": "automatic hidden-layer torus discovery",
    }
    _write_json(out / "run_provenance.json", provenance)

    # Fresh and independent data sets.
    cluster_sampler = OpenFieldSampler(of_cfg, seed=source_seed + 30001)
    cluster_data = _generate_ratemaps(
        model=model, place_encoder=encoder, sampler=cluster_sampler, device=device,
        hidden_dim=hidden_dim, arena_size=arena_size,
        num_trajectories=int(params["cluster_trajectories"]),
        seq_len=int(params["cluster_seq_len"]),
        start_step=int(params["cluster_start_step"]),
        resolution=int(params["cluster_resolution"]),
        batch_size=int(params["rollout_batch_size"]), label="CLUSTER",
    )
    np.savez_compressed(
        out / "cluster_true_ratemaps.npz",
        ratemaps=cluster_data["ratemaps"], occupancy=cluster_data["occupancy"],
        mean_error_by_step=cluster_data["mean_error_by_step"],
        sd_error_by_step=cluster_data["sd_error_by_step"],
    )
    _save_occupancy(cluster_data["occupancy"], out / "cluster_true_occupancy.png", "TRUE-position occupancy: clustering")

    tda_sampler = OpenFieldSampler(of_cfg, seed=source_seed + 40001)
    tda_data = _generate_ratemaps(
        model=model, place_encoder=encoder, sampler=tda_sampler, device=device,
        hidden_dim=hidden_dim, arena_size=arena_size,
        num_trajectories=int(params["tda_trajectories"]),
        seq_len=int(params["tda_seq_len"]),
        start_step=int(params["tda_start_step"]),
        resolution=int(params["tda_resolution"]),
        batch_size=int(params["rollout_batch_size"]), label="TDA",
    )
    np.savez_compressed(
        out / "tda_true_ratemaps.npz",
        ratemaps=tda_data["ratemaps"], occupancy=tda_data["occupancy"],
        mean_error_by_step=tda_data["mean_error_by_step"],
        sd_error_by_step=tda_data["sd_error_by_step"],
    )
    _save_occupancy(tda_data["occupancy"], out / "tda_true_occupancy.png", "TRUE-position occupancy: TDA")

    tda_missing = 1.0 - float(tda_data["coverage_fraction"])
    if tda_missing > float(params["max_missing_fraction_tda"]):
        raise RuntimeError(
            f"TDA occupancy missing {100*tda_missing:.2f}% > allowed "
            f"{100*float(params['max_missing_fraction_tda']):.2f}%"
        )

    cluster = _cluster_units(
        cluster_data["ratemaps"], cluster_data["occupancy"], params, out
    )
    initial_candidates, ranked_clusters = _candidate_cluster_sets(cluster.rows, params)
    print("[auto] ranked band clusters:", ranked_clusters, flush=True)
    print("[auto] initial candidate ensembles:", initial_candidates, flush=True)

    candidate_root = out / "candidates"; candidate_root.mkdir(exist_ok=True)
    evaluated: List[Dict[str, Any]] = []
    evaluated_keys = set()

    def evaluate(candidate: Sequence[int]) -> Dict[str, Any]:
        key = tuple(sorted(int(x) for x in candidate))
        if key in evaluated_keys:
            return next(z for z in evaluated if tuple(z["clusters"]) == key)
        evaluated_keys.add(key)
        cdir = candidate_root / f"clusters_{_cluster_key(key)}"
        print(f"[auto] PH candidate clusters={list(key)}", flush=True)
        z = _candidate_eval(
            labels=key, cluster=cluster,
            tda_maps=tda_data["ratemaps"], tda_occupancy=tda_data["occupancy"],
            params=params, out_dir=cdir,
        )
        evaluated.append(z)
        return z

    for c in initial_candidates:
        evaluate(c)

    # If the morphology-guided strict/broad candidates do not produce a robust
    # signature, expand to pairwise combinations among the top band clusters.
    if not any(z["passes_robust_121_signature"] for z in evaluated) and bool(params["fallback_pairwise"]):
        for i in range(len(ranked_clusters)):
            for j in range(i + 1, len(ranked_clusters)):
                evaluate(sorted([ranked_clusters[i], ranked_clusters[j]]))

    selected = _select_best_candidate(evaluated)
    selected_key = tuple(int(x) for x in selected["clusters"])
    selected_dir = candidate_root / f"clusters_{_cluster_key(selected_key)}"
    final = _finalize_selected(
        selected=selected, cluster=cluster,
        tda_maps=tda_data["ratemaps"], tda_occupancy=tda_data["occupancy"],
        candidate_dir=selected_dir, out=out, params=params,
    )

    # Candidate table for later cohort aggregation.
    rows = []
    for z in evaluated:
        rows.append({
            "clusters": _cluster_key(z["clusters"]),
            "n_units": z["n_units"],
            "passes_robust_121_signature": z["passes_robust_121_signature"],
            "robust_min_gap_score": z["robust_min_gap_score"],
            "geometric_mean_gap_score": z["geometric_mean_gap_score"],
            "euclidean_H1_2_to_3": z["euclidean"]["H1_second_to_third"],
            "euclidean_H2_1_to_2": z["euclidean"]["H2_first_to_second"],
            "cosine_H1_2_to_3": z["cosine"]["H1_second_to_third"],
            "cosine_H2_1_to_2": z["cosine"]["H2_first_to_second"],
        })
    with (out / "candidate_topology.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    eu = selected["euclidean"]; co = selected["cosine"]
    is_torus_like = bool(selected["passes_robust_121_signature"])
    summary = {
        "status": "success",
        "analysis": "schoyen_auto_torus_v1",
        "is_torus_like": is_torus_like,
        "source_seed": source_seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_role": "best_validation",
        "selection": {
            "automatic": True,
            "ranked_band_clusters": ranked_clusters,
            "selected_clusters": list(selected_key),
            "n_selected_units": int(selected["n_units"]),
            "n_candidates_evaluated": len(evaluated),
            "selection_rule": (
                "prefer candidates passing H1_2/H1_3 and H2_1/H2_2 thresholds "
                "under both Euclidean and cosine; maximize the minimum of the four gaps"
            ),
            "robust_min_gap_score": float(selected["robust_min_gap_score"]),
            "geometric_mean_gap_score": float(selected["geometric_mean_gap_score"]),
            "thresholds": selected["thresholds"],
        },
        "data": {
            "position_source": "true",
            "cluster_trajectories": int(params["cluster_trajectories"]),
            "cluster_seq_len": int(params["cluster_seq_len"]),
            "cluster_start_step": int(params["cluster_start_step"]),
            "cluster_resolution": int(params["cluster_resolution"]),
            "cluster_coverage_fraction": float(cluster_data["coverage_fraction"]),
            "tda_trajectories": int(params["tda_trajectories"]),
            "tda_seq_len": int(params["tda_seq_len"]),
            "tda_resolution": int(params["tda_resolution"]),
            "tda_coverage_fraction": float(tda_data["coverage_fraction"]),
            "cluster_mean_pi_error": float(cluster_data["mean_error_all"]),
            "cluster_final_pi_error": float(cluster_data["mean_error_final"]),
            "tda_mean_pi_error": float(tda_data["mean_error_all"]),
            "tda_final_pi_error": float(tda_data["mean_error_final"]),
        },
        "persistent_cohomology": {
            "is_torus_like": is_torus_like,
            "primary_metric": "euclidean",
            "metrics": {"euclidean": eu, "cosine": co},
            # Keep old collector compatibility: Euclidean is the paper-text primary.
            "lifetimes": {
                "1": eu["H1_top10_lifetimes"],
                "2": eu["H2_top10_lifetimes"],
            },
            "observed_lifetimes": {
                "H1": eu["H1_top10_lifetimes"],
                "H2": eu["H2_top10_lifetimes"],
            },
            "n_shuffles": 0,
            "p_values": None,
            "null_max_lifetime": None,
            "signature": [1, 2, 1] if is_torus_like else None,
        },
        "legacy_source_v3_compatible": False,
        "note": (
            "This topology replaces the legacy grid-score-selected topology experiment. "
            "It does not produce the legacy cohomological phase-coordinate contract used "
            "by Source-v3."
        ),
    }
    _write_json(out / "torus_analysis_summary.json", summary)

    # Compact source-level result for scripts that only need the main topology decision.
    _write_json(
        out / "selected_ensemble.json",
        {
            "source_seed": source_seed,
            "selected_clusters": list(selected_key),
            "n_selected_units": int(selected["n_units"]),
            "is_torus_like": is_torus_like,
            "robust_min_gap_score": float(selected["robust_min_gap_score"]),
            "euclidean": eu,
            "cosine": co,
        },
    )
    print("=" * 80)
    print("AUTO TORUS RESULT")
    print("selected clusters:", list(selected_key))
    print("selected units   :", selected["n_units"])
    print("torus-like       :", is_torus_like)
    print("robust min gap   :", selected["robust_min_gap_score"])
    print("Euclidean H1/H2 gaps:", eu["H1_second_to_third"], eu["H2_first_to_second"])
    print("Cosine    H1/H2 gaps:", co["H1_second_to_third"], co["H2_first_to_second"])
    print("output:", out)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    # Optional operational overrides; scientific defaults live in config/DEFAULTS.
    ap.add_argument("--cluster_trajectories", type=int, default=None)
    ap.add_argument("--tda_trajectories", type=int, default=None)
    ap.add_argument("--rollout_batch_size", type=int, default=None)
    ap.add_argument("--min_cluster_size", type=int, default=None)
    ap.add_argument("--top_band_clusters", type=int, default=None)
    ap.add_argument("--candidate_union_depth", type=int, default=None)
    ap.add_argument("--downsample_points", type=int, default=None)
    ap.add_argument("--ripser_n_perm", type=int, default=None)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
