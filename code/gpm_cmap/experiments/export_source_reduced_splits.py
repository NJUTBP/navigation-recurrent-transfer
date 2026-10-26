"""Export independent fit/validation/test navigation trajectories for source v3.

The command preserves the locked navigation checkpoint format and writes into a
new, empty output directory.  It exports the formal grid module and, by default,
a deterministic same-size non-grid control module in separate NPZ files.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from gpm_cmap.analysis.source_reduced_v3 import CLAIM_SCOPE, SCHEMA_VERSION
from gpm_cmap.experiments.export_navigation_calibration import (
    _load_selected_units,
    _make_loader,
    _make_model,
)
from gpm_cmap.utils import (
    configure_runtime,
    compute_source_tree_hash,
    load_config,
    prepare_output_dir,
    save_config_snapshot,
    save_run_provenance,
    set_seed,
    sha256_file,
    write_json,
)
from gpm_cmap.utils.checkpointing import load_checkpoint
from gpm_cmap.utils.logging_utils import get_logger


_LEGACY_RESULT_DIR_NAMES = {
    "shared_manifold_001",
    "shared_manifold_v2_003",
    "manifold_interventions_001",
}


def _trajectory_fingerprints_from_arrays(
    arrays: Dict[str, np.ndarray],
) -> Dict[str, list[int]]:
    coordinates = np.asarray(arrays["positions"], dtype=np.float64)
    trajectory_ids = np.asarray(arrays["trajectory_ids"], dtype=np.int64)
    time_indices = np.asarray(arrays["time_indices"], dtype=np.int64)
    fingerprints: Dict[str, list[int]] = {}
    for trajectory_id in np.unique(trajectory_ids):
        rows = np.flatnonzero(trajectory_ids == trajectory_id)
        rows = rows[np.argsort(time_indices[rows], kind="stable")]
        relative_time = time_indices[rows] - int(time_indices[rows][0])
        digest = hashlib.sha256()
        digest.update(np.asarray([rows.size], dtype=np.int64).tobytes())
        digest.update(np.asarray(relative_time, dtype=np.int64).tobytes())
        digest.update(
            np.round(coordinates[rows], decimals=10)
            .astype(np.float64)
            .tobytes()
        )
        fingerprints.setdefault(digest.hexdigest(), []).append(int(trajectory_id))
    return fingerprints


@torch.no_grad()
def _extract_two_modules(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    grid_units: np.ndarray,
    non_grid_units: Optional[np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    model.eval()
    grid_hidden = []
    non_grid_hidden = []
    positions = []
    headings = []
    trajectory_ids = []
    time_indices = []
    step_displacements = []
    trajectory_offset = 0
    grid_tensor = torch.as_tensor(grid_units, dtype=torch.long, device=device)
    non_grid_tensor = (
        None
        if non_grid_units is None
        else torch.as_tensor(non_grid_units, dtype=torch.long, device=device)
    )

    for batch in loader:
        velocities = batch["velocities"].to(device)
        init_place = batch["init_place"].to(device)
        output = model(velocities, init_place=init_place)
        hidden_seq = output["hidden_seq"]
        grid = torch.index_select(hidden_seq, dim=-1, index=grid_tensor)
        grid_hidden.append(
            grid.reshape(-1, grid.shape[-1]).cpu().numpy().astype(np.float32)
        )
        if non_grid_tensor is not None:
            control = torch.index_select(hidden_seq, dim=-1, index=non_grid_tensor)
            non_grid_hidden.append(
                control.reshape(-1, control.shape[-1])
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        batch_positions = batch["positions"]
        batch_size, seq_len = batch_positions.shape[:2]
        positions.append(batch_positions.reshape(-1, 2).cpu().numpy().astype(np.float32))
        headings.append(batch["headings"].reshape(-1).cpu().numpy().astype(np.float32))
        trajectory_ids.append(
            np.repeat(
                np.arange(
                    trajectory_offset,
                    trajectory_offset + batch_size,
                    dtype=np.int64,
                ),
                seq_len,
            )
        )
        time_indices.append(np.tile(np.arange(seq_len, dtype=np.int64), batch_size))
        step_displacements.append(
            torch.linalg.vector_norm(batch["velocities"], dim=-1)
            .reshape(-1)
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        trajectory_offset += batch_size

    common = {
        "positions": np.concatenate(positions, axis=0),
        "headings": np.concatenate(headings, axis=0),
        "trajectory_ids": np.concatenate(trajectory_ids, axis=0),
        "time_indices": np.concatenate(time_indices, axis=0),
        "step_displacements": np.concatenate(step_displacements, axis=0),
    }
    grid_arrays = {
        **common,
        "hidden": np.concatenate(grid_hidden, axis=0),
        "unit_indices": grid_units.astype(np.int64, copy=False),
        "schema_version": np.asarray(SCHEMA_VERSION),
    }
    non_grid_arrays = None
    if non_grid_units is not None:
        non_grid_arrays = {
            **common,
            "hidden": np.concatenate(non_grid_hidden, axis=0),
            "unit_indices": non_grid_units.astype(np.int64, copy=False),
            "schema_version": np.asarray(SCHEMA_VERSION),
            "control_kind": np.asarray("size_matched_non_grid_module"),
        }
    return grid_arrays, non_grid_arrays


def _select_non_grid_units(
    *,
    hidden_dim: int,
    grid_units: np.ndarray,
    seed: int,
) -> np.ndarray:
    all_units = np.arange(int(hidden_dim), dtype=np.int64)
    available = np.setdiff1d(all_units, grid_units, assume_unique=False)
    if available.size < grid_units.size:
        raise ValueError(
            "The recurrent layer does not contain enough non-grid units for a same-size control"
        )
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(available, size=grid_units.size, replace=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Locked navigation training YAML")
    parser.add_argument("--checkpoint", required=True, help="Locked navigation checkpoint")
    parser.add_argument(
        "--nav_analysis_dir",
        required=True,
        help="Formal analyze_torus output containing selected grid units",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fit_trajectories", type=int, default=128)
    parser.add_argument("--validation_trajectories", type=int, default=64)
    parser.add_argument("--test_trajectories", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--fit_seed_offset", type=int, default=30001)
    parser.add_argument("--validation_seed_offset", type=int, default=31001)
    parser.add_argument("--test_seed_offset", type=int, default=32001)
    parser.add_argument(
        "--no_non_grid_control",
        action="store_true",
        help="Do not export the deterministic same-size non-grid control module",
    )
    parser.add_argument("--non_grid_seed_offset", type=int, default=33001)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    split_specs = {
        "fit": (int(args.fit_trajectories), int(args.fit_seed_offset)),
        "validation": (
            int(args.validation_trajectories),
            int(args.validation_seed_offset),
        ),
        "test": (int(args.test_trajectories), int(args.test_seed_offset)),
    }
    if any(count < 2 for count, _offset in split_specs.values()):
        raise ValueError("Every source split must contain at least two trajectories")
    if int(args.seq_len) < 2:
        raise ValueError("--seq_len must be at least 2")
    offsets = [offset for _count, offset in split_specs.values()]
    if len(set(offsets)) != len(offsets):
        raise ValueError("fit/validation/test trajectory seed offsets must be distinct")

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    cfg["output_dir"] = args.output_dir
    if Path(args.output_dir).name in _LEGACY_RESULT_DIR_NAMES:
        raise ValueError(
            "Refusing to reuse a protected legacy result-directory name: "
            f"{Path(args.output_dir).name}"
        )
    cfg["source_reduced_v3_export"] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve()),
        "nav_analysis_dir": str(Path(args.nav_analysis_dir).expanduser().resolve()),
        "splits": {
            name: {
                "num_trajectories": count,
                "trajectory_seed_offset": offset,
                "seq_len": int(args.seq_len),
            }
            for name, (count, offset) in split_specs.items()
        },
        "export_non_grid_control": not bool(args.no_non_grid_control),
        "non_grid_seed_offset": int(args.non_grid_seed_offset),
    }
    seed = int(cfg["seed"])
    set_seed(
        seed,
        deterministic_algorithms=bool(
            cfg.get("runtime", {}).get("deterministic_algorithms", False)
        ),
    )
    device = configure_runtime(cfg)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    analysis_dir = Path(args.nav_analysis_dir).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    required_analysis_inputs = [
        analysis_dir / "torus_analysis_arrays.npz",
        analysis_dir / "torus_analysis_summary.json",
        analysis_dir / "run_provenance.json",
    ]
    missing_analysis_inputs = [
        str(path) for path in required_analysis_inputs if not path.exists()
    ]
    if missing_analysis_inputs:
        raise FileNotFoundError(
            "Formal navigation analysis is incomplete; missing "
            f"{missing_analysis_inputs}"
        )
    grid_units = _load_selected_units(analysis_dir)
    hidden_dim = int(cfg["model"]["hidden_dim"])
    non_grid_units = None
    if not args.no_non_grid_control:
        non_grid_units = _select_non_grid_units(
            hidden_dim=hidden_dim,
            grid_units=grid_units,
            seed=seed + int(args.non_grid_seed_offset),
        )

    source_tree = compute_source_tree_hash()
    output_dir = prepare_output_dir(args.output_dir)
    logger = get_logger("export_source_reduced_splits", output_dir / "export.log")
    save_config_snapshot(cfg, output_dir)
    input_config_sha256 = sha256_file(Path(args.config))
    effective_config_sha256 = sha256_file(output_dir / "config_used.yaml")

    model = _make_model(cfg, device)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    if "model_state" not in checkpoint:
        raise KeyError(f"{checkpoint_path} lacks model_state")
    model.load_state_dict(checkpoint["model_state"])

    manifest: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "navigation_seed": seed,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "input_config_sha256": input_config_sha256,
        "effective_config_sha256": effective_config_sha256,
        "nav_analysis_dir": str(analysis_dir.resolve()),
        "nav_analysis_inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
            for name, path in {
                "torus_analysis_arrays": analysis_dir / "torus_analysis_arrays.npz",
                "torus_analysis_summary": analysis_dir / "torus_analysis_summary.json",
                "torus_run_provenance": analysis_dir / "run_provenance.json",
            }.items()
            if path.exists()
        },
        "grid_module_unit_indices": grid_units.tolist(),
        "matched_non_grid_unit_indices": (
            None if non_grid_units is None else non_grid_units.tolist()
        ),
        "seq_len": int(args.seq_len),
        "splits": {},
    }
    for split_name, (num_trajectories, offset) in split_specs.items():
        trajectory_seed = seed + offset
        logger.info(
            "Exporting source-%s: %d trajectories x %d steps; trajectory_seed=%d",
            split_name,
            num_trajectories,
            int(args.seq_len),
            trajectory_seed,
        )
        loader = _make_loader(
            cfg,
            num_trajectories=num_trajectories,
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            trajectory_seed=trajectory_seed,
        )
        grid_arrays, non_grid_arrays = _extract_two_modules(
            model,
            loader,
            device,
            grid_units=grid_units,
            non_grid_units=non_grid_units,
        )
        grid_path = output_dir / f"source_{split_name}_arrays.npz"
        np.savez_compressed(grid_path, **grid_arrays)
        split_manifest: Dict[str, object] = {
            "num_trajectories": num_trajectories,
            "trajectory_seed": trajectory_seed,
            "trajectory_seed_offset": offset,
            "grid_arrays": str(grid_path.resolve()),
            "grid_arrays_sha256": sha256_file(grid_path),
            "n_states": int(grid_arrays["hidden"].shape[0]),
            "n_grid_units": int(grid_arrays["hidden"].shape[1]),
            "trajectory_fingerprints": _trajectory_fingerprints_from_arrays(
                grid_arrays
            ),
        }
        if non_grid_arrays is not None:
            control_path = output_dir / f"source_{split_name}_non_grid_arrays.npz"
            np.savez_compressed(control_path, **non_grid_arrays)
            split_manifest.update(
                {
                    "non_grid_arrays": str(control_path.resolve()),
                    "non_grid_arrays_sha256": sha256_file(control_path),
                    "n_non_grid_units": int(non_grid_arrays["hidden"].shape[1]),
                }
            )
        manifest["splits"][split_name] = split_manifest

    split_fingerprints = {
        name: set(record["trajectory_fingerprints"])
        for name, record in manifest["splits"].items()
    }
    overlap_records = []
    names = list(split_fingerprints)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            shared = sorted(split_fingerprints[left].intersection(split_fingerprints[right]))
            if shared:
                overlap_records.append(
                    {
                        "left": left,
                        "right": right,
                        "n_shared": len(shared),
                        "example_fingerprint": shared[0],
                    }
                )
    manifest["trajectory_fingerprint_audit"] = {
        "algorithm": "sha256(n_states, relative_time, rounded_coordinates)",
        "all_disjoint": not overlap_records,
        "pairwise_overlaps": overlap_records,
    }
    if overlap_records:
        raise RuntimeError(
            "Generated source fit/validation/test trajectories overlap; refusing "
            f"to publish split manifest: {overlap_records[0]}"
        )
    manifest["source_tree_sha256"] = source_tree["sha256"]
    manifest["source_tree_file_count"] = source_tree["file_count"]

    manifest_path = output_dir / "source_reduced_split_manifest.json"
    write_json(manifest, manifest_path)
    save_run_provenance(
        output_dir,
        "export_source_reduced_splits",
        cfg=cfg,
        extra={
            "schema_version": SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "input_config_sha256": input_config_sha256,
            "effective_config_sha256": effective_config_sha256,
            "source_tree_sha256": source_tree["sha256"],
            "source_tree_file_count": source_tree["file_count"],
            "split_manifest": str(manifest_path.resolve()),
            "split_manifest_sha256": sha256_file(manifest_path),
        },
    )
    required_outputs = [
        manifest_path,
        output_dir / "run_provenance.json",
        output_dir / "config_used.yaml",
    ]
    for split_name in split_specs:
        required_outputs.append(output_dir / f"source_{split_name}_arrays.npz")
        if non_grid_units is not None:
            required_outputs.append(
                output_dir / f"source_{split_name}_non_grid_arrays.npz"
            )
    missing = [str(path) for path in required_outputs if not path.exists()]
    empty = [str(path) for path in required_outputs if path.exists() and path.stat().st_size == 0]
    if missing or empty:
        raise RuntimeError(
            f"Incomplete source split export: missing={missing}, empty={empty}"
        )
    logger.info("Independent source fit/validation/test exports completed in %s", output_dir)


if __name__ == "__main__":
    main()
