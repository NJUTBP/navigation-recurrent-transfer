"""Export independent Schøyen phase-calibration/fit/validation/test source splits.

All trajectories are freshly generated from the locked navigation checkpoint.
The selected recurrent units are read from the already-completed Schøyen
``torus_topology`` artifact.  The four split trajectory seeds are disjoint by construction, deliberately
separated from the Schøyen clustering (+30001) and TDA (+40001) trajectory
seeds, and recorded in a manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from gpm_cmap.experiments.export_navigation_calibration import _load_selected_units, _make_loader, _make_model
from gpm_cmap.experiments.export_source_reduced_splits import _extract_two_modules
from gpm_cmap.utils import configure_runtime, load_config, save_config_snapshot, save_run_provenance, set_seed, write_json
from gpm_cmap.utils.checkpointing import load_checkpoint, sha256_file
from gpm_cmap.utils.logging_utils import get_logger

SCHEMA_VERSION = "source_reduced_schoyen_v1"


def trajectory_fingerprints(arrays: Dict[str, np.ndarray]) -> Dict[str, list[int]]:
    positions = np.asarray(arrays["positions"], dtype=np.float64)
    trajectory_ids = np.asarray(arrays["trajectory_ids"], dtype=np.int64)
    time_indices = np.asarray(arrays["time_indices"], dtype=np.int64)
    out: Dict[str, list[int]] = {}
    for tid in np.unique(trajectory_ids):
        rows = np.flatnonzero(trajectory_ids == tid)
        rows = rows[np.argsort(time_indices[rows], kind="stable")]
        h = hashlib.sha256()
        h.update(np.asarray([rows.size], dtype=np.int64).tobytes())
        h.update((time_indices[rows] - time_indices[rows][0]).astype(np.int64).tobytes())
        h.update(np.round(positions[rows], 10).astype(np.float64).tobytes())
        out.setdefault(h.hexdigest(), []).append(int(tid))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--topology-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--phase-trajectories", type=int, default=64)
    ap.add_argument("--fit-trajectories", type=int, default=128)
    ap.add_argument("--validation-trajectories", type=int, default=64)
    ap.add_argument("--test-trajectories", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--phase-seed-offset", type=int, default=50001)
    ap.add_argument("--fit-seed-offset", type=int, default=51001)
    ap.add_argument("--validation-seed-offset", type=int, default=52001)
    ap.add_argument("--test-seed-offset", type=int, default=53001)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    cfg.setdefault("data", {})["num_workers"] = 0
    seed = int(cfg["seed"])
    set_seed(seed, deterministic_algorithms=bool(cfg.get("runtime", {}).get("deterministic_algorithms", False)))
    device = configure_runtime(cfg)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    topology_dir = Path(args.topology_dir).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    logger = get_logger("export_schoyen_source_splits", out / "export.log")
    save_config_snapshot(cfg, out)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    selected = _load_selected_units(topology_dir)
    if selected is None:
        raise RuntimeError("Schøyen topology did not provide selected_unit_indices")

    model = _make_model(cfg, device)
    payload = load_checkpoint(checkpoint, map_location=device)
    if "model_state" not in payload:
        raise KeyError(f"{checkpoint} lacks model_state")
    model.load_state_dict(payload["model_state"])

    specs = {
        "phase": (int(args.phase_trajectories), int(args.phase_seed_offset)),
        "fit": (int(args.fit_trajectories), int(args.fit_seed_offset)),
        "validation": (int(args.validation_trajectories), int(args.validation_seed_offset)),
        "test": (int(args.test_trajectories), int(args.test_seed_offset)),
    }
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_seed": seed,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "topology_dir": str(topology_dir),
        "topology_arrays_sha256": sha256_file(topology_dir / "torus_analysis_arrays.npz"),
        "selected_unit_indices": selected.tolist(),
        "seq_len": int(args.seq_len),
        "splits": {},
    }
    all_fingerprints: Dict[str, str] = {}
    for split, (ntraj, offset) in specs.items():
        trajectory_seed = seed + offset
        logger.info("Exporting %s: %d x %d, trajectory_seed=%d", split, ntraj, args.seq_len, trajectory_seed)
        loader = _make_loader(
            cfg,
            num_trajectories=ntraj,
            seq_len=int(args.seq_len),
            batch_size=int(args.batch_size),
            trajectory_seed=trajectory_seed,
        )
        # _extract_two_modules predates this rerun and does not open its own
        # no_grad context.  These exports are evaluation-only; retaining a
        # 500-step autograd graph for a 4096-unit RNN can otherwise exhaust GPU
        # memory for no scientific benefit.
        with torch.no_grad():
            arrays, _ = _extract_two_modules(
                model, loader, device, grid_units=selected, non_grid_units=None
            )
        arrays["schema_version"] = np.asarray(SCHEMA_VERSION)
        arrays["split_name"] = np.asarray(split)
        path = out / f"source_{split}_arrays.npz"
        np.savez_compressed(path, **arrays)
        fps = trajectory_fingerprints(arrays)
        overlap = sorted(set(fps).intersection(all_fingerprints))
        if overlap:
            raise RuntimeError(f"Trajectory fingerprint overlap detected for split {split}: {overlap[:3]}")
        for fp in fps:
            all_fingerprints[fp] = split
        manifest["splits"][split] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "num_trajectories": ntraj,
            "trajectory_seed": trajectory_seed,
            "n_fingerprints": len(fps),
        }
    write_json(manifest, out / "source_reduced_schoyen_split_manifest.json")
    save_run_provenance(
        out,
        "export_schoyen_source_splits",
        cfg=cfg,
        extra={
            "schema_version": SCHEMA_VERSION,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "topology_arrays_sha256": manifest["topology_arrays_sha256"],
            "selected_unit_indices": selected.tolist(),
            "split_manifest": str((out / "source_reduced_schoyen_split_manifest.json").resolve()),
        },
    )
    print(json.dumps({"source_seed": seed, "selected_units": int(selected.size), "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
