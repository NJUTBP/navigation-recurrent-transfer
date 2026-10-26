#!/usr/bin/env python3
from __future__ import annotations
import os

import json
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def compute_rate_map(positions: np.ndarray, activations: np.ndarray, bins: int = 50, arena_size: float = 2.2):
    half = arena_size / 2.0
    hist_range = [[-half, half], [-half, half]]

    occupancy, xedges, yedges = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=bins, range=hist_range
    )
    weighted, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=bins, range=hist_range, weights=activations
    )

    rate_map = np.where(occupancy < 1e-6, 0.0, weighted / np.maximum(occupancy, 1e-6))
    return rate_map, xedges, yedges


def normalize_map(rate_map: np.ndarray):
    rm = np.asarray(rate_map, dtype=float)
    lo = np.nanmin(rm)
    hi = np.nanmax(rm)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(rm, dtype=float)
    return (rm - lo) / (hi - lo)


def find_nav_arrays(project_dir: Path, source_seed: int) -> Path:
    rel = Path(f"nav_seed_{source_seed}") / "artifacts" / "navigation" / "checkpoint_analysis" / "best_validation" / "nav_eval_arrays.npz"
    candidates = [
        project_dir / "sources" / rel,
        project_dir / "rerun_best_validation" / "sources" / rel,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find nav_eval_arrays.npz. Tried:\n" + "\n".join(str(p) for p in candidates)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-dir",
        type=Path,
        default=Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[3]))),
    )
    ap.add_argument(
        "--aggregate-dir",
        type=Path,
        default=None,
        help="Default: <project-dir>/rerun_best_validation/aggregate_full",
    )
    ap.add_argument("--bins", type=int, default=50)
    ap.add_argument("--arena-size", type=float, default=2.2)
    ap.add_argument("--suffix", type=str, default="_50x50")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    project_dir = args.project_dir.resolve()
    aggregate_dir = (
        args.aggregate_dir.resolve()
        if args.aggregate_dir is not None
        else (project_dir / "rerun_best_validation" / "aggregate_full").resolve()
    )
    fig2_dir = aggregate_dir / "figure_source_data" / "Figure2"

    meta_path = fig2_dir / "representative_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_seed = int(meta["source_seed"])

    nav_arrays = find_nav_arrays(project_dir, source_seed)
    print(f"[INFO] representative source_seed = {source_seed}")
    print(f"[INFO] nav arrays = {nav_arrays}")
    print(f"[INFO] fig2 dir   = {fig2_dir}")

    with np.load(nav_arrays, allow_pickle=False) as z:
        positions = np.asarray(z["positions"], dtype=float)
        hidden = np.asarray(z["hidden"], dtype=float)

    if hidden.ndim != 2:
        raise ValueError(f"Expected hidden to be 2D [samples, units], got {hidden.shape}")
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"Expected positions to be [samples, 2], got {positions.shape}")

    for rank in [1, 2, 3, 4]:
        old_csv = fig2_dir / f"rate_map_rank_{rank}.csv"
        if not old_csv.exists():
            raise FileNotFoundError(f"Missing {old_csv}")

        old = pd.read_csv(old_csv)
        if old.empty:
            raise ValueError(f"{old_csv} is empty")

        unit = int(old["unit"].iloc[0])
        grid_score = float(old["grid_score"].iloc[0])
        selected_by_schoyen = bool(old["selected_by_schoyen"].iloc[0]) if "selected_by_schoyen" in old.columns else False

        if unit < 0 or unit >= hidden.shape[1]:
            raise IndexError(f"Unit {unit} out of range for hidden shape {hidden.shape}")

        rm, _, _ = compute_rate_map(
            positions=positions,
            activations=hidden[:, unit],
            bins=args.bins,
            arena_size=args.arena_size,
        )
        norm = normalize_map(rm)

        yy, xx = np.indices(norm.shape)
        new = pd.DataFrame({
            "x_bin": xx.ravel(),
            "y_bin": yy.ravel(),
            "normalized_activation": norm.ravel(),
            "unit": unit,
            "grid_score": grid_score,  # keep frozen score/rank
            "selected_by_schoyen": selected_by_schoyen,
        })

        if args.overwrite:
            backup = fig2_dir / f"rate_map_rank_{rank}.csv.bak20x20"
            if not backup.exists():
                old_csv.replace(backup)
            out_csv = fig2_dir / f"rate_map_rank_{rank}.csv"
        else:
            out_csv = fig2_dir / f"rate_map_rank_{rank}{args.suffix}.csv"

        new.to_csv(out_csv, index=False)
        print(f"[OK] rank {rank}: unit={unit}, rows={len(new)}, out={out_csv}")

    print("\n[DONE] Finished regenerating Figure2D rate-map CSVs.")
    if not args.overwrite:
        print("Generated non-destructive *_50x50.csv files.")
        print("After checking them, you can manually swap them into rate_map_rank_1~4.csv.")


if __name__ == "__main__":
    main()
