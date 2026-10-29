#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("NRT_PROJECT_ROOT", Path(__file__).resolve().parents[3]))

ARRAYS = (
    ROOT
    / "sources/nav_seed_6101/artifacts/navigation/"
      "checkpoint_analysis/best_validation/nav_eval_arrays.npz"
)

OUT_DIR = ROOT / "outputs/source_data_builders/Figure2/trajectory_candidates_20step"

N_EXPORT = 100
EXPECTED_STEPS = 20


def main():

    if not ARRAYS.exists():
        raise FileNotFoundError(ARRAYS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(ARRAYS, allow_pickle=False) as z:
        positions = np.asarray(z["positions"], dtype=float)
        decoded = np.asarray(z["decoded_positions"], dtype=float)
        trajectory_ids = np.asarray(z["trajectory_ids"])
        time_indices = np.asarray(z["time_indices"])

    unique_ids = np.unique(trajectory_ids)
    unique_ids.sort()

    print("[INFO] total trajectories in nav_eval_arrays:", len(unique_ids))

    summary_rows = []
    exported = 0

    for trajectory_id in unique_ids:

        idx = np.flatnonzero(trajectory_ids == trajectory_id)
        idx = idx[np.argsort(time_indices[idx])]

        # Only export complete ordinary 20-step trajectories.
        if len(idx) != EXPECTED_STEPS:
            continue

        pos = positions[idx]
        dec = decoded[idx]
        times = time_indices[idx]

        error = np.linalg.norm(dec - pos, axis=1)

        df = pd.DataFrame({
            "time": times,
            "true_x": pos[:, 0],
            "true_y": pos[:, 1],
            "decoded_x": dec[:, 0],
            "decoded_y": dec[:, 1],
        })

        outfile = OUT_DIR / (
            "representative_trajectory_id_%04d.csv"
            % int(trajectory_id)
        )
        df.to_csv(outfile, index=False)

        summary_rows.append({
            "trajectory_id": int(trajectory_id),
            "n_steps": int(len(idx)),
            "mean_error_m": float(error.mean()),
            "final_error_m": float(error[-1]),
            "max_error_m": float(error.max()),
            "path_length_m": float(
                np.linalg.norm(
                    np.diff(pos, axis=0),
                    axis=1
                ).sum()
            ),
            "file": outfile.name,
        })

        exported += 1

        if exported >= N_EXPORT:
            break

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        OUT_DIR / "trajectory_candidates_summary.csv",
        index=False,
    )

    if exported == 0:
        raise RuntimeError("No complete 20-step trajectories found.")

    print()
    print("[DONE]")
    print("exported =", exported)
    print("directory =", OUT_DIR)
    print()
    print(summary[
        [
            "trajectory_id",
            "mean_error_m",
            "final_error_m",
            "path_length_m",
        ]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
