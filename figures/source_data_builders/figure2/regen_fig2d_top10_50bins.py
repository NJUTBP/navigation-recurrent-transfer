#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("NRT_PROJECT_ROOT", Path(__file__).resolve().parents[3]))

NAV_ARRAYS = (
    ROOT
    / "sources/nav_seed_6101/artifacts/navigation/"
      "checkpoint_analysis/best_validation/nav_eval_arrays.npz"
)

FIG2_DIR = ROOT / "data/aggregate_full/figure_source_data/Figure2"

GRID_SCORES = FIG2_DIR / "representative_grid_scores.csv"

OUT_DIR = ROOT / "outputs/source_data_builders/Figure2/rate_maps_top10_50x50"

BINS = 50
ARENA_SIZE = 2.2


def compute_rate_map(positions, activations, bins=50, arena_size=2.2):
    half = arena_size / 2.0
    hist_range = [[-half, half], [-half, half]]

    occupancy, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=bins,
        range=hist_range,
    )

    weighted, _, _ = np.histogram2d(
        positions[:, 0],
        positions[:, 1],
        bins=bins,
        range=hist_range,
        weights=activations,
    )

    rate_map = np.where(
        occupancy < 1e-6,
        0.0,
        weighted / np.maximum(occupancy, 1e-6),
    )

    return rate_map


def normalize_map(rate_map):
    lo = np.nanmin(rate_map)
    hi = np.nanmax(rate_map)

    if (not np.isfinite(lo)) or (not np.isfinite(hi)) or (hi - lo < 1e-12):
        return np.zeros_like(rate_map, dtype=float)

    return (rate_map - lo) / (hi - lo)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not NAV_ARRAYS.exists():
        raise FileNotFoundError(str(NAV_ARRAYS))

    if not GRID_SCORES.exists():
        raise FileNotFoundError(str(GRID_SCORES))

    print("[INFO] nav arrays:")
    print(NAV_ARRAYS)

    print("[INFO] grid ranking:")
    print(GRID_SCORES)

    with np.load(NAV_ARRAYS, allow_pickle=False) as z:
        positions = np.asarray(z["positions"], dtype=float)
        hidden = np.asarray(z["hidden"], dtype=float)

    ranking = pd.read_csv(GRID_SCORES)

    if "rank" not in ranking.columns:
        ranking = ranking.sort_values(
            "grid_score",
            ascending=False,
            na_position="last",
            kind="stable",
        ).reset_index(drop=True)
        ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))

    ranking = ranking.sort_values("rank").reset_index(drop=True)

    top10 = ranking.iloc[:10].copy()

    print()
    print("[INFO] Top 10 units:")
    print(top10[["rank", "unit", "grid_score"]].to_string(index=False))
    print()

    summary_rows = []

    for _, row in top10.iterrows():

        rank = int(row["rank"])
        unit = int(row["unit"])
        grid_score = float(row["grid_score"])

        selected = False
        if "selected_by_schoyen" in row.index:
            selected = bool(row["selected_by_schoyen"])

        if unit < 0 or unit >= hidden.shape[1]:
            raise IndexError(
                "Unit {} outside hidden shape {}".format(
                    unit, hidden.shape
                )
            )

        rate_map = compute_rate_map(
            positions,
            hidden[:, unit],
            bins=BINS,
            arena_size=ARENA_SIZE,
        )

        normalized = normalize_map(rate_map)

        yy, xx = np.indices(normalized.shape)

        out = pd.DataFrame({
            "x_bin": xx.ravel(),
            "y_bin": yy.ravel(),
            "normalized_activation": normalized.ravel(),
            "unit": unit,
            "grid_score": grid_score,
            "selected_by_schoyen": selected,
        })

        outfile = OUT_DIR / "rate_map_rank_{}.csv".format(rank)
        out.to_csv(outfile, index=False)

        summary_rows.append({
            "rank": rank,
            "unit": unit,
            "grid_score": grid_score,
            "rows": len(out),
            "bins": BINS,
        })

        print(
            "[OK] rank {:2d} | unit {:4d} | G={:.6f} | {} rows".format(
                rank, unit, grid_score, len(out)
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "top10_summary.csv", index=False)

    print()
    print("[DONE]")
    print("Output directory:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
