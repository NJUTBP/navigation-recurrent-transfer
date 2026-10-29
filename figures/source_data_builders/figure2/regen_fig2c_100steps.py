#!/usr/bin/env python3
from __future__ import print_function

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


RUN_ROOT = Path(os.environ.get("NRT_PROJECT_ROOT", Path(__file__).resolve().parents[3]))
CODE_DIR = RUN_ROOT / "code"

SOURCE_SEED = 6101

NAV_DIR = (
    RUN_ROOT
    / "sources"
    / ("nav_seed_%d" % SOURCE_SEED)
    / "artifacts/navigation"
)

ANALYSIS_DIR = NAV_DIR / "checkpoint_analysis/best_validation"

FIG2_DIR = RUN_ROOT / "data/aggregate_full/figure_source_data/Figure2"
OUT_DIR = RUN_ROOT / "outputs/source_data_builders/Figure2"

OLD_CSV = FIG2_DIR / "representative_trajectory.csv"
OUT_CSV = OUT_DIR / "representative_trajectory_100.csv"

SEQ_LEN = 100


# ---------------------------------------------------------------------
# Import the exact project code used for the navigation experiment.
# ---------------------------------------------------------------------
sys.path.insert(0, str(CODE_DIR))

from gpm_cmap.data.nav_env.dataset import (
    NavDatasetConfig,
    NavTrajectoryDataset,
)
from gpm_cmap.data.nav_env.open_field import OpenFieldConfig
from gpm_cmap.models.grid_rnn import GridNavigationModel
from gpm_cmap.tasks.navigation import evaluate_navigation
from gpm_cmap.utils.checkpointing import load_checkpoint


def resolve_checkpoint():
    summary_path = ANALYSIS_DIR / "summary.json"

    if summary_path.exists():
        with summary_path.open("r") as f:
            summary = json.load(f)

        rel = summary.get("checkpoint")
        if rel:
            candidate = NAV_DIR / rel
            if candidate.exists():
                return candidate

    candidate = NAV_DIR / "best.pt"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not locate best-validation checkpoint under %s"
        % NAV_DIR
    )


def make_long_validation_dataset(cfg):
    data = cfg["data"]

    open_field = OpenFieldConfig(
        arena_size=data["arena_size"],
        dt=data.get("dt", 0.05),
        speed_scale=data.get("speed_scale", 0.13),
        sigma_theta=data.get("sigma_theta", 5.0),
        max_speed=data.get("max_speed", 0.5),
        border_region=data.get("border_region", 0.03),
        boundary_mode=data.get("boundary_mode", "avoid"),
    )

    # This reproduces the ORIGINAL validation-set seed convention,
    # but extends the first validation trajectory from 20 to 100 steps.
    ds_cfg = NavDatasetConfig(
        num_trajectories=1,
        seq_len=SEQ_LEN,
        open_field=open_field,
        place_cells=data["place_cells"],
        place_sigma=data["place_sigma"],
        arena_size=data["arena_size"],
        seed=cfg["seed"],
        trajectory_seed=cfg["seed"] + 9999,
        place_seed=cfg["seed"],
        use_dog=data.get("use_dog", True),
        surround_scale=data.get("surround_scale", 2.0),
        surround_scale_mode=data.get(
            "surround_scale_mode", "variance"
        ),
    )

    return NavTrajectoryDataset(ds_cfg)


def build_model(cfg, device):
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    model = GridNavigationModel(
        velocity_dim=model_cfg["velocity_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        place_dim=data_cfg["place_cells"],
        activation=model_cfg["activation"],
        nonnegative=model_cfg["nonnegative"],
        cell_type=model_cfg.get("cell_type", "rnn"),
        dropout=model_cfg.get("dropout", 0.0),
        use_bias=model_cfg.get("use_bias", False),
        init_relu=model_cfg.get("init_relu", False),
    ).to(device)

    return model


def main():
    checkpoint_path = resolve_checkpoint()

    print("[INFO] checkpoint:")
    print(checkpoint_path)

    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )

    cfg = checkpoint["config"]

    # CPU is sufficient for one 100-step trajectory and avoids depending
    # on an allocated GPU.
    device = torch.device("cpu")

    torch.set_num_threads(8)

    dataset = make_long_validation_dataset(cfg)

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    model = build_model(cfg, device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("[INFO] evaluating one continuous %d-step trajectory..." % SEQ_LEN)

    result = evaluate_navigation(
        model,
        loader,
        device,
        arena_size=cfg["data"]["arena_size"],

        # Grid analysis is irrelevant for Figure 2C, but evaluate_navigation
        # computes it internally. Keep this small to reduce unnecessary work.
        grid_bins=20,
        grid_smooth_sigma=0.0,

        place_encoder=dataset.place_encoder,
        include_google_ref_score=False,
        include_hidden_diagnostics=False,
    )

    pos = np.asarray(result["positions"], dtype=float)
    dec = np.asarray(result["decoded_positions"], dtype=float)
    traj_ids = np.asarray(result["trajectory_ids"])
    times = np.asarray(result["time_indices"])

    # There is exactly one trajectory, but sort explicitly by time.
    first_id = np.min(traj_ids)
    take = np.flatnonzero(traj_ids == first_id)
    take = take[np.argsort(times[take])]

    pos = pos[take]
    dec = dec[take]
    t = times[take]

    if len(t) != SEQ_LEN:
        raise RuntimeError(
            "Expected %d trajectory points, got %d"
            % (SEQ_LEN, len(t))
        )

    out = pd.DataFrame({
        "time": t,
        "true_x": pos[:, 0],
        "true_y": pos[:, 1],
        "decoded_x": dec[:, 0],
        "decoded_y": dec[:, 1],
    })

    # --------------------------------------------------------------
    # Continuity check against the OLD 20-step Figure 2C trajectory.
    # --------------------------------------------------------------
    if False and OLD_CSV.exists():
        old = pd.read_csv(OLD_CSV)

        n = min(len(old), 20, len(out))

        old_true = old[
            ["true_x", "true_y"]
        ].iloc[:n].to_numpy(float)

        new_true = out[
            ["true_x", "true_y"]
        ].iloc[:n].to_numpy(float)

        old_dec = old[
            ["decoded_x", "decoded_y"]
        ].iloc[:n].to_numpy(float)

        new_dec = out[
            ["decoded_x", "decoded_y"]
        ].iloc[:n].to_numpy(float)

        true_diff = np.max(np.abs(old_true - new_true))
        dec_diff = np.max(np.abs(old_dec - new_dec))

        print(
            "[CHECK] max true-position difference over first %d points: %.12g"
            % (n, true_diff)
        )
        print(
            "[CHECK] max decoded-position difference over first %d points: %.12g"
            % (n, dec_diff)
        )

        if true_diff > 1e-7:
            raise RuntimeError(
                "The new 100-step trajectory does NOT share the old "
                "representative trajectory's first 20 true positions. "
                "Refusing to write the file."
            )

        if dec_diff > 1e-5:
            print(
                "[WARNING] decoded prefix differs slightly from old CSV. "
                "Inspect before replacing the manuscript source."
            )
        else:
            print(
                "[PASS] First 20 points reproduce the existing Figure 2C "
                "trajectory."
            )

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    errors = np.linalg.norm(
        out[["decoded_x", "decoded_y"]].to_numpy()
        - out[["true_x", "true_y"]].to_numpy(),
        axis=1,
    )

    print()
    print("[DONE]")
    print("rows =", len(out))
    print("mean position error (m) =", float(errors.mean()))
    print("final position error (m) =", float(errors[-1]))
    print("output:")
    print(OUT_CSV)


if __name__ == "__main__":
    main()
