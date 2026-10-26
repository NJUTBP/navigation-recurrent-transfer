#!/usr/bin/env python3
"""Freeze final Phase-F harder-8x8 plotting data into aggregate_full.

This is a light file-management step only. It does not rerun training, aggregation,
or statistics. It preserves the previous FigureA5 source-data directory as a legacy
backup, then replaces FigureA5 plotting inputs with the frozen Phase-F tables.
"""
from __future__ import annotations
import os

import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone

RUN = Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[2])))
RERUN = RUN / "rerun_best_validation"
PHASEF = RERUN / "hard8x8_diagnostics" / "audit_v1" / "phaseF_fullscale_v1" / "aggregate"
AGG_LINK = RERUN / "aggregate_full"

SELECTED = [
    "condition_performance_matrix.csv",
    "source_level_metrics.csv",
    "source_level_baselines.csv",
    "primary_generalization_contrasts.csv",
    "secondary_id_contrasts.csv",
    "all_per_length.csv",
    "source_level_relation_interventions.csv",
    "source_level_condition_diagnostics.csv",
    "all_update_curves.csv",
    "FINAL_RESULT.json",
    "FINAL_RESULT.md",
    "READINESS.json",
    "READINESS.md",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not PHASEF.is_dir():
        raise FileNotFoundError(f"Missing Phase-F aggregate: {PHASEF}")
    readiness = json.loads((PHASEF / "READINESS.json").read_text())
    if not bool(readiness.get("ready")):
        raise RuntimeError(f"Phase F is not ready: {readiness}")

    agg = AGG_LINK.resolve()
    if not agg.is_dir():
        raise FileNotFoundError(f"aggregate_full does not resolve to a directory: {AGG_LINK} -> {agg}")

    # Store the complete final follow-up aggregate inside aggregate_full so the
    # user's normal 'download aggregate_full and redraw locally' workflow is self-contained.
    frozen_root = agg / "hard8x8_phaseF_followup"
    if frozen_root.exists():
        shutil.rmtree(frozen_root)
    shutil.copytree(PHASEF, frozen_root)

    source_root = agg / "figure_source_data"
    source_root.mkdir(parents=True, exist_ok=True)
    fig = source_root / "FigureA5"
    legacy = source_root / "FigureA5_legacy_pre_phaseF"

    if fig.exists() and not legacy.exists():
        shutil.copytree(fig, legacy)
    if fig.exists():
        shutil.rmtree(fig)
    fig.mkdir(parents=True)

    hashes = {}
    for name in SELECTED:
        src = PHASEF / name
        if not src.is_file():
            raise FileNotFoundError(f"Required Phase-F table missing: {src}")
        dst = fig / name
        shutil.copy2(src, dst)
        hashes[name] = sha256(dst)

    manifest = {
        "figure": "FigureA5",
        "source_stage": "phaseF_fullscale_v1",
        "source_path": str(PHASEF),
        "aggregate_full_resolved": str(agg),
        "frozen_followup_copy": str(frozen_root),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "n_navigation_sources": 10,
        "n_fresh_targets_per_source": 5,
        "n_training_conditions": 15,
        "plot_contract": "Phase-F final result only; legacy low-competence FigureA5 is not plotted",
        "files_sha256": hashes,
    }
    (fig / "PHASEF_FIGURE_SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print("SUCCESS")
    print(f"aggregate_full resolves to: {agg}")
    print(f"Phase-F final aggregate copied to: {frozen_root}")
    print(f"FigureA5 source data frozen at: {fig}")
    if legacy.exists():
        print(f"Legacy FigureA5 source data retained at: {legacy}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
