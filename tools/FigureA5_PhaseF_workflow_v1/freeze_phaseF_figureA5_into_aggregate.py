#!/usr/bin/env python3
"""Copy the Phase F (extended 8 x 8) aggregate tables into an aggregate directory used for plotting.

Writes
  <aggregate>/hard8x8_phaseF_followup/      the complete Phase F aggregate (read by Figs. 6, S5, S6)
  <aggregate>/figure_source_data/FigureA5/  the tables drawn in Figure 6 and Supplementary Figures
                                            S5-S6, with PHASEF_FIGURE_SOURCE_MANIFEST.json (checksums)
Only files are copied: no training, aggregation or statistics.  The released figure data in
data/aggregate_full already contain these tables; use this script after rerunning Phase F, with a
new aggregate directory as target.

    python tools/FigureA5_PhaseF_workflow_v1/freeze_phaseF_figureA5_into_aggregate.py --aggregate outputs/aggregate_full
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.environ.get('NRT_PROJECT_ROOT', str(REPOSITORY_ROOT)))
PHASEF = (PROJECT_ROOT / "rerun_best_validation" / "hard8x8_diagnostics" / "audit_v1"
          / "phaseF_fullscale_v1" / "aggregate")

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


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aggregate", type=Path, required=True, help="target aggregate directory (created if missing)")
    ap.add_argument("--phasef-aggregate", type=Path, default=PHASEF, help=f"Phase F aggregate (default: {PHASEF})")
    args = ap.parse_args()
    phasef = args.phasef_aggregate.resolve()
    agg = args.aggregate.resolve()
    if is_within(agg, (REPOSITORY_ROOT / "data").resolve()):
        raise SystemExit(f"{agg} lies inside the released figure data; choose a new aggregate directory")
    if not phasef.is_dir():
        raise FileNotFoundError(f"Missing Phase-F aggregate: {phasef}")
    readiness = json.loads((phasef / "READINESS.json").read_text())
    if not bool(readiness.get("ready")):
        raise RuntimeError(f"Phase F is not ready: {readiness}")
    agg.mkdir(parents=True, exist_ok=True)

    frozen_root = agg / "hard8x8_phaseF_followup"
    if frozen_root.exists():
        shutil.rmtree(frozen_root)
    shutil.copytree(phasef, frozen_root)

    fig = agg / "figure_source_data" / "FigureA5"
    if fig.exists():
        shutil.rmtree(fig)
    fig.mkdir(parents=True)

    hashes = {}
    for name in SELECTED:
        src = phasef / name
        if not src.is_file():
            raise FileNotFoundError(f"Required Phase-F table missing: {src}")
        dst = fig / name
        shutil.copy2(src, dst)
        hashes[name] = sha256(dst)

    manifest = {
        "figure": "FigureA5",
        "source_stage": "phaseF_fullscale_v1",
        "source_path": str(phasef),
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
    print(f"Phase-F aggregate copied to: {frozen_root}")
    print(f"FigureA5 source data written to: {fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
