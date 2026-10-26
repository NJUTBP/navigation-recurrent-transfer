"""Derive the Figure-A3 formal topology contract from Schøyen Euclidean PH.

This stage does not recompute topology.  It turns the already-selected Schøyen
candidate into an explicit, hashed formal gate for downstream mechanism work.
Candidate selection remains the locked Schøyen procedure; Figure A3 uses the
selected candidate's Euclidean metric as the primary formal topology evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict

from gpm_cmap.utils.checkpointing import sha256_file

SCHEMA_VERSION = "schoyen_euclidean_formal_topology_v1"


def read_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return obj


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def safe_log2_ratio(value: float | None, threshold: float) -> float | None:
    if value is None or value <= 0 or threshold <= 0:
        return None
    return float(math.log2(float(value) / float(threshold)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topology-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--source-seed", required=True, type=int)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--h1-threshold", type=float, default=None)
    ap.add_argument("--h2-threshold", type=float, default=None)
    args = ap.parse_args()

    topology_dir = Path(args.topology_dir).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary_path = topology_dir / "torus_analysis_summary.json"
    arrays_path = topology_dir / "torus_analysis_arrays.npz"
    ensemble_path = topology_dir / "selected_ensemble.json"
    for path in (summary_path, arrays_path, ensemble_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = read_json(summary_path)
    ensemble = read_json(ensemble_path)
    ph = summary.get("persistent_cohomology") or {}
    metrics = ph.get("metrics") or {}
    eu = metrics.get("euclidean") or {}
    selection = summary.get("selection") or {}
    thresholds = selection.get("thresholds") or {}
    h1_thr = float(args.h1_threshold if args.h1_threshold is not None else thresholds.get("H1_2_to_3", 1.5))
    h2_thr = float(args.h2_threshold if args.h2_threshold is not None else thresholds.get("H2_1_to_2", 1.5))
    h0 = int(eu.get("H0_infinite_components", -1))
    h1 = eu.get("H1_second_to_third")
    h2 = eu.get("H2_first_to_second")
    h1 = None if h1 is None else float(h1)
    h2 = None if h2 is None else float(h2)

    checks = {
        "connected_H0": {
            "metric": "euclidean_H0_infinite_components",
            "value": h0,
            "operator": "==",
            "threshold": 1,
            "passed": h0 == 1,
            "log2_margin": None,
        },
        "H1_two_loop_gap": {
            "metric": "euclidean_H1_second_to_third",
            "value": h1,
            "operator": ">=",
            "threshold": h1_thr,
            "passed": bool(h1 is not None and h1 >= h1_thr),
            "log2_margin": safe_log2_ratio(h1, h1_thr),
        },
        "H2_single_void_gap": {
            "metric": "euclidean_H2_first_to_second",
            "value": h2,
            "operator": ">=",
            "threshold": h2_thr,
            "passed": bool(h2 is not None and h2 >= h2_thr),
            "log2_margin": safe_log2_ratio(h2, h2_thr),
        },
    }
    formal_pass = all(v["passed"] for v in checks.values())

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_seed": int(args.source_seed),
        "formal_topology_definition": "selected Schoyen candidate evaluated under Euclidean persistent homology",
        "candidate_selection_scope": "locked Schoyen dual-metric candidate selection; formal qualification uses Euclidean metric only",
        "formal_topology_passed": formal_pass,
        "betti_signature": [1, 2, 1] if formal_pass else None,
        "checks": checks,
        "selected_clusters": selection.get("selected_clusters", ensemble.get("selected_clusters")),
        "n_selected_units": selection.get("n_selected_units", ensemble.get("n_selected_units")),
        "robust_min_gap_score": selection.get("robust_min_gap_score", ensemble.get("robust_min_gap_score")),
        "euclidean": eu,
        "inputs": {
            "torus_analysis_summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
            "torus_analysis_arrays": {"path": str(arrays_path), "sha256": sha256_file(arrays_path)},
            "selected_ensemble": {"path": str(ensemble_path), "sha256": sha256_file(ensemble_path)},
        },
    }
    if args.checkpoint:
        cp = Path(args.checkpoint).expanduser().resolve()
        if not cp.is_file():
            raise FileNotFoundError(cp)
        payload["navigation_checkpoint"] = str(cp)
        payload["navigation_checkpoint_sha256"] = sha256_file(cp)

    write_json(out / "formal_topology_schoyen.json", payload)
    rows = []
    for criterion, check in checks.items():
        rows.append({"source_seed": int(args.source_seed), "criterion": criterion, **check})
    with (out / "formal_topology_components.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({"source_seed": args.source_seed, "formal_topology_passed": formal_pass, "H1_gap": h1, "H2_gap": h2}, indent=2))


if __name__ == "__main__":
    main()
