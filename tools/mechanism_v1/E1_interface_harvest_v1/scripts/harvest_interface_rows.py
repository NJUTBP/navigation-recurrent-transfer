#!/usr/bin/env python3
"""E1 interface harvest: extract learned relation-interface metrics from the
historical exp8 symbolic-adapter artifacts.

Scientific target
-----------------
For every navigation source and every symbolic target repeat, exp8 already
stored the learned 2xK relation adapter ``W_rel`` together with a geometry
report.  This script collects, per (source, target, arm, budget):

  * the MAGNITUDE of the learned relation vectors
        M_rel_codebook = mean_k ||W_rel[:,k]|| / ||codebook column||
    which is the exact analogue of the Phase F 8x8 scalar ``gain``;
  * the SHAPE metrics that exp8 pre-registered as criterion S2
        antipodality_index, pair_axis_angle_deg, procrustes_residual;
  * the recurrence-ablation drop, plus a chance-normalised version.

Nothing is recomputed from checkpoints.  This is a pure read of frozen
artifacts and is safe to run on a login node, although the project convention
is to submit it through LSF anyway.

Key implementation notes
------------------------
1.  Target seeds are DISCOVERED by globbing, never derived from a formula.
    ``symbolic_baseline7000_t*`` directories are excluded by an exact regex on
    the attempt directory name.
2.  Both historical layouts are searched:
        sources/nav_seed_<S>/artifacts/symbolic_t<T>/symbolic_adapter_rows.json
        sources/nav_seed_<S>/attempts/symbolic_t<T>/try_<NNN>/symbolic_adapter_rows.json
    Candidates are de-duplicated by realpath; the ``artifacts`` view is
    preferred when both resolve to different files.
3.  ``column_norms`` lives ONLY in the JSON, not in the flattened CSV that sits
    beside it.  The magnitude analysis therefore requires the JSON.
4.  The codebook magnitude is read from ``protocol_manifest.json`` in the same
    attempt directory rather than hard-coded, so the normalisation is exact.
5.  ``1 / procrustes_scale`` should equal ``M_rel_codebook`` whenever the
    learned frame is close to a rotation+scale of the codebook.  The ratio of
    the two is written out as ``scale_consistency`` and is a harvest-integrity
    diagnostic, not a filter: at tiny budgets the frame is far from canonical
    and the two legitimately disagree.

Usage
-----
    "$PYTHON_BIN" harvest_interface_rows.py \
        --project-root <PROJECT_ROOT>/.../multinav_paper_v1 \
        --out          <PROJECT_ROOT>/.../rerun_best_validation/followups/mechanism_v1/E1_interface
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Navigation input scale, used only for the SECONDARY normalisation.
# Trajectory speeds are Rayleigh(sigma) with sigma = 0.8168140899 m/s and
# dt = 0.02 s, so the mean per-step displacement is sigma * sqrt(pi/2) * dt.
DEFAULT_SPEED_SCALE = 0.8168140899
DEFAULT_DT = 0.02

SOURCE_SEEDS = tuple(range(6101, 6111))
REQUIRED_ARMS = ("nav_frozen", "ctl01_frozen", "random_frozen")
ATTEMPT_NAME_RE = re.compile(r"^symbolic_t(\d+)$")
TRY_NAME_RE = re.compile(r"^try_(\d+)$")

ROW_FIELDS = [
    "source_seed", "target_seed", "arm", "arm_kind", "budget",
    "test_accuracy", "best_val_accuracy", "n_trainable_params",
    "col_norm_mean", "col_norm_min", "col_norm_max", "col_norm_anisotropy",
    "codebook_norm", "M_rel_codebook", "M_rel_navstep",
    "procrustes_scale", "inv_procrustes_scale", "scale_consistency",
    "antipodality_index", "pair_axis_angle_deg",
    "procrustes_residual", "procrustes_rotation_deg",
    "test_accuracy_no_recurrence", "recurrence_ablation_drop",
    "recurrence_drop_normalized",
    "n_nodes", "chance_level", "source_checkpoint", "attempt_dir",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    tmp.replace(path)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")
    tmp.replace(path)


def nav_step_metres(speed_scale: float, dt: float) -> float:
    """Mean per-step displacement of a Rayleigh(speed_scale) walker."""
    return float(speed_scale) * math.sqrt(math.pi / 2.0) * float(dt)


def read_nav_step(project_root: Path, source_seed: int) -> tuple[float, str]:
    """Read speed_scale / dt from the navigation config when available.

    Falls back to the documented defaults.  Returns (step_metres, provenance).
    """
    config = (project_root / "sources" / f"nav_seed_{source_seed}"
              / "artifacts" / "navigation" / "config_used.yaml")
    if config.is_file():
        try:
            import yaml  # available in the project environment
            payload = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
            data = payload.get("data", {}) or {}
            speed = float(data["speed_scale"])
            dt = float(data["dt"])
            return nav_step_metres(speed, dt), f"config_used.yaml:{config}"
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            return (nav_step_metres(DEFAULT_SPEED_SCALE, DEFAULT_DT),
                    f"default (config unreadable: {exc})")
    return nav_step_metres(DEFAULT_SPEED_SCALE, DEFAULT_DT), "default (config absent)"


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------

def candidate_paths(rerun_root: Path, source_seed: int) -> list[dict[str, Any]]:
    """All plausible main-symbolic row files for one navigation source."""
    base = rerun_root / "sources" / f"nav_seed_{source_seed}"
    found: list[dict[str, Any]] = []

    artifacts = base / "artifacts"
    if artifacts.is_dir():
        for attempt in sorted(artifacts.iterdir()):
            match = ATTEMPT_NAME_RE.match(attempt.name)
            if not (attempt.is_dir() and match):
                continue
            path = attempt / "symbolic_adapter_rows.json"
            if path.is_file():
                found.append({
                    "source_seed": source_seed,
                    "target_seed": int(match.group(1)),
                    "layout": "artifacts",
                    "try_index": -1,
                    "path": path,
                })

    attempts = base / "attempts"
    if attempts.is_dir():
        for attempt in sorted(attempts.iterdir()):
            match = ATTEMPT_NAME_RE.match(attempt.name)
            if not (attempt.is_dir() and match):
                continue
            for try_dir in sorted(attempt.iterdir()):
                try_match = TRY_NAME_RE.match(try_dir.name)
                if not (try_dir.is_dir() and try_match):
                    continue
                path = try_dir / "symbolic_adapter_rows.json"
                if path.is_file():
                    found.append({
                        "source_seed": source_seed,
                        "target_seed": int(match.group(1)),
                        "layout": "attempts",
                        "try_index": int(try_match.group(1)),
                        "path": path,
                    })
    return found


def validate_rows(rows: list[dict[str, Any]]) -> tuple[bool, str, dict[str, Any]]:
    """Require all three recurrent arms with an identical, complete budget set."""
    if not isinstance(rows, list) or not rows:
        return False, "empty or non-list payload", {}

    budgets_by_arm: dict[str, set[int]] = {}
    for row in rows:
        arm = str(row.get("arm", ""))
        budget = row.get("budget")
        if arm and budget is not None:
            budgets_by_arm.setdefault(arm, set()).add(int(budget))

    missing = [arm for arm in REQUIRED_ARMS if arm not in budgets_by_arm]
    if missing:
        return False, f"missing required arms: {missing}", {"arms": sorted(budgets_by_arm)}

    reference = budgets_by_arm["nav_frozen"]
    mismatched = {
        arm: sorted(budgets_by_arm[arm] ^ reference)
        for arm in REQUIRED_ARMS
        if budgets_by_arm[arm] != reference
    }
    if mismatched:
        return False, f"budget sets differ across required arms: {mismatched}", {}

    geometry_missing = [
        (row["arm"], row["budget"])
        for row in rows
        if str(row.get("arm")) in REQUIRED_ARMS
        and not (row.get("adapter_geometry") or {}).get("column_norms")
    ]
    if geometry_missing:
        return False, f"adapter_geometry.column_norms absent for {geometry_missing[:4]}", {}

    detail = {
        "arms": sorted(budgets_by_arm),
        "budgets_by_required_arm": {arm: sorted(budgets_by_arm[arm]) for arm in REQUIRED_ARMS},
        "n_rows": len(rows),
    }
    return True, "ok", detail


def select_files(rerun_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (selected, scan_log). One selected file per (source, target)."""
    scan: list[dict[str, Any]] = []
    by_key: dict[tuple[int, int], list[dict[str, Any]]] = {}
    seen_real: set[str] = set()

    for source_seed in SOURCE_SEEDS:
        for candidate in candidate_paths(rerun_root, source_seed):
            real = str(candidate["path"].resolve())
            entry = dict(candidate)
            entry["realpath"] = real
            entry["duplicate_of_earlier_realpath"] = real in seen_real
            try:
                rows = read_json(candidate["path"])
                ok, reason, detail = validate_rows(rows)
            except Exception as exc:  # noqa: BLE001
                ok, reason, detail = False, f"unreadable: {exc}", {}
            entry.update({"ok": ok, "reason": reason, **detail})
            entry["path"] = str(candidate["path"])
            scan.append(entry)
            if not ok:
                continue
            seen_real.add(real)
            by_key.setdefault((source_seed, candidate["target_seed"]), []).append(
                {**candidate, "realpath": real}
            )

    selected: list[dict[str, Any]] = []
    for key in sorted(by_key):
        group = by_key[key]
        # Prefer the canonical ``artifacts`` view; otherwise the highest try index.
        group.sort(key=lambda item: (item["layout"] != "artifacts", -item["try_index"]))
        selected.append(group[0])
    return selected, scan


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------

def codebook_norm_and_nodes(attempt_dir: Path, target_seed: int) -> tuple[float, int, str]:
    """Read the exact action-codebook magnitude and node count for this attempt."""
    manifest = attempt_dir / "protocol_manifest.json"
    if not manifest.is_file():
        return 0.02, 16, "default (protocol_manifest.json absent)"
    payload = read_json(manifest)
    entries = payload.get("seeds") or []
    chosen = None
    for entry in entries:
        if int(entry.get("seed", -1)) == int(target_seed):
            chosen = entry
            break
    if chosen is None and entries:
        chosen = entries[0]
    if chosen is None:
        return 0.02, 16, "default (no seed entry)"
    codebook = np.asarray(chosen.get("action_codebook") or [], dtype=np.float64)
    if codebook.ndim != 2 or codebook.shape[0] < 1:
        return 0.02, int(chosen.get("n_nodes", 16)), "default (malformed codebook)"
    norms = np.linalg.norm(codebook, axis=1)
    return float(norms.mean()), int(chosen.get("n_nodes", 16)), str(manifest)


def extract_rows(entry: dict[str, Any], nav_step: float) -> list[dict[str, Any]]:
    path = Path(entry["path"]) if isinstance(entry["path"], str) else entry["path"]
    attempt_dir = path.parent
    source_seed = int(entry["source_seed"])
    target_seed = int(entry["target_seed"])
    codebook_norm, n_nodes, codebook_provenance = codebook_norm_and_nodes(attempt_dir, target_seed)
    chance = 1.0 / float(n_nodes) if n_nodes else float("nan")

    out: list[dict[str, Any]] = []
    for raw in read_json(path):
        geometry = raw.get("adapter_geometry") or {}
        norms = geometry.get("column_norms")
        record: dict[str, Any] = {
            "source_seed": source_seed,
            "target_seed": target_seed,
            "arm": raw.get("arm"),
            "arm_kind": raw.get("arm_kind"),
            "budget": raw.get("budget"),
            "test_accuracy": raw.get("test_accuracy"),
            "best_val_accuracy": raw.get("best_val_accuracy"),
            "n_trainable_params": raw.get("n_trainable_params"),
            "codebook_norm": codebook_norm,
            "antipodality_index": raw.get("antipodality_index"),
            "pair_axis_angle_deg": raw.get("pair_axis_angle_deg"),
            "procrustes_residual": raw.get("procrustes_residual"),
            "procrustes_rotation_deg": raw.get("procrustes_rotation_deg"),
            "procrustes_scale": geometry.get("procrustes_scale"),
            "test_accuracy_no_recurrence": raw.get("test_accuracy_no_recurrence"),
            "recurrence_ablation_drop": raw.get("recurrence_ablation_drop"),
            "n_nodes": n_nodes,
            "chance_level": chance,
            "source_checkpoint": raw.get("source"),
            "attempt_dir": str(attempt_dir),
        }

        if norms:
            vector = np.asarray(norms, dtype=np.float64)
            mean_norm = float(vector.mean())
            record["col_norm_mean"] = mean_norm
            record["col_norm_min"] = float(vector.min())
            record["col_norm_max"] = float(vector.max())
            record["col_norm_anisotropy"] = float(vector.max() / max(vector.min(), 1e-12))
            record["M_rel_codebook"] = mean_norm / max(codebook_norm, 1e-12)
            record["M_rel_navstep"] = mean_norm / max(nav_step, 1e-12)
            scale = geometry.get("procrustes_scale")
            if scale:
                inverse = 1.0 / float(scale)
                record["inv_procrustes_scale"] = inverse
                record["scale_consistency"] = abs(
                    record["M_rel_codebook"] - inverse
                ) / max(record["M_rel_codebook"], 1e-12)

        accuracy = raw.get("test_accuracy")
        no_recurrence = raw.get("test_accuracy_no_recurrence")
        if accuracy is not None and no_recurrence is not None and math.isfinite(chance):
            headroom = float(accuracy) - chance
            if headroom > 1e-9:
                record["recurrence_drop_normalized"] = (
                    (float(accuracy) - float(no_recurrence)) / headroom
                )

        out.append(record)
    out.sort(key=lambda item: (str(item["arm"]), int(item["budget"] or 0)))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="E1 interface harvest")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--require-complete", action="store_true",
        help="exit non-zero unless exactly 3 valid target repeats were found per source",
    )
    args = parser.parse_args()

    project_root: Path = args.project_root.resolve()
    rerun_root = project_root / "rerun_best_validation"
    if not rerun_root.is_dir():
        print(f"ERROR: rerun root not found: {rerun_root}", file=sys.stderr)
        return 2

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    selected, scan = select_files(rerun_root)
    atomic_json(out_dir / "candidate_scan.json", scan)

    if not selected:
        print("ERROR: no valid main symbolic row files were discovered", file=sys.stderr)
        return 3

    nav_step_by_source: dict[int, tuple[float, str]] = {
        seed: read_nav_step(project_root, seed) for seed in SOURCE_SEEDS
    }

    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for entry in selected:
        nav_step, nav_step_provenance = nav_step_by_source[int(entry["source_seed"])]
        extracted = extract_rows(entry, nav_step)
        rows.extend(extracted)
        path = Path(entry["path"])
        inputs.append({
            "source_seed": int(entry["source_seed"]),
            "target_seed": int(entry["target_seed"]),
            "layout": entry["layout"],
            "path": str(path),
            "realpath": entry["realpath"],
            "sha256": sha256_file(path),
            "n_rows": len(extracted),
            "nav_step_m": nav_step,
            "nav_step_provenance": nav_step_provenance,
        })

    write_csv(out_dir / "interface_rows.csv", rows, ROW_FIELDS)

    per_source_targets: dict[int, list[int]] = {}
    for entry in selected:
        per_source_targets.setdefault(int(entry["source_seed"]), []).append(int(entry["target_seed"]))

    inconsistent = {
        source: sorted(targets)
        for source, targets in per_source_targets.items()
        if len(targets) != 3
    }
    missing_sources = [seed for seed in SOURCE_SEEDS if seed not in per_source_targets]

    scale_values = [
        row["scale_consistency"] for row in rows
        if row.get("scale_consistency") is not None
        and str(row.get("arm")) in REQUIRED_ARMS
        and int(row.get("budget") or 0) >= 1000
    ]

    manifest = {
        "schema_version": "e1_interface_harvest_v1",
        "project_root": str(project_root),
        "output_dir": str(out_dir),
        "n_sources_found": len(per_source_targets),
        "n_files_selected": len(selected),
        "n_rows": len(rows),
        "targets_by_source": {str(k): sorted(v) for k, v in sorted(per_source_targets.items())},
        "sources_with_unexpected_repeat_count": {str(k): v for k, v in inconsistent.items()},
        "sources_missing": missing_sources,
        "required_arms": list(REQUIRED_ARMS),
        "scale_consistency_at_budget_ge_1000": {
            "n": len(scale_values),
            "median": float(np.median(scale_values)) if scale_values else None,
            "p95": float(np.percentile(scale_values, 95)) if scale_values else None,
            "note": "|M_rel_codebook - 1/procrustes_scale| / M_rel_codebook; "
                    "a small value confirms the harvest read the right fields",
        },
        "inputs": inputs,
    }
    atomic_json(out_dir / "harvest_manifest.json", manifest)

    print(json.dumps({
        "n_sources": len(per_source_targets),
        "n_files": len(selected),
        "n_rows": len(rows),
        "sources_missing": missing_sources,
        "sources_with_unexpected_repeat_count": {str(k): v for k, v in inconsistent.items()},
        "scale_consistency_median": manifest["scale_consistency_at_budget_ge_1000"]["median"],
        "outputs": [str(out_dir / "interface_rows.csv"),
                    str(out_dir / "harvest_manifest.json"),
                    str(out_dir / "candidate_scan.json")],
    }, indent=2))

    if args.require_complete and (missing_sources or inconsistent):
        print("ERROR: incomplete harvest under --require-complete", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
