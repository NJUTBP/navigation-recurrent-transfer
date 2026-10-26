#!/usr/bin/env python3
"""Standalone runner for the current 10-source Schøyen-style topology cohort.

This script is intentionally separate from the frozen multinav plan. It reads
navigation best.pt checkpoints from the original run root and writes new topology
artifacts into the current best-validation result root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

SEEDS = [6101, 6102, 6103, 6104, 6105, 6106, 6107, 6108, 6109, 6110]
REQUIRED_OUTPUTS = [
    "torus_analysis_summary.json",
    "selected_ensemble.json",
    "torus_analysis_arrays.npz",
    "cluster_summary.csv",
    "candidate_topology.csv",
    "run_provenance.json",
]
SCRIPT_VERSION = "schoyen_topology_best_validation_cohort_v2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def checkpoint_loader(code_root: Path):
    code = str(code_root.resolve())
    if code not in sys.path:
        sys.path.insert(0, code)
    from gpm_cmap.utils.checkpointing import load_checkpoint  # type: ignore
    return load_checkpoint


def source_paths(run_root: Path, result_root: Path, seed: int) -> Dict[str, Path]:
    old = run_root / "sources" / f"nav_seed_{seed}"
    new = result_root / "sources" / f"nav_seed_{seed}"
    nav = old / "artifacts" / "navigation"
    return {
        "old_source": old,
        "new_source": new,
        "checkpoint": nav / "best.pt",
        "history": nav / "history.json",
        "config": nav / "config_used.yaml",
        "artifact": new / "artifacts" / "torus_topology",
        "state": new / "state" / "torus_topology.json",
        "qualification": new / "topology_qualification.json",
        "attempts": new / "attempts" / "torus_topology",
    }


def validate_best_pt(run_root: Path, result_root: Path, code_root: Path, seed: int) -> Dict[str, Any]:
    p = source_paths(run_root, result_root, seed)
    missing = [str(p[k]) for k in ("checkpoint", "history", "config") if not p[k].is_file()]
    if missing:
        raise FileNotFoundError("Missing topology input(s):\n  " + "\n  ".join(missing))
    load_checkpoint = checkpoint_loader(code_root)
    payload = load_checkpoint(p["checkpoint"], map_location="cpu")
    epoch = int(payload.get("epoch", -1))
    history = read_json(p["history"])
    rows = [r for r in history if isinstance(r, dict) and r.get("validation_task_loss") is not None]
    if not rows:
        raise ValueError(f"No validation_task_loss rows: {p['history']}")
    best = min(rows, key=lambda r: float(r["validation_task_loss"]))
    best_epoch = int(best["epoch"])
    if epoch != best_epoch:
        raise RuntimeError(
            f"seed {seed}: navigation/best.pt epoch {epoch} != minimum validation-loss epoch {best_epoch}"
        )
    return {
        "seed": seed,
        "checkpoint": str(p["checkpoint"]),
        "checkpoint_sha256": sha256(p["checkpoint"]),
        "checkpoint_epoch": epoch,
        "validation_task_loss": float(best["validation_task_loss"]),
        "config": str(p["config"]),
    }


def check_code(code_root: Path) -> Dict[str, Any]:
    analyzer = code_root / "gpm_cmap" / "experiments" / "analyze_schoyen_torus.py"
    aggregator = code_root / "gpm_cmap" / "experiments" / "aggregate_schoyen_torus.py"
    cfg = code_root / "multinav" / "config.yaml"
    missing = [str(p) for p in (analyzer, aggregator, cfg) if not p.is_file()]
    if missing:
        raise FileNotFoundError("Updated topology code not installed:\n  " + "\n  ".join(missing))
    text = cfg.read_text(encoding="utf-8")
    if "checkpoint_policy: best_validation" not in text:
        raise RuntimeError(
            f"Current code config is not best_validation policy: {cfg}. "
            "Install the supplied code patch first."
        )
    return {
        "analyzer": str(analyzer),
        "analyzer_sha256": sha256(analyzer),
        "aggregator": str(aggregator),
        "aggregator_sha256": sha256(aggregator),
        "config": str(cfg),
    }


def preflight(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    result_root = Path(args.result_root).resolve()
    code_root = Path(args.code_root).resolve()
    code = check_code(code_root)
    deps = {}
    for name in ["torch", "numpy", "scipy", "sklearn", "umap", "ripser"]:
        module = importlib.import_module(name)
        deps[name] = getattr(module, "__version__", "unknown")
    sources = []
    for seed in SEEDS:
        z = validate_best_pt(run_root, result_root, code_root, seed)
        sources.append(z)
        print(
            f"[preflight ok] source={seed} best.pt epoch={z['checkpoint_epoch']} "
            f"val_loss={z['validation_task_loss']:.8g}"
        )
    manifest = {
        "schema": "schoyen_topology_preflight_v2",
        "script_version": SCRIPT_VERSION,
        "run_root": str(run_root),
        "result_root": str(result_root),
        "code_root": str(code_root),
        "code": code,
        "dependencies": deps,
        "sources": sources,
        "checked_utc": now(),
    }
    write_json(result_root / "topology_preflight.json", manifest)
    print("All 10 navigation best.pt checkpoints passed preflight.")
    return 0


def next_attempt(attempt_root: Path) -> Path:
    attempt_root.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in attempt_root.glob("try_*"):
        try:
            nums.append(int(p.name.split("_")[-1]))
        except ValueError:
            pass
    return attempt_root / f"try_{(max(nums) + 1 if nums else 1):03d}"


def commit_symlink(public: Path, target: Path) -> None:
    public.parent.mkdir(parents=True, exist_ok=True)
    tmp = public.parent / ("." + public.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        if tmp.is_dir() and not tmp.is_symlink():
            shutil.rmtree(tmp)
        else:
            tmp.unlink()
    if public.exists() or public.is_symlink():
        if public.is_symlink():
            public.unlink()
        elif public.is_dir():
            backup = public.parent / f"{public.name}.pre_schoyen_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            public.rename(backup)
        else:
            public.unlink()
    tmp.symlink_to(os.path.relpath(target, public.parent), target_is_directory=True)
    os.replace(tmp, public)


def run_source(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    result_root = Path(args.result_root).resolve()
    code_root = Path(args.code_root).resolve()
    seed = int(args.seed)
    if seed not in SEEDS:
        raise ValueError(f"Unexpected source seed: {seed}")
    code = check_code(code_root)
    inp = validate_best_pt(run_root, result_root, code_root, seed)
    p = source_paths(run_root, result_root, seed)
    fingerprint_raw = json.dumps(
        {
            "script_version": SCRIPT_VERSION,
            "source_seed": seed,
            "checkpoint_sha256": inp["checkpoint_sha256"],
            "config_sha256": sha256(p["config"]),
            "analyzer_sha256": code["analyzer_sha256"],
        },
        sort_keys=True,
    ).encode()
    fingerprint = hashlib.sha256(fingerprint_raw).hexdigest()
    state = read_json(p["state"]) if p["state"].is_file() else None
    if (
        not args.force
        and isinstance(state, dict)
        and state.get("status") == "success"
        and state.get("fingerprint") == fingerprint
        and p["artifact"].exists()
        and all((p["artifact"] / f).exists() for f in REQUIRED_OUTPUTS)
    ):
        print(f"[reuse] seed={seed} artifact={p['artifact']}")
        return 0

    attempt = next_attempt(p["attempts"])
    attempt.mkdir(parents=True, exist_ok=False)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "gpm_cmap.experiments.analyze_schoyen_torus",
        "--config",
        str(p["config"]),
        "--checkpoint",
        str(p["checkpoint"]),
        "--output_dir",
        str(attempt),
        "--seed",
        str(seed),
        "--device",
        args.device,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["MPLBACKEND"] = "Agg"
    env["PYTHONHASHSEED"] = str(seed)
    start_state = {
        "stage": "torus_topology",
        "source_seed": seed,
        "status": "running",
        "fingerprint": fingerprint,
        "checkpoint": inp,
        "attempt": str(attempt),
        "started_utc": now(),
    }
    write_json(p["state"], start_state)
    print("[run]", " ".join(cmd), flush=True)
    try:
        subprocess.run(cmd, cwd=str(code_root), env=env, check=True)
        missing = [f for f in REQUIRED_OUTPUTS if not (attempt / f).exists()]
        if missing:
            raise RuntimeError("Topology analysis completed without required outputs: " + ", ".join(missing))
        summary = read_json(attempt / "torus_analysis_summary.json")
        if summary.get("checkpoint_role") != "best_validation":
            raise RuntimeError("Topology summary does not declare checkpoint_role=best_validation")
        commit_symlink(p["artifact"], attempt)
        persistent = (summary.get("persistent_cohomology") or {}).get("metrics") or {}
        selected = summary.get("selection") or {}
        qualification = {
            "source_seed": seed,
            "analysis_completed": True,
            "checkpoint_role": "best_validation",
            "checkpoint": str(p["checkpoint"]),
            "checkpoint_sha256": inp["checkpoint_sha256"],
            "checkpoint_epoch": inp["checkpoint_epoch"],
            "is_torus_like": summary.get("is_torus_like"),
            "status": summary.get("status"),
            "selected_clusters": selected.get("selected_clusters"),
            "n_selected_units": selected.get("n_selected_units"),
            "robust_min_gap_score": selected.get("robust_min_gap_score"),
            "euclidean": persistent.get("euclidean"),
            "cosine": persistent.get("cosine"),
            "usable_for_source_v3": False,
            "legacy_source_v3_status": "retired_not_evaluable",
            "note": "Automatic Schoyen-style torus discovery on validation best.pt; legacy cohomological phase-coordinate contract is not produced.",
        }
        write_json(p["qualification"], qualification)
        done_state = dict(start_state)
        done_state.update(
            status="success",
            artifact=str(p["artifact"]),
            ended_utc=now(),
            is_torus_like=summary.get("is_torus_like"),
            robust_min_gap_score=selected.get("robust_min_gap_score"),
        )
        write_json(p["state"], done_state)
        print(
            f"[success] seed={seed} torus_like={summary.get('is_torus_like')} "
            f"robust_min_gap={selected.get('robust_min_gap_score')}"
        )
        return 0
    except Exception as exc:
        failed = dict(start_state)
        failed.update(status="failed", ended_utc=now(), error=repr(exc))
        write_json(p["state"], failed)
        raise


def aggregate(args: argparse.Namespace) -> int:
    result_root = Path(args.result_root).resolve()
    code_root = Path(args.code_root).resolve()
    check_code(code_root)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(code_root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["MPLBACKEND"] = "Agg"
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "gpm_cmap.experiments.aggregate_schoyen_torus",
        "--run_root",
        str(result_root),
    ]
    print("[aggregate]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(code_root), env=env, check=True)
    return 0


def postflight(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    result_root = Path(args.result_root).resolve()
    rows: List[Dict[str, Any]] = []
    bad = []
    for seed in SEEDS:
        p = source_paths(run_root, result_root, seed)
        summary_path = p["artifact"] / "torus_analysis_summary.json"
        state_path = p["state"]
        if not summary_path.is_file():
            rows.append({"seed": seed, "status": "missing"})
            bad.append(seed)
            continue
        summary = read_json(summary_path)
        state = read_json(state_path) if state_path.is_file() else {}
        sel = summary.get("selection") or {}
        rows.append(
            {
                "seed": seed,
                "status": state.get("status", summary.get("status")),
                "is_torus_like": summary.get("is_torus_like"),
                "selected_clusters": sel.get("selected_clusters"),
                "n_selected_units": sel.get("n_selected_units"),
                "robust_min_gap_score": sel.get("robust_min_gap_score"),
            }
        )
        if state.get("status") != "success":
            bad.append(seed)
    sanity_6101 = None
    p6101 = source_paths(run_root, result_root, 6101)
    cand_path = p6101["artifact"] / "candidate_topology.csv"
    summary_path = p6101["artifact"] / "torus_analysis_summary.json"
    if cand_path.is_file() and summary_path.is_file():
        passed = set()
        with cand_path.open("r", encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                flag = str(r.get("passes_robust_121_signature", "")).strip().lower()
                if flag in {"true", "1", "yes"}:
                    passed.add(str(r.get("clusters", "")).strip())
        s6101 = read_json(summary_path)
        selected = sorted(int(x) for x in ((s6101.get("selection") or {}).get("selected_clusters") or []))
        sanity_6101 = {
            "cluster_4_passed": "4" in passed,
            "cluster_0_plus_4_passed": ("0_4" in passed or "4_0" in passed),
            "selected_clusters": selected,
            "selected_0_plus_4": selected == [0, 4],
            "note": "Diagnostic only; not a hard cohort gate.",
        }
    report = {
        "schema": "schoyen_topology_postflight_v2",
        "script_version": SCRIPT_VERSION,
        "n_expected": len(SEEDS),
        "n_success": len(SEEDS) - len(set(bad)),
        "failed_or_missing": sorted(set(bad)),
        "sanity_6101": sanity_6101,
        "sources": rows,
        "checked_utc": now(),
    }
    out = result_root / "aggregate" / "torus_auto" / "coverage.json"
    write_json(out, report)
    print(json.dumps(report, indent=2))
    return 2 if bad else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--run-root", required=True)
        p.add_argument("--result-root", required=True)
        p.add_argument("--code-root", required=True)

    p = sub.add_parser("preflight")
    common(p)
    p.set_defaults(func=preflight)

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=run_source)

    p = sub.add_parser("aggregate")
    p.add_argument("--result-root", required=True)
    p.add_argument("--code-root", required=True)
    p.set_defaults(func=aggregate)

    p = sub.add_parser("postflight")
    common(p)
    p.set_defaults(func=postflight)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
