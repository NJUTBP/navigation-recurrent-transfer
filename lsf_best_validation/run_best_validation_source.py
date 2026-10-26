#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run downstream multinav cognitive experiments from existing navigation best.pt files.

This launcher is intentionally external to the frozen scientific code. It does not
edit code/, sources/, plan.json, or the original artifacts. It only:
  * reuses each source's existing navigation best.pt (minimum navigation validation loss);
  * reuses the existing sequence-control best_validation.pt for CTL arms;
  * regenerates downstream configs from the frozen code/configs;
  * writes all new outputs under a separate rerun root;
  * skips topology_export, formal_topology, source_splits, source_reduced,
    action_causality, and path_invariance.

Families:
  preflight          validate inputs and checkpoint identity
  main               Exp5 NAV + CTL + mapping + test-time transforms
  adapter_symbolic   Exp7 adapter rescue + Exp8 symbolic adapter
  hard               hard abstract path-integration benchmark
  factorial          recurrent/input-origin factorial experiment
  all                run all four cognitive families sequentially
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

SCRIPT_VERSION = "best_validation_downstream_v1"
SKIPPED_TOPOLOGY_CHAIN = [
    "topology_export",
    "formal_topology",
    "source_splits",
    "source_reduced",
    "action_causality",
    "path_invariance",
]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    return value


def digest(value: Any) -> str:
    raw = json.dumps(clean(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(clean(obj), f, ensure_ascii=False, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_yaml(path: Path) -> Dict[str, Any]:
    obj = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return obj


def write_yaml(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(clean(obj), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def safe_stage(name: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    if not name or any(ch not in allowed for ch in name):
        raise ValueError(f"Unsafe stage name: {name!r}")
    return name


class Launcher:
    def __init__(
        self,
        run_root: Path,
        output_root: Path,
        source_seed: int,
        python_exe: Path,
        device: str,
        threads: int,
    ):
        self.run_root = run_root.resolve()
        self.output_root = output_root.resolve()
        self.source_seed = int(source_seed)
        self.python_exe = python_exe.resolve()
        self.device = device
        self.threads = int(threads)

        self.plan_path = self.run_root / "plan.json"
        self.code_dir = self.run_root / "code"
        self.bootstrap = self.code_dir / "multinav" / "bootstrap.py"
        self.old_source = self.run_root / "sources" / f"nav_seed_{self.source_seed}"
        self.new_source = self.output_root / "sources" / f"nav_seed_{self.source_seed}"

        self._validate_layout()

        self.plan = read_json(self.plan_path)
        self.pair = next(
            (p for p in self.plan.get("source_pairs", []) if int(p["source_seed"]) == self.source_seed),
            None,
        )
        if self.pair is None:
            raise ValueError(f"Source seed {self.source_seed} is not present in {self.plan_path}")

        self.nav_checkpoint = self.old_source / "artifacts" / "navigation" / "best.pt"
        self.nav_history = self.old_source / "artifacts" / "navigation" / "history.json"
        self.ctl_checkpoint = (
            self.old_source / "artifacts" / "sequence_control" / "best_validation.pt"
        )
        self.ctl_gate = self.old_source / "artifacts" / "sequence_control" / "training_gate.json"

        self._load_frozen_code()
        self.input_manifest = self.preflight(write_manifest=False)
        self.identity_path = self.new_source / "source_identity.json"
        self._write_identity()

    def _validate_layout(self) -> None:
        required = [
            self.plan_path,
            self.code_dir / "gpm_cmap" / "configs" / "abstract_pathint.yaml",
            self.bootstrap,
            self.old_source,
            self.python_exe,
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Required input(s) missing:\n  " + "\n  ".join(str(p) for p in missing)
            )

    def _load_frozen_code(self) -> None:
        code_str = str(self.code_dir)
        if code_str not in sys.path:
            sys.path.insert(0, code_str)
        from gpm_cmap.utils.checkpointing import load_checkpoint  # type: ignore

        self.load_checkpoint = load_checkpoint

    def config(self, name: str, seed: int) -> Dict[str, Any]:
        path = self.code_dir / "gpm_cmap" / "configs" / f"{name}.yaml"
        cfg = read_yaml(path)
        cfg["seed"] = int(seed)
        if "n_seeds" in cfg:
            cfg["n_seeds"] = 1
        cfg["device"] = self.device
        cfg.pop("output_dir", None)

        runtime = cfg.setdefault("runtime", {})
        if not isinstance(runtime, dict):
            cfg["runtime"] = runtime = {}
        runtime["cpu_num_threads"] = self.threads
        runtime["allow_device_fallback"] = False

        data = cfg.get("data")
        if isinstance(data, dict):
            data["num_workers"] = 0

        return cfg

    def preflight(self, *, write_manifest: bool = True) -> Dict[str, Any]:
        required = [
            self.nav_checkpoint,
            self.nav_history,
            self.ctl_checkpoint,
        ]
        missing = [p for p in required if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "Existing source checkpoints are required; navigation/control retraining is disabled.\n"
                "Missing:\n  " + "\n  ".join(str(p) for p in missing)
            )

        nav = self.load_checkpoint(self.nav_checkpoint, map_location="cpu")
        ctl = self.load_checkpoint(self.ctl_checkpoint, map_location="cpu")

        nav_state = nav.get("model_state", {})
        ctl_state = ctl.get("model_state", {})
        needed = ("grid_rnn.cell.in_proj.weight", "grid_rnn.cell.h_proj.weight")
        for key in needed:
            if key not in nav_state:
                raise KeyError(f"Navigation checkpoint lacks {key}: {self.nav_checkpoint}")
            if key not in ctl_state:
                raise KeyError(f"Control checkpoint lacks {key}: {self.ctl_checkpoint}")
            if tuple(nav_state[key].shape) != tuple(ctl_state[key].shape):
                raise RuntimeError(
                    f"NAV/CTL tensor shape mismatch for {key}: "
                    f"{tuple(nav_state[key].shape)} vs {tuple(ctl_state[key].shape)}"
                )

        history = read_json(self.nav_history)
        if not isinstance(history, list) or not history:
            raise ValueError(f"Invalid navigation history: {self.nav_history}")
        val_rows = [
            row
            for row in history
            if isinstance(row, dict) and row.get("validation_task_loss") is not None
        ]
        if not val_rows:
            raise ValueError("Navigation history has no validation_task_loss rows")
        best_row = min(val_rows, key=lambda row: float(row["validation_task_loss"]))
        best_history_epoch = int(best_row["epoch"])
        checkpoint_epoch = int(nav.get("epoch", -1))
        if checkpoint_epoch != best_history_epoch:
            raise RuntimeError(
                "best.pt does not match the minimum navigation validation-loss epoch: "
                f"checkpoint epoch={checkpoint_epoch}, history minimum epoch={best_history_epoch}"
            )

        gate = read_json(self.ctl_gate) if self.ctl_gate.is_file() else None
        manifest = {
            "schema": "best_validation_downstream_preflight_v1",
            "script_version": SCRIPT_VERSION,
            "source_seed": self.source_seed,
            "control_seed": int(self.pair["control_seed"]),
            "main_target_seeds": self.pair["main_target_seeds"],
            "hard_target_seeds": self.pair["hard_target_seeds"],
            "factorial_target_seeds": self.pair["factorial_target_seeds"],
            "navigation_checkpoint": str(self.nav_checkpoint),
            "navigation_checkpoint_sha256": sha256(self.nav_checkpoint),
            "navigation_checkpoint_epoch": checkpoint_epoch,
            "navigation_validation_task_loss": float(best_row["validation_task_loss"]),
            "history_minimum_epoch": best_history_epoch,
            "sequence_control_checkpoint": str(self.ctl_checkpoint),
            "sequence_control_checkpoint_sha256": sha256(self.ctl_checkpoint),
            "sequence_control_checkpoint_epoch": ctl.get("epoch"),
            "sequence_control_training_gate": gate,
            "tensor_shapes": {
                key: list(nav_state[key].shape) for key in needed
            },
            "topology_chain_skipped": SKIPPED_TOPOLOGY_CHAIN,
            "checked_utc": now(),
        }
        if write_manifest:
            write_json(self.new_source / "preflight.json", manifest)
            print(f"[preflight ok] source={self.source_seed} nav_epoch={checkpoint_epoch}")
        return manifest

    def _write_identity(self) -> None:
        nav = self.load_checkpoint(self.nav_checkpoint, map_location="cpu")
        h = hashlib.sha256()
        for key in ("grid_rnn.cell.in_proj.weight", "grid_rnn.cell.h_proj.weight"):
            tensor = nav["model_state"][key].detach().cpu().contiguous().numpy()
            h.update(key.encode("utf-8"))
            h.update(str(tensor.shape).encode("utf-8"))
            h.update(tensor.tobytes())

        history = read_json(self.nav_history)
        identity = {
            "schema": "multinav_source_identity_best_validation_reuse_v1",
            "source_seed": self.source_seed,
            "navigation_checkpoint": str(self.nav_checkpoint),
            "navigation_checkpoint_sha256": sha256(self.nav_checkpoint),
            "core_sha256": h.hexdigest(),
            "checkpoint_epoch": nav.get("epoch"),
            "checkpoint_policy": "best_validation",
            "checkpoint_rule": "minimum navigation validation task loss",
            "reused_existing_navigation_training": True,
            "completed_epochs": len(history),
            "source_reference_kind": "navigation_checkpoint_only",
            "original_plan_sha256": self.plan.get("plan_sha256"),
            "topology_chain_skipped": SKIPPED_TOPOLOGY_CHAIN,
            "created_utc": now(),
        }
        if self.identity_path.exists():
            old = read_json(self.identity_path)
            for key in (
                "source_seed",
                "navigation_checkpoint_sha256",
                "core_sha256",
                "checkpoint_epoch",
                "checkpoint_policy",
            ):
                if old.get(key) != identity.get(key):
                    raise RuntimeError(
                        f"Existing rerun source_identity conflicts on {key}: "
                        f"{old.get(key)!r} != {identity.get(key)!r}"
                    )
            return
        write_json(self.identity_path, identity)

    def _stage_fingerprint(
        self,
        kind: str,
        cfg: Dict[str, Any],
        input_paths: Iterable[Path],
    ) -> str:
        inputs: Dict[str, str] = {}
        for p in input_paths:
            p = Path(p)
            if p.is_file():
                inputs[str(p.resolve())] = sha256(p)
        payload = {
            "script_version": SCRIPT_VERSION,
            "kind": kind,
            "config": cfg,
            "inputs": inputs,
            "code_manifest_sha256": self.plan.get("code_manifest", {}).get("sha256"),
        }
        return digest(payload)

    def run_stage(
        self,
        name: str,
        kind: str,
        cfg: Dict[str, Any],
        required: List[str],
        *,
        input_paths: Iterable[Path] = (),
    ) -> Path:
        safe_stage(name)
        self.new_source.mkdir(parents=True, exist_ok=True)
        configs = self.new_source / "configs"
        logs = self.new_source / "logs"
        states = self.new_source / "state"
        attempts = self.new_source / "attempts" / name
        artifacts = self.new_source / "artifacts"
        cache = self.new_source / "cell_cache" / name
        for d in (configs, logs, states, attempts, artifacts, cache):
            d.mkdir(parents=True, exist_ok=True)

        public = artifacts / name
        cfg_base = copy.deepcopy(cfg)
        cfg_base["multinav_cell_cache"] = str(cache.resolve())

        fp = self._stage_fingerprint(
            kind,
            cfg_base,
            [self.nav_checkpoint, self.ctl_checkpoint, self.identity_path, *input_paths],
        )
        state_path = states / f"{name}.json"
        old = read_json(state_path) if state_path.exists() else {}

        if old.get("status") == "done":
            if old.get("fingerprint") != fp:
                raise RuntimeError(
                    f"Completed stage {name} has a different fingerprint. "
                    "Use a new OUTPUT_ROOT instead of mixing protocols."
                )
            if not public.exists():
                raise RuntimeError(f"Completed public artifact missing: {public}")
            for pattern in required:
                found = list(public.glob(pattern))
                if not found or any((not p.is_file()) or p.stat().st_size == 0 for p in found):
                    raise RuntimeError(f"Completed stage output is incomplete: {name}/{pattern}")
            print(f"[reuse] {name}", flush=True)
            return public

        tries = sorted(p for p in attempts.glob("try_*") if p.is_dir())
        number = 1
        if tries:
            number = max(int(p.name.split("_")[-1]) for p in tries) + 1
        out = attempts / f"try_{number:03d}"
        out.mkdir(parents=True, exist_ok=False)

        cfg_run = copy.deepcopy(cfg_base)
        cfg_run["output_dir"] = str(out.resolve())
        cfg_run.setdefault("_best_validation_rerun", {})
        cfg_run["_best_validation_rerun"].update(
            {
                "script_version": SCRIPT_VERSION,
                "source_seed": self.source_seed,
                "checkpoint_policy": "best_validation",
                "navigation_checkpoint": str(self.nav_checkpoint),
                "navigation_checkpoint_sha256": sha256(self.nav_checkpoint),
                "topology_chain_skipped": SKIPPED_TOPOLOGY_CHAIN,
            }
        )

        cfg_path = configs / f"{name}_{number:03d}.yaml"
        log_path = logs / f"{name}_{number:03d}.log"
        write_yaml(cfg_path, cfg_run)

        cmd = [
            str(self.python_exe),
            "-u",
            str(self.bootstrap),
            "multinav.stage",
            kind,
            "--config",
            str(cfg_path.resolve()),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.code_dir) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONHASHSEED"] = str(cfg_run.get("seed", self.source_seed))
        env["MPLBACKEND"] = "Agg"
        env["OMP_NUM_THREADS"] = str(self.threads)
        env["MKL_NUM_THREADS"] = str(self.threads)
        env["OPENBLAS_NUM_THREADS"] = str(self.threads)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

        state = {
            "stage": name,
            "kind": kind,
            "status": "running",
            "fingerprint": fp,
            "attempt": str(out),
            "config": str(cfg_path),
            "log": str(log_path),
            "command": cmd,
            "started_utc": now(),
        }
        write_json(state_path, state)
        print(f"[run] {name} -> {log_path}", flush=True)

        try:
            with log_path.open("w", encoding="utf-8") as stream:
                stream.write("COMMAND:\n" + " ".join(cmd) + "\n\n")
                stream.flush()
                subprocess.run(
                    cmd,
                    cwd=str(self.run_root),
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )

            outputs: Dict[str, str] = {}
            for pattern in required:
                found = list(out.glob(pattern))
                if not found:
                    raise RuntimeError(f"Missing required artifact: {name}/{pattern}")
                for p in found:
                    if not p.is_file() or p.stat().st_size == 0:
                        raise RuntimeError(f"Empty required artifact: {p}")
                    outputs[str(p)] = sha256(p)

            tmp = public.with_name(public.name + ".new")
            if tmp.is_symlink() or tmp.exists():
                if tmp.is_dir() and not tmp.is_symlink():
                    shutil.rmtree(tmp)
                else:
                    tmp.unlink()
            tmp.symlink_to(os.path.relpath(out, public.parent))
            os.replace(tmp, public)

            state.update(
                status="done",
                ended_utc=now(),
                outputs=outputs,
                public_artifact=str(public),
            )
            write_json(state_path, state)
            return public
        except Exception as exc:
            state.update(status="failed", ended_utc=now(), error=repr(exc))
            write_json(state_path, state)
            print(f"[FAILED] {name}: {exc}", flush=True)
            raise

    def common_exp5(self, target: int) -> Dict[str, Any]:
        cfg = self.config("abstract_pathint", target)
        ckpt = str(self.nav_checkpoint)
        cfg["transfer_checkpoint"] = ckpt
        cfg["transfer"] = {"mode": "full", "checkpoint_candidates": [ckpt]}
        controls = cfg.setdefault("controls", {})
        if isinstance(controls, dict):
            for key in list(controls):
                if str(key).startswith("run_"):
                    controls[key] = False
            controls["action_map_mode"] = "aligned"
        return cfg

    def run_main(self) -> None:
        for target in self.pair["main_target_seeds"]:
            target = int(target)
            common = self.common_exp5(target)
            nav_name = f"exp5_nav_t{target}"
            nav_out = self.run_stage(
                nav_name,
                "exp5",
                common,
                ["raw_per_seed.json", "seed_*/*/history.json", "seed_*/*/*_model.pt"],
                input_paths=[self.nav_checkpoint],
            )

            ctl_cfg = copy.deepcopy(common)
            ctl_ckpt = str(self.ctl_checkpoint)
            ctl_cfg["transfer_checkpoint"] = ctl_ckpt
            ctl_cfg["transfer"] = {"mode": "full", "checkpoint_candidates": [ctl_ckpt]}
            self.run_stage(
                f"exp5_ctl_t{target}",
                "exp5_ctl",
                ctl_cfg,
                ["raw_per_seed.json", "transfer_source_manifest.json"],
                input_paths=[self.ctl_checkpoint],
            )

            self.run_stage(
                f"mapping_t{target}",
                "mapping",
                common,
                ["raw_per_seed.json"],
                input_paths=[self.nav_checkpoint],
            )

            vt = copy.deepcopy(common)
            vt["multinav_main_output"] = str(nav_out.resolve())
            self.run_stage(
                f"test_transforms_t{target}",
                "test_transforms",
                vt,
                ["test_transform_rows.json"],
                input_paths=[nav_out / "raw_per_seed.json"],
            )

    def run_adapter_symbolic(self) -> None:
        for target in self.pair["main_target_seeds"]:
            target = int(target)

            adapter = self.config("adapter_rescue", target)
            adapter["nav_checkpoints"] = [str(self.nav_checkpoint)]
            self.run_stage(
                f"adapter_t{target}",
                "adapter",
                adapter,
                ["adapter_rescue_rows.json", "adapter_rescue_rows.csv"],
                input_paths=[self.nav_checkpoint],
            )

            symbolic = self.config("symbolic_adapter_ctl01", target)
            symbolic["pretrained_sources"] = [
                {
                    "arm": "nav_frozen",
                    "checkpoint": str(self.nav_checkpoint),
                    "source_type": "navigation",
                    "require_gate": False,
                },
                {
                    "arm": "ctl01_frozen",
                    "checkpoint": str(self.ctl_checkpoint),
                    "source_type": "sequence_control",
                    "require_gate": False,
                },
            ]
            self.run_stage(
                f"symbolic_t{target}",
                "symbolic",
                symbolic,
                [
                    "symbolic_adapter_rows.csv",
                    "symbolic_adapter_rows.json",
                    "protocol_manifest.json",
                ],
                input_paths=[self.nav_checkpoint, self.ctl_checkpoint],
            )

    def run_hard(self) -> None:
        for target in self.pair["hard_target_seeds"]:
            target = int(target)
            cfg = self.config("abstract_pathint_hard", target)
            cfg["navigation_checkpoint"] = str(self.nav_checkpoint)
            cfg["source_reduced_artifact"] = str(self.identity_path)
            cfg["confirmatory"]["target_seeds"] = [target]
            self.run_stage(
                f"hard_t{target}",
                "hard",
                cfg,
                ["hard_learning_curves.csv", "checkpoint_manifest.json"],
                input_paths=[self.nav_checkpoint, self.identity_path],
            )

    def run_factorial(self) -> None:
        contrasts = self.plan.get("factorial_contrasts")
        if not contrasts:
            raise ValueError("Frozen plan has no factorial_contrasts")

        for target in self.pair["factorial_target_seeds"]:
            target = int(target)
            cfg = self.config("recurrent_factorial", target)
            cfg["navigation_checkpoint"] = str(self.nav_checkpoint)
            cfg["source_reduced_artifact"] = str(self.identity_path)
            cfg["confirmatory"]["target_seeds"] = [target]
            cfg.setdefault("statistics", {})["confirmatory_comparisons"] = contrasts
            self.run_stage(
                f"factorial_t{target}",
                "factorial",
                cfg,
                [
                    "factorial_learning_curves.csv",
                    "factorial_per_seed.csv",
                    "checkpoint_manifest.json",
                ],
                input_paths=[self.nav_checkpoint, self.identity_path],
            )

    def write_status(self, families: List[str]) -> None:
        states = []
        state_dir = self.new_source / "state"
        if state_dir.exists():
            for p in sorted(state_dir.glob("*.json")):
                states.append(read_json(p))
        failures = [x.get("stage") for x in states if x.get("status") == "failed"]
        tag = "_".join(families) if families else "unknown"
        write_json(
            self.new_source / f"status_{tag}.json",
            {
                "schema": "best_validation_downstream_source_status_v1",
                "script_version": SCRIPT_VERSION,
                "source_seed": self.source_seed,
                "control_seed": int(self.pair["control_seed"]),
                "checkpoint_policy": "best_validation",
                "reused_existing_navigation_training": True,
                "reused_existing_sequence_control_training": True,
                "requested_families": families,
                "topology_chain_skipped": SKIPPED_TOPOLOGY_CHAIN,
                "status": "failed" if failures else "finished_requested_families",
                "failed_stages": failures,
                "states": states,
                "ended_utc": now(),
            },
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", required=True, type=Path,
                    help="Existing multinav_paper_v1 run root containing plan.json, code/, sources/.")
    ap.add_argument("--output-root", required=True, type=Path,
                    help="New output root; original sources/ are never modified.")
    ap.add_argument("--source-seed", required=True, type=int)
    ap.add_argument(
        "--family",
        required=True,
        choices=["preflight", "main", "adapter_symbolic", "hard", "factorial", "all"],
    )
    ap.add_argument("--python", dest="python_exe", required=True, type=Path)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    launcher = Launcher(
        run_root=args.run_root,
        output_root=args.output_root,
        source_seed=args.source_seed,
        python_exe=args.python_exe,
        device=args.device,
        threads=args.threads,
    )

    if args.family == "preflight":
        launcher.preflight(write_manifest=True)
        launcher.write_status(["preflight"])
        return 0

    families = (
        ["main", "adapter_symbolic", "hard", "factorial"]
        if args.family == "all"
        else [args.family]
    )
    for family in families:
        if family == "main":
            launcher.run_main()
        elif family == "adapter_symbolic":
            launcher.run_adapter_symbolic()
        elif family == "hard":
            launcher.run_hard()
        elif family == "factorial":
            launcher.run_factorial()
        else:
            raise AssertionError(family)

    launcher.write_status(families)
    print(
        f"[done] source={launcher.source_seed} family={args.family} "
        f"output={launcher.new_source}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
