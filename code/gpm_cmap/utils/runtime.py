from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict

from .config import save_config
from .json_io import write_json


_SOURCE_SUFFIXES = {
    ".py", ".yaml", ".yml", ".toml", ".md", ".txt", ".ps1", ".json",
}
_SOURCE_FILENAMES = {".gitignore"}
_SOURCE_EXCLUDED_PARTS = {
    ".git", ".github_cache", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "venv", "env", "__pycache__", "archive", "build", "dist",
    "outputs", "results", "checkpoints", "logs",
}


def resolve_device(
    device_name: str | None = None,
    *,
    allow_fallback: bool = False,
) -> "torch.device":
    """Resolve a requested device without silently changing hardware class."""
    import torch

    requested = (device_name or "cpu").lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        if allow_fallback:
            return torch.device("cpu")
        raise RuntimeError(
            f"CUDA device {requested!r} was requested, but CUDA is unavailable. "
            "Set runtime.allow_device_fallback=true only for an intentional CPU fallback."
        )
    return torch.device(requested)


def configure_runtime(cfg: Dict[str, Any]) -> "torch.device":
    """Configure thread counts/determinism and return the resolved device."""
    import torch

    runtime_cfg = cfg.get("runtime", {}) or {}
    device = resolve_device(
        cfg.get("device", "cpu"),
        allow_fallback=bool(runtime_cfg.get("allow_device_fallback", False)),
    )
    cpu_threads = max(1, int(runtime_cfg.get("cpu_num_threads", 1)))

    if device.type == "cpu":
        os.environ.setdefault("OMP_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(cpu_threads))
        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch allows this setting only before inter-op work starts. The
            # requested value is process-global, so a second call is safe.
            pass

    if bool(runtime_cfg.get("deterministic_algorithms", False)):
        torch.use_deterministic_algorithms(True)
    return device


def prepare_output_dir(output_dir: str | Path, *, allow_existing_empty: bool = True) -> Path:
    """Create a result directory without overwriting an existing experiment."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        entries = list(output_dir.iterdir())
        if entries or not allow_existing_empty:
            raise FileExistsError(
                f"Refusing to write into existing non-empty result directory: {output_dir}. "
                "Choose a new --output_dir so prior results are not overwritten."
            )
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def save_config_snapshot(
    cfg: Dict[str, Any], output_dir: str | Path, filename: str = "config_used.yaml"
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / filename)


def compute_source_tree_hash(project_root: str | Path | None = None) -> Dict[str, Any]:
    """Hash the auditable source/configuration tree, excluding run artifacts.

    The digest includes each relative path and its bytes, sorted by path. Historical
    material under ``archive/`` is deliberately excluded so archiving documentation
    does not change the executable-source identity.
    """
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[2]
    root = root.resolve()
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in _SOURCE_EXCLUDED_PARTS for part in rel.parts):
            continue
        if path.name not in _SOURCE_FILENAMES and path.suffix.lower() not in _SOURCE_SUFFIXES:
            continue
        files.append((rel, path))

    digest = hashlib.sha256()
    for rel, path in sorted(files, key=lambda item: item[0].as_posix()):
        digest.update(rel.as_posix().encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return {
        "algorithm": "sha256(relative_path + NUL + bytes + NUL)",
        "sha256": digest.hexdigest(),
        "file_count": len(files),
    }


def save_run_provenance(
    output_dir: str | Path,
    experiment_name: str,
    cfg: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
    filename: str = "run_provenance.json",
) -> Dict[str, Any]:
    """Record code, command, environment and seed for a result directory."""
    import platform
    import subprocess
    import sys
    from datetime import datetime, timezone

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[2]

    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        value = result.stdout.strip()
        return value or None

    cfg = cfg or {}
    runtime_cfg = cfg.get("runtime", {}) or {}
    git_commit = _git("rev-parse", "HEAD")
    git_branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    git_status = _git("status", "--porcelain") if git_commit is not None else None
    source_tree = compute_source_tree_hash(project_root)
    prov: Dict[str, Any] = {
        "experiment_name": experiment_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": None if git_commit is None else bool(git_status),
        "source_tree_sha256": source_tree["sha256"],
        "source_tree_hash_algorithm": source_tree["algorithm"],
        "source_tree_file_count": source_tree["file_count"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "command_argv": list(sys.argv),
        "seed": cfg.get("seed"),
        "device_requested": cfg.get("device", "cpu"),
        "allow_device_fallback": bool(runtime_cfg.get("allow_device_fallback", False)),
        "deterministic_algorithms": bool(runtime_cfg.get("deterministic_algorithms", False)),
        "output_dir": str(output_dir.resolve()),
    }
    try:
        import torch

        prov["torch_version"] = torch.__version__
        prov["cuda_available"] = bool(torch.cuda.is_available())
        prov["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            prov["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError as exc:  # pragma: no cover - torch-free environments
        prov["torch_version"] = f"(not installed: {exc})"
    if extra:
        prov.update(extra)

    write_json(prov, output_dir / filename)
    return prov
