from __future__ import annotations
import csv, hashlib, json, math, os, re, tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import yaml


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(clean(value), sort_keys=True,
                         separators=(",", ":")).encode()).hexdigest()


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(clean(value), f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def read_yaml(path: str | Path) -> dict:
    obj = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict): raise ValueError(f"Expected YAML mapping: {path}")
    return obj


def write_yaml(path: str | Path, obj: dict) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(clean(obj), sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_csv(path: str | Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fields, extrasaction="ignore"); w.writeheader()
        w.writerows(clean(rows))


def project_root(path: str | None = None) -> Path:
    root = Path(path or os.environ.get("PROJECT_DIR", os.getcwd())).resolve()
    if not (root / "gpm_cmap/configs/nav_pretrain_full.yaml").is_file():
        raise FileNotFoundError(f"Not a project root (pyproject.toml is NOT required): {root}")
    return root


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"Unsafe run/stage name: {value!r}")
    return value


def rel(path: Path, root: Path) -> str:
    return str(path.absolute().relative_to(root))


def original_config(root: Path, name: str) -> dict:
    return read_yaml(root / "gpm_cmap/configs" / (name + ".yaml"))


def tree_hash(root: Path) -> dict:
    files = {}
    for folder in ("gpm_cmap", "multinav"):
        for p in sorted((root / folder).rglob("*")):
            if p.is_file() and p.suffix in {".py", ".yaml"} and "__pycache__" not in p.parts:
                files[str(p.relative_to(root))] = sha(p)
    return {"sha256": digest(files), "files": files}


def verify_tree(root: Path, expected: dict) -> None:
    for name, h in expected["files"].items():
        p = root / name
        if not p.is_file() or sha(p) != h:
            raise RuntimeError(f"Frozen source/config changed: {p}")
