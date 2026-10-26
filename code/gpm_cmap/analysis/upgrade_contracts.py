"""Shared contracts for upgrade stages I2--I4.

The module centralises the formal single-source claim scope, Source Reduced
Model v3 linkage, input hashing, trajectory fingerprints, strict output
manifests, and fail-loud artifact validation.  Downstream experiments never
infer an exact torus claim from topology alone: they must consume the completed
Source v3 artifact and its held-out quality-gate verdict.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from gpm_cmap.utils.checkpointing import sha256_file
from gpm_cmap.utils.json_io import write_json

SCHEMA_FAMILY = "gpm_cmap_upgrade_i2_i4_v1"
CLAIM_SCOPE = "conditional on the analyzed navigation source"
SOURCE_SCHEMA = "source_reduced_v3"
SOURCE_SCHEMA_SCHOYEN = "source_reduced_schoyen_v1"
SOURCE_SCHEMAS = {SOURCE_SCHEMA, SOURCE_SCHEMA_SCHOYEN}
SOURCE_CLASSIFICATIONS = {
    "exact_2d_controlled_torus",
    "toroidal_base_plus_neural_fibre",
    "torus_like_predictive_scaffold",
}


@dataclass(frozen=True)
class SourceReducedArtifact:
    path: Path
    sha256: str
    schema_version: str
    classification: str
    downstream_scope: str
    claim_scope: str
    model: Any
    quality_gates: Mapping[str, Any]
    provenance: Mapping[str, Any]

    @property
    def exact_2d(self) -> bool:
        return self.classification == "exact_2d_controlled_torus"

    @property
    def has_toroidal_base(self) -> bool:
        return self.classification in {
            "exact_2d_controlled_torus",
            "toroidal_base_plus_neural_fibre",
        }

    @property
    def has_low_dimensional_fibre(self) -> bool:
        return self.classification == "toroidal_base_plus_neural_fibre"


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


def load_source_reduced_artifact(
    artifact_path: str | Path,
    *,
    require_completed_test: bool = True,
    require_model: bool = True,
) -> SourceReducedArtifact:
    """Load and strictly validate a completed Source Reduced Model v3 artifact."""
    path = Path(artifact_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Completed Source Reduced Model v3 artifact not found: {path}. "
            "Run analyze_source_reduced_model after applying the numerical hotfix."
        )
    payload = _load_pickle(path)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Source v3 artifact must contain a mapping; got {type(payload).__name__}")
    schema = payload.get("schema_version")
    if schema not in SOURCE_SCHEMAS:
        raise ValueError(
            f"Expected one of source schemas {sorted(SOURCE_SCHEMAS)!r}; got {schema!r}"
        )
    classification = str(payload.get("quality_gates", {}).get("classification", ""))
    if classification not in SOURCE_CLASSIFICATIONS:
        raise ValueError(
            "Source v3 artifact lacks a valid held-out classification. "
            f"Found {classification!r}; expected one of {sorted(SOURCE_CLASSIFICATIONS)}."
        )
    gates = payload.get("quality_gates")
    if not isinstance(gates, Mapping):
        raise ValueError("Source v3 artifact lacks quality_gates")
    if require_completed_test:
        test_metrics = gates.get("test_metric_means")
        if not isinstance(test_metrics, Mapping) or not test_metrics:
            raise ValueError("Source v3 artifact has no completed held-out source-test metrics")
    model = payload.get("model")
    if require_model and model is None:
        raise ValueError("Source v3 artifact does not contain the frozen reduced model")
    claim_scope = str(payload.get("claim_scope", CLAIM_SCOPE))
    if claim_scope != CLAIM_SCOPE:
        raise ValueError(
            f"Unexpected claim scope {claim_scope!r}; formal downstream analyses require {CLAIM_SCOPE!r}"
        )
    downstream_scope = str(gates.get("downstream_scope", classification))
    provenance = payload.get("provenance") or {}
    if not isinstance(provenance, Mapping):
        raise TypeError("Source v3 provenance must be a mapping")
    return SourceReducedArtifact(
        path=path.resolve(),
        sha256=sha256_file(path),
        schema_version=str(schema),
        classification=classification,
        downstream_scope=downstream_scope,
        claim_scope=claim_scope,
        model=model,
        quality_gates=gates,
        provenance=provenance,
    )


def require_toroidal_projection_scope(source: SourceReducedArtifact, mode: str) -> None:
    """Fail loudly when a projection or perturbation exceeds Source v3 scope.

    ``torus_plus_fibre`` is a legitimate explicit comparator for an exact-2D
    source with ``k=0``; in that case it is mathematically identical to the
    torus-base projection and the downstream summary records that equivalence.
    A standalone fibre direction, however, is undefined when ``k=0``.
    """
    toroidal_modes = {"torus_base", "torus_plus_fibre", "local_tangent"}
    if mode in toroidal_modes and not source.has_toroidal_base:
        raise RuntimeError(
            f"Projection mode {mode!r} requires a held-out toroidal-base gate, but "
            f"Source v3 classification is {source.classification!r}. Use scaffold-safe "
            "diagnostics (identity/matched PCA/random) instead."
        )
    n_fibre = int(
        getattr(getattr(source.model, "fibre_model", None), "n_components", 0)
    )
    if mode == "fibre" and n_fibre <= 0:
        raise RuntimeError(
            "A standalone fibre perturbation was requested, but Source v3 selected k=0."
        )
    if (
        mode == "torus_plus_fibre"
        and source.classification == "toroidal_base_plus_neural_fibre"
        and n_fibre <= 0
    ):
        raise RuntimeError(
            "Source v3 classified the model as toroidal-base-plus-fibre but the "
            "serialized fibre dimension is not positive."
        )


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_jsonable(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hash_bytes(text.encode("utf-8"))


def hash_paths(paths: Iterable[str | Path], *, require_all: bool = True) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            if require_all:
                raise FileNotFoundError(f"Required input file not found: {path}")
            result[str(path)] = None
        else:
            result[str(path)] = sha256_file(path)
    return result


def array_fingerprint(*arrays: np.ndarray, metadata: Mapping[str, Any] | None = None) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        arr = np.ascontiguousarray(array)
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
        digest.update(arr.tobytes())
    if metadata:
        digest.update(hash_jsonable(metadata).encode("ascii"))
    return digest.hexdigest()


def trajectory_fingerprint(
    *,
    start_node: int,
    target_node: int,
    velocities: np.ndarray,
    start_coords: np.ndarray | None = None,
    target_coords: np.ndarray | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    arrays = [
        np.asarray([start_node, target_node], dtype=np.int64),
        np.asarray(velocities, dtype=np.float32),
    ]
    if start_coords is not None:
        arrays.append(np.asarray(start_coords, dtype=np.float32))
    if target_coords is not None:
        arrays.append(np.asarray(target_coords, dtype=np.float32))
    return array_fingerprint(*arrays, metadata=extra)


def assert_disjoint_fingerprint_sets(named: Mapping[str, Sequence[str]]) -> dict[str, int]:
    sets = {name: set(values) for name, values in named.items()}
    overlaps: dict[str, int] = {}
    names = list(sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            key = f"{left}__{right}"
            count = len(sets[left] & sets[right])
            overlaps[key] = count
            if count:
                raise RuntimeError(f"Trajectory leakage detected between {left} and {right}: {count}")
    return overlaps


def write_csv_strict(path: str | Path, rows: Sequence[Mapping[str, Any]], *, required: bool = True) -> Path:
    path = Path(path)
    if required and not rows:
        raise RuntimeError(f"Refusing to write required empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(str(key))
    if required and not keys:
        raise RuntimeError(f"Required CSV has no columns: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized: MutableMapping[str, Any] = {}
            for key in keys:
                value = row.get(key)
                if isinstance(value, (dict, list, tuple, np.ndarray)):
                    value = json.dumps(value if not isinstance(value, np.ndarray) else value.tolist(), ensure_ascii=False, sort_keys=True)
                elif isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, float) and not np.isfinite(value):
                    value = ""
                normalized[key] = value
            writer.writerow(normalized)
    return path


def build_artifact_manifest(
    output_dir: str | Path,
    *,
    schema_version: str,
    required_files: Sequence[str],
    optional_files: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    filename: str = "artifact_manifest.json",
) -> dict[str, Any]:
    """Validate formal artifacts and save their hashes; missing/empty required data fail loudly."""
    output = Path(output_dir)
    files: list[dict[str, Any]] = []
    for rel in required_files:
        path = output / rel
        if not path.is_file():
            raise FileNotFoundError(f"Required formal artifact missing: {path}")
        size = path.stat().st_size
        if size <= 0:
            raise RuntimeError(f"Required formal artifact is empty: {path}")
        files.append({"path": rel, "required": True, "bytes": size, "sha256": sha256_file(path)})
    for rel in optional_files:
        path = output / rel
        if path.is_file() and path.stat().st_size > 0:
            files.append({"path": rel, "required": False, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {
        "schema_version": schema_version,
        "schema_family": SCHEMA_FAMILY,
        "claim_scope": CLAIM_SCOPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "metadata": dict(metadata or {}),
    }
    write_json(payload, output / filename)
    return payload


def source_linkage_payload(source: SourceReducedArtifact) -> dict[str, Any]:
    return {
        "claim_scope": CLAIM_SCOPE,
        "source_reduced_artifact": str(source.path),
        "source_reduced_artifact_sha256": source.sha256,
        "source_reduced_schema_version": source.schema_version,
        "source_classification": source.classification,
        "source_downstream_scope": source.downstream_scope,
        "source_quality_gates": dict(source.quality_gates),
    }
