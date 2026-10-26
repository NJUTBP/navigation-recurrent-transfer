"""Fit and evaluate the leakage-controlled source reduced model v3.

This is a source-only stage.  It does not run hard target tasks, factorial
controls, target action causality, path-independence, manifold interventions or
visual experiments.  All formal wording is conditional on the analyzed
navigation source.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from gpm_cmap.analysis.shared_manifold import (
    load_navigation_trajectory_data,
    torus_artifacts_from_analysis_dir,
)
from gpm_cmap.analysis.source_reduced_v3 import (
    CLAIM_SCOPE,
    SCHEMA_VERSION,
    SourceReducedModelV3,
    audit_disjoint_trajectory_sets,
    evaluate_non_grid_control_v3,
    evaluate_source_quality_gates,
    flatten_test_summary_for_seed,
    trajectory_fingerprints,
)
from gpm_cmap.utils import (
    compute_source_tree_hash,
    configure_runtime,
    load_config,
    prepare_output_dir,
    save_config_snapshot,
    save_run_provenance,
    set_seed,
    sha256_file,
    write_json,
)
from gpm_cmap.utils.logging_utils import get_logger


_LEGACY_RESULT_DIR_NAMES = {
    "shared_manifold_001",
    "shared_manifold_v2_003",
    "manifold_interventions_001",
}


def _resolve_path(value: Optional[str | Path]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser().resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read JSON {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return payload


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fingerprint_by_trajectory(data) -> Dict[int, str]:
    inverse: Dict[int, str] = {}
    for fingerprint, trajectory_ids in trajectory_fingerprints(data).items():
        for trajectory_id in trajectory_ids:
            inverse[int(trajectory_id)] = str(fingerprint)
    if len(inverse) != int(data.unique_trajectories.size):
        raise RuntimeError("Could not assign exactly one fingerprint per trajectory")
    return inverse


def _torus_provenance(analysis_dir: Path) -> Dict[str, Any]:
    path = analysis_dir / "run_provenance.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}; formal source linkage requires analyze_torus provenance"
        )
    return _load_json(path)


def _stored_unit_indices(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if "unit_indices" not in archive:
            raise KeyError(f"{path} lacks unit_indices")
        indices = np.asarray(archive["unit_indices"], dtype=np.int64).reshape(-1)
    if indices.size < 2 or np.unique(indices).size != indices.size:
        raise ValueError(f"Malformed unit_indices in {path}")
    return indices


def _atomic_pickle(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with open(temporary, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write empty formal source-data CSV: {path}")
    fields = sorted({str(key) for row in rows for key in row.keys()})
    if not fields:
        raise RuntimeError(f"Formal source-data CSV has no columns: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            for row in rows:
                normalized: Dict[str, Any] = {}
                for field in fields:
                    value = row.get(field)
                    if isinstance(value, (dict, list, tuple, np.ndarray)):
                        value = json.dumps(
                            np.asarray(value).tolist()
                            if isinstance(value, np.ndarray)
                            else value,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    elif isinstance(value, np.generic):
                        value = value.item()
                    normalized[field] = value
                writer.writerow(normalized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _require_distinct_files(named_paths: Mapping[str, Path]) -> Dict[str, str]:
    hashes = {name: sha256_file(path) for name, path in named_paths.items()}
    names = list(named_paths)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if named_paths[left].resolve() == named_paths[right].resolve():
                raise RuntimeError(f"{left} and {right} point to the same file")
            if hashes[left] == hashes[right]:
                raise RuntimeError(
                    f"{left} and {right} have identical SHA-256 digests; "
                    "independent splits are required"
                )
    return hashes


def _validate_split_manifest(
    path: Path,
    *,
    checkpoint_sha256: str,
    selected_units: np.ndarray,
    grid_paths: Mapping[str, Path],
    grid_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    manifest = _load_json(path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(
            f"Source split manifest schema must be {SCHEMA_VERSION!r}: {path}"
        )
    if manifest.get("claim_scope") != CLAIM_SCOPE:
        raise RuntimeError(
            "Source split manifest claim scope is missing or incompatible: "
            f"{manifest.get('claim_scope')!r}"
        )
    fingerprint_audit = manifest.get("trajectory_fingerprint_audit", {})
    if fingerprint_audit.get("all_disjoint") is not True:
        raise RuntimeError(
            "Source split manifest does not certify disjoint raw trajectory fingerprints"
        )
    if str(manifest.get("checkpoint_sha256")) != str(checkpoint_sha256):
        raise RuntimeError(
            "Source split manifest checkpoint hash does not match the configured "
            f"checkpoint: manifest={manifest.get('checkpoint_sha256')}, "
            f"configured={checkpoint_sha256}"
        )
    manifest_units = np.asarray(
        manifest.get("grid_module_unit_indices", []), dtype=np.int64
    ).reshape(-1)
    if not np.array_equal(manifest_units, np.asarray(selected_units, dtype=np.int64)):
        raise RuntimeError(
            "Source split manifest grid-module indices do not match formal torus units"
        )
    splits = manifest.get("splits")
    if not isinstance(splits, dict):
        raise RuntimeError(f"Malformed source split manifest: {path}")
    trajectory_seeds: List[int] = []
    for split in ("fit", "validation", "test"):
        record = splits.get(split)
        if not isinstance(record, dict):
            raise RuntimeError(f"Split manifest lacks source-{split}")
        expected_hash = str(record.get("grid_arrays_sha256"))
        actual_hash = str(grid_hashes[f"{split}_arrays"])
        if expected_hash != actual_hash:
            raise RuntimeError(
                f"source-{split} hash differs from split manifest: "
                f"manifest={expected_hash}, actual={actual_hash}, path={grid_paths[split]}"
            )
        if record.get("trajectory_seed") is None:
            raise RuntimeError(f"source-{split} manifest record lacks trajectory_seed")
        trajectory_seeds.append(int(record["trajectory_seed"]))
    if len(set(trajectory_seeds)) != 3:
        raise RuntimeError("Source fit/validation/test manifest trajectory seeds overlap")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": manifest.get("schema_version"),
        "checkpoint_hash_matches": True,
        "grid_unit_indices_match": True,
        "split_hashes_match": True,
        "trajectory_seeds": trajectory_seeds,
        "raw_trajectory_fingerprints_disjoint": True,
        "payload": manifest,
    }


def _validate_non_grid_manifest_records(
    manifest_audit: Mapping[str, Any],
    *,
    non_grid_paths: Mapping[str, Path],
    non_grid_hashes: Mapping[str, str],
) -> Dict[str, Any]:
    manifest = manifest_audit.get("payload", {})
    splits = manifest.get("splits", {})
    manifest_units = np.asarray(
        manifest.get("matched_non_grid_unit_indices", []), dtype=np.int64
    ).reshape(-1)
    if manifest_units.size == 0:
        raise RuntimeError(
            "Non-grid control was requested, but the source split manifest has no "
            "matched_non_grid_unit_indices"
        )
    stored_units = _stored_unit_indices(non_grid_paths["fit"])
    if not np.array_equal(stored_units, manifest_units):
        raise RuntimeError(
            "Non-grid fit unit indices differ from source split manifest"
        )
    for split in ("fit", "validation", "test"):
        record = splits.get(split, {})
        expected = str(record.get("non_grid_arrays_sha256"))
        actual = str(non_grid_hashes[f"non_grid_{split}"])
        if expected != actual:
            raise RuntimeError(
                f"source-{split} non-grid hash differs from split manifest: "
                f"manifest={expected}, actual={actual}"
            )
        if not np.array_equal(_stored_unit_indices(non_grid_paths[split]), manifest_units):
            raise RuntimeError(
                f"source-{split} non-grid unit indices differ from split manifest"
            )
    return {
        "hashes_match": True,
        "unit_indices_match": True,
        "n_units": int(manifest_units.size),
    }


def _validate_formal_torus(
    artifacts: Mapping[str, Any],
    analysis_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    minimum_shuffles = int(analysis_cfg.get("minimum_torus_shuffles", 1000))
    confirmed = artifacts.get("source_torus_confirmed") is True
    n_shuffles = int(artifacts.get("n_shuffles") or 0)
    module_selection = artifacts.get("module_selection_input_indices")
    topology_evaluation = artifacts.get("topology_evaluation_input_indices")
    independent_selection = bool(
        module_selection is not None
        and topology_evaluation is not None
        and np.intersect1d(module_selection, topology_evaluation).size == 0
    )
    if bool(analysis_cfg.get("require_formal_source_torus", True)):
        if not confirmed or n_shuffles < minimum_shuffles:
            raise RuntimeError(
                "Source torus is not publication-grade: require is_torus_like=true "
                f"and at least {minimum_shuffles} shuffles; got confirmed={confirmed}, "
                f"n_shuffles={n_shuffles}."
            )
    if bool(analysis_cfg.get("require_independent_module_selection", True)):
        if not independent_selection:
            raise RuntimeError(
                "Grid-module selection and topology evaluation are not independently held out"
            )
    if artifacts.get("topology_evaluation_input_indices") is None:
        raise RuntimeError(
            "torus_analysis_arrays.npz lacks topology_evaluation_input_indices; "
            "the phase anchor must use only the formal topology-evaluation subset"
        )
    if artifacts.get("landmark_indices") is None or artifacts.get("landmark_torus_turns") is None:
        raise RuntimeError(
            "Formal torus artifact lacks decoded cohomological landmark coordinates"
        )
    return {
        "confirmed": confirmed,
        "n_shuffles": n_shuffles,
        "minimum_shuffles": minimum_shuffles,
        "independent_module_selection": independent_selection,
    }


def _load_grid_split(
    path: Path,
    *,
    selected_units: np.ndarray,
    burn_in_steps: int,
    min_step_displacement: float,
):
    return load_navigation_trajectory_data(
        path,
        unit_indices=selected_units,
        burn_in_steps=burn_in_steps,
        min_step_displacement=min_step_displacement,
        require_consecutive=True,
    )


def _load_non_grid_split(
    path: Path,
    *,
    burn_in_steps: int,
    min_step_displacement: float,
):
    indices = _stored_unit_indices(path)
    return load_navigation_trajectory_data(
        path,
        unit_indices=indices,
        burn_in_steps=burn_in_steps,
        min_step_displacement=min_step_displacement,
        require_consecutive=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--analysis_dir", default=None)
    parser.add_argument("--topology_arrays", default=None)
    parser.add_argument("--fit_arrays", default=None)
    parser.add_argument("--validation_arrays", default=None)
    parser.add_argument("--test_arrays", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config_path = _resolve_path(args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(config_path)
    cfg = load_config(config_path)
    source_cfg = dict(cfg.get("source", {}))
    analysis_cfg = dict(cfg.get("analysis", {}))
    model_cfg = dict(cfg.get("model", {}))
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    output_value = cfg.get("output_dir")
    if not output_value:
        raise ValueError("Set output_dir in the config or pass --output_dir")
    if Path(output_value).name in _LEGACY_RESULT_DIR_NAMES:
        raise ValueError(
            "Refusing to reuse a protected legacy result-directory name: "
            f"{Path(output_value).name}"
        )

    analysis_dir = _resolve_path(args.analysis_dir or source_cfg.get("analysis_dir"))
    fit_path = _resolve_path(args.fit_arrays or source_cfg.get("fit_arrays"))
    validation_path = _resolve_path(
        args.validation_arrays or source_cfg.get("validation_arrays")
    )
    test_path = _resolve_path(args.test_arrays or source_cfg.get("test_arrays"))
    checkpoint_path = _resolve_path(args.checkpoint or source_cfg.get("checkpoint"))
    if analysis_dir is None:
        raise ValueError("source.analysis_dir is required")
    torus_provenance = _torus_provenance(analysis_dir)
    topology_path = _resolve_path(
        args.topology_arrays
        or source_cfg.get("topology_arrays")
        or torus_provenance.get("input_arrays")
    )
    required_paths = {
        "topology_arrays": topology_path,
        "fit_arrays": fit_path,
        "validation_arrays": validation_path,
        "test_arrays": test_path,
        "checkpoint": checkpoint_path,
    }
    missing_names = [name for name, path in required_paths.items() if path is None]
    if missing_names:
        raise ValueError(f"Missing required source paths: {missing_names}")
    for name, path in required_paths.items():
        if path is None or not path.exists():
            raise FileNotFoundError(f"{name}: {path}")
    split_manifest_path = _resolve_path(source_cfg.get("split_manifest"))
    if split_manifest_path is None:
        split_manifest_path = fit_path.parent / "source_reduced_split_manifest.json"
    source_cfg.update(
        {
            "analysis_dir": str(analysis_dir),
            "topology_arrays": str(topology_path),
            "fit_arrays": str(fit_path),
            "validation_arrays": str(validation_path),
            "test_arrays": str(test_path),
            "checkpoint": str(checkpoint_path),
            "split_manifest": str(split_manifest_path),
        }
    )
    cfg["source"] = source_cfg
    source_tree = compute_source_tree_hash()

    output_dir = prepare_output_dir(output_value)
    logger = get_logger("analyze_source_reduced_model", output_dir / "analysis.log")
    seed = int(cfg.get("seed", 42))
    set_seed(
        seed,
        deterministic_algorithms=bool(
            cfg.get("runtime", {}).get("deterministic_algorithms", False)
        ),
    )
    configure_runtime(cfg)
    save_config_snapshot(cfg, output_dir)
    effective_config_path = output_dir / "config_used.yaml"
    input_config_sha256 = sha256_file(config_path)
    effective_config_sha256 = sha256_file(effective_config_path)

    artifacts = torus_artifacts_from_analysis_dir(analysis_dir)
    formal_torus = _validate_formal_torus(artifacts, analysis_cfg)
    selected_units = np.asarray(artifacts["selected_unit_indices"], dtype=np.int64)
    source_parameters = artifacts.get("summary", {}).get("parameters", {})
    burn_in_steps = int(
        source_cfg.get(
            "burn_in_steps", source_parameters.get("burn_in_steps", 50)
        )
    )
    min_step_displacement = float(
        source_cfg.get(
            "min_step_displacement",
            source_parameters.get("min_step_displacement", 0.0005),
        )
    )

    input_files = {
        "topology_arrays": topology_path,
        "fit_arrays": fit_path,
        "validation_arrays": validation_path,
        "test_arrays": test_path,
    }
    input_hashes = _require_distinct_files(input_files)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    require_split_manifest = bool(
        analysis_cfg.get("require_source_split_manifest", True)
    )
    split_manifest_audit: Optional[Dict[str, Any]] = None
    if split_manifest_path.exists():
        split_manifest_audit = _validate_split_manifest(
            split_manifest_path,
            checkpoint_sha256=checkpoint_sha256,
            selected_units=selected_units,
            grid_paths={
                "fit": fit_path,
                "validation": validation_path,
                "test": test_path,
            },
            grid_hashes=input_hashes,
        )
        input_files["source_split_manifest"] = split_manifest_path
        input_hashes["source_split_manifest"] = split_manifest_audit["sha256"]
    elif require_split_manifest:
        raise FileNotFoundError(
            "Missing source split manifest required for checkpoint/input linkage: "
            f"{split_manifest_path}"
        )
    expected_topology_hash = torus_provenance.get("input_arrays_sha256")
    topology_hash_matches = bool(
        expected_topology_hash
        and str(expected_topology_hash) == str(input_hashes["topology_arrays"])
    )
    if bool(analysis_cfg.get("require_topology_input_hash_match", True)):
        if not topology_hash_matches:
            raise RuntimeError(
                "Configured topology arrays do not match formal analyze_torus provenance: "
                f"expected {expected_topology_hash}, got {input_hashes['topology_arrays']}"
            )

    topology_data = load_navigation_trajectory_data(
        topology_path,
        unit_indices=selected_units,
        state_indices=np.asarray(
            artifacts["topology_evaluation_input_indices"], dtype=np.int64
        ),
        require_consecutive=True,
    )
    source_fit = _load_grid_split(
        fit_path,
        selected_units=selected_units,
        burn_in_steps=burn_in_steps,
        min_step_displacement=min_step_displacement,
    )
    source_validation = _load_grid_split(
        validation_path,
        selected_units=selected_units,
        burn_in_steps=burn_in_steps,
        min_step_displacement=min_step_displacement,
    )
    source_test = _load_grid_split(
        test_path,
        selected_units=selected_units,
        burn_in_steps=burn_in_steps,
        min_step_displacement=min_step_displacement,
    )
    trajectory_audit = audit_disjoint_trajectory_sets(
        {
            "formal_topology": topology_data,
            "source_fit": source_fit,
            "source_validation": source_validation,
            "source_test": source_test,
        },
        fail_on_overlap=bool(
            analysis_cfg.get("require_disjoint_trajectory_fingerprints", True)
        ),
    )
    logger.info(
        "Loaded disjoint topology/fit/validation/test sets with %d/%d/%d/%d trajectories",
        topology_data.unique_trajectories.size,
        source_fit.unique_trajectories.size,
        source_validation.unique_trajectories.size,
        source_test.unique_trajectories.size,
    )

    model = SourceReducedModelV3.fit(
        topology_data=topology_data,
        source_fit=source_fit,
        source_validation=source_validation,
        selected_unit_indices=selected_units,
        landmark_indices=np.asarray(artifacts["landmark_indices"], dtype=np.int64),
        landmark_torus_turns=np.asarray(
            artifacts["landmark_torus_turns"], dtype=np.float64
        ),
        config=model_cfg,
        seed=seed,
    )
    test_evaluation = model.evaluate_test(
        source_test,
        seed=seed + 1000,
        n_bootstrap=int(analysis_cfg.get("bootstrap_samples", 2000)),
        min_rollout_steps=int(analysis_cfg.get("min_rollout_steps", 2)),
    )
    gates = evaluate_source_quality_gates(
        model,
        test_evaluation,
        thresholds=analysis_cfg.get("quality_gates", {}),
    )

    non_grid_result = None
    non_grid_manifest_audit = None
    non_grid_paths_cfg = source_cfg.get("non_grid", {}) or {}
    if bool(non_grid_paths_cfg.get("enabled", True)):
        non_grid_paths = {
            split: _resolve_path(non_grid_paths_cfg.get(f"{split}_arrays"))
            for split in ("fit", "validation", "test")
        }
        configured = [path is not None for path in non_grid_paths.values()]
        if any(configured) and not all(configured):
            raise ValueError(
                "source.non_grid must specify fit_arrays, validation_arrays and "
                "test_arrays together"
            )
        if all(configured):
            for split, path in non_grid_paths.items():
                if path is None or not path.exists():
                    raise FileNotFoundError(f"non_grid.{split}_arrays: {path}")
            non_grid_hashes = _require_distinct_files(
                {f"non_grid_{name}": path for name, path in non_grid_paths.items()}
            )
            input_files.update(
                {f"non_grid_{name}": path for name, path in non_grid_paths.items()}
            )
            input_hashes.update(non_grid_hashes)
            if split_manifest_audit is not None:
                non_grid_manifest_audit = _validate_non_grid_manifest_records(
                    split_manifest_audit,
                    non_grid_paths=non_grid_paths,
                    non_grid_hashes=non_grid_hashes,
                )
            elif require_split_manifest:
                raise RuntimeError(
                    "A split manifest is required to validate non-grid control provenance"
                )
            non_grid_fit = _load_non_grid_split(
                non_grid_paths["fit"],
                burn_in_steps=burn_in_steps,
                min_step_displacement=min_step_displacement,
            )
            non_grid_validation = _load_non_grid_split(
                non_grid_paths["validation"],
                burn_in_steps=burn_in_steps,
                min_step_displacement=min_step_displacement,
            )
            non_grid_test = _load_non_grid_split(
                non_grid_paths["test"],
                burn_in_steps=burn_in_steps,
                min_step_displacement=min_step_displacement,
            )
            non_grid_result = evaluate_non_grid_control_v3(
                model,
                grid_fit=source_fit,
                grid_validation=source_validation,
                grid_test=source_test,
                non_grid_fit=non_grid_fit,
                non_grid_validation=non_grid_validation,
                non_grid_test=non_grid_test,
                seed=seed + 2000,
                n_bootstrap=int(analysis_cfg.get("bootstrap_samples", 2000)),
                min_rollout_steps=int(analysis_cfg.get("min_rollout_steps", 2)),
            )

    input_hash_bundle_sha256 = _canonical_json_sha256(input_hashes)
    test_fingerprints = _fingerprint_by_trajectory(source_test)
    trajectory_fingerprint_bundle_sha256 = _canonical_json_sha256(
        sorted(test_fingerprints.values())
    )
    provenance_payload = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "seed": seed,
        "config_path": str(config_path),
        "input_config_sha256": input_config_sha256,
        "effective_config_path": str(effective_config_path.resolve()),
        "effective_config_sha256": effective_config_sha256,
        "config_sha256": effective_config_sha256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "source_tree_sha256": source_tree["sha256"],
        "source_tree_file_count": source_tree["file_count"],
        "input_files": {name: str(path) for name, path in input_files.items()},
        "input_hashes": input_hashes,
        "input_hash_bundle_sha256": input_hash_bundle_sha256,
        "torus_analysis_dir": str(analysis_dir),
        "torus_formal_gate": formal_torus,
        "topology_hash_matches_torus_provenance": topology_hash_matches,
        "trajectory_fingerprints": trajectory_audit,
        "source_test_trajectory_fingerprint_by_id": test_fingerprints,
        "source_test_trajectory_fingerprint_bundle_sha256": (
            trajectory_fingerprint_bundle_sha256
        ),
        "source_split_manifest_audit": (
            None
            if split_manifest_audit is None
            else {
                key: value
                for key, value in split_manifest_audit.items()
                if key != "payload"
            }
        ),
        "non_grid_manifest_audit": non_grid_manifest_audit,
        "selected_unit_indices": selected_units.tolist(),
        "model_selection_used_source_test": False,
    }
    run_provenance = save_run_provenance(
        output_dir,
        "analyze_source_reduced_model",
        cfg=cfg,
        extra=provenance_payload,
    )

    artifact_path = output_dir / "source_reduced_model_v3.pkl"
    _atomic_pickle(
        {
            "schema_version": SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "model": model,
            "quality_gates": gates,
            "provenance": provenance_payload,
        },
        artifact_path,
    )
    per_seed_row = flatten_test_summary_for_seed(
        test_evaluation, gates, seed=seed
    )
    per_seed_row.update(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "source_tree_sha256": source_tree["sha256"],
            "config_sha256": effective_config_sha256,
            "input_config_sha256": input_config_sha256,
            "input_hash_bundle_sha256": input_hash_bundle_sha256,
            "trajectory_fingerprint_bundle_sha256": (
                trajectory_fingerprint_bundle_sha256
            ),
            "selected_fibre_dimension": model.fibre_model.n_components,
            "selected_decoder_context": model.selected_decoder.context_mode,
            "selected_fourier_order": model.selected_decoder.feature_map.order,
        }
    )
    trajectory_common = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "source_seed": seed,
        "checkpoint_sha256": checkpoint_sha256,
        "source_tree_sha256": source_tree["sha256"],
        "config_sha256": effective_config_sha256,
        "input_hash_bundle_sha256": input_hash_bundle_sha256,
        "source_test_arrays_sha256": input_hashes["test_arrays"],
    }
    for row in test_evaluation["trajectory_metrics"]:
        trajectory_id = int(row["trajectory_id"])
        row.update(trajectory_common)
        row["trajectory_fingerprint"] = test_fingerprints[trajectory_id]
    if non_grid_result is not None:
        for row in non_grid_result["trajectory_metrics"]:
            trajectory_id = int(row["trajectory_id"])
            row.update(trajectory_common)
            row["trajectory_fingerprint"] = test_fingerprints[trajectory_id]
    per_seed_csv = output_dir / "source_reduced_model_v3_test_metrics.csv"
    per_trajectory_csv = (
        output_dir / "source_reduced_model_v3_per_trajectory_metrics.csv"
    )
    _write_csv([per_seed_row], per_seed_csv)
    _write_csv(test_evaluation["trajectory_metrics"], per_trajectory_csv)
    if non_grid_result is not None:
        _write_csv(
            non_grid_result["trajectory_metrics"],
            output_dir / "source_reduced_model_v3_non_grid_per_trajectory_metrics.csv",
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "stage": "source_reduced_model_v3_only",
        "later_upgrade_stages_executed": False,
        "config": cfg,
        "model_selection": model.diagnostics,
        "source_test": test_evaluation,
        "quality_gates": gates,
        "non_grid_control": non_grid_result,
        "provenance": provenance_payload,
        "run_provenance": run_provenance,
        "artifacts": {
            "model_pickle": str(artifact_path.resolve()),
            "per_seed_metrics_csv": str(per_seed_csv.resolve()),
            "per_trajectory_metrics_csv": str(per_trajectory_csv.resolve()),
            "artifact_manifest": str(
                (output_dir / "source_reduced_model_v3_artifact_manifest.json").resolve()
            ),
        },
        "legacy_compatibility": {
            "old_checkpoint_format_changed": False,
            "old_analysis_entrypoints_removed": False,
            "legacy_metric_aliases": {
                "surface_cross_mfe": "surface_mfe_torus_base",
                "rollout_cross_nmte": "rollout_nmte_torus_base",
                "torus_tangent_fraction": "torus_base_tangent_fraction",
            },
        },
    }
    summary_path = output_dir / "source_reduced_model_v3_summary.json"
    write_json(summary, summary_path)

    artifact_manifest_path = (
        output_dir / "source_reduced_model_v3_artifact_manifest.json"
    )
    artifact_manifest_inputs = {
        "source_reduced_model_v3.pkl": artifact_path,
        "source_reduced_model_v3_summary.json": summary_path,
        "source_reduced_model_v3_test_metrics.csv": per_seed_csv,
        "source_reduced_model_v3_per_trajectory_metrics.csv": per_trajectory_csv,
        "run_provenance.json": output_dir / "run_provenance.json",
        "config_used.yaml": effective_config_path,
    }
    non_grid_csv = (
        output_dir / "source_reduced_model_v3_non_grid_per_trajectory_metrics.csv"
    )
    if non_grid_csv.exists():
        artifact_manifest_inputs[non_grid_csv.name] = non_grid_csv
    write_json(
        {
            "schema_version": SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "seed": seed,
            "checkpoint_sha256": checkpoint_sha256,
            "source_tree_sha256": source_tree["sha256"],
            "config_sha256": effective_config_sha256,
            "input_hash_bundle_sha256": input_hash_bundle_sha256,
            "artifacts": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                    "size_bytes": int(path.stat().st_size),
                }
                for name, path in artifact_manifest_inputs.items()
            },
        },
        artifact_manifest_path,
    )

    # Fail-loud final artifact contract.  No formal downstream figure should be
    # able to discover a partial run with missing source data.
    required_outputs = [
        artifact_path,
        summary_path,
        per_seed_csv,
        per_trajectory_csv,
        output_dir / "run_provenance.json",
        output_dir / "config_used.yaml",
        artifact_manifest_path,
    ]
    missing_outputs = [str(path) for path in required_outputs if not path.exists()]
    empty_outputs = [str(path) for path in required_outputs if path.stat().st_size == 0]
    if missing_outputs or empty_outputs:
        raise RuntimeError(
            f"Incomplete formal v3 output: missing={missing_outputs}, empty={empty_outputs}"
        )
    logger.info("Source reduced model v3 classification: %s", gates["classification"])
    logger.info("Permitted wording: %s", gates["permitted_claim"])
    logger.info("Saved frozen v3 artifact to %s", artifact_path)


if __name__ == "__main__":
    main()
