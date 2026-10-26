"""Fit the Schøyen-anchored source reduced model for the rebuilt Figure A3.

Formal topology is *not* recomputed here.  The source is eligible when the
``schoyen_euclidean_formal_topology_v1`` contract passes.  Two periodic
coordinates are then anchored on an independent navigation calibration split
using the dominant Fourier wavevectors of the Schøyen-selected ensemble.
All reduced-model hyperparameters are selected with source fit/validation data;
source test is consumed once for the existing preregistered Source-v3 quality
metrics.  Thus the mature dynamics machinery is reused while the obsolete
cohomological phase anchor is completely removed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from gpm_cmap.analysis.shared_manifold import load_navigation_trajectory_data
from gpm_cmap.analysis.source_reduced_schoyen import (
    build_schoyen_phase_encoder,
    evaluate_schoyen_phase_encoder,
)
from gpm_cmap.analysis.source_reduced_v3 import (
    CLAIM_SCOPE,
    SourceReducedModelV3,
    audit_disjoint_trajectory_sets,
    evaluate_source_quality_gates,
    flatten_test_summary_for_seed,
)
from gpm_cmap.experiments.analyze_source_reduced_model import _atomic_pickle, _write_csv
from gpm_cmap.utils import (
    configure_runtime,
    load_config,
    save_config_snapshot,
    save_run_provenance,
    set_seed,
    write_json,
)
from gpm_cmap.utils.checkpointing import sha256_file
from gpm_cmap.utils.logging_utils import get_logger

SCHEMA_VERSION = "source_reduced_schoyen_v1"
FORMAL_SCHEMA = "schoyen_euclidean_formal_topology_v1"


def read_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return obj


def stored_topology_selection(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with np.load(path, allow_pickle=False) as z:
        units = np.asarray(z["selected_unit_indices"], dtype=np.int64).reshape(-1)
        families = (
            np.asarray(z["kmeans3_family_labels"], dtype=np.int64).reshape(-1)
            if "kmeans3_family_labels" in z else None
        )
    if units.size < 2 or np.unique(units).size != units.size:
        raise ValueError("Malformed Schøyen selected_unit_indices")
    if families is not None and families.size != units.size:
        raise ValueError("kmeans3_family_labels length differs from selected units")
    return units, families


def load_split(path: Path, units: np.ndarray, burn_in: int, min_step: float):
    return load_navigation_trajectory_data(
        path,
        unit_indices=units,
        burn_in_steps=int(burn_in),
        min_step_displacement=float(min_step),
        require_consecutive=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--formal-topology", required=True)
    ap.add_argument("--topology-dir", required=True)
    ap.add_argument("--phase-arrays", required=True)
    ap.add_argument("--fit-arrays", required=True)
    ap.add_argument("--validation-arrays", required=True)
    ap.add_argument("--test-arrays", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg["output_dir"] = str(out)
    save_config_snapshot(cfg, out)
    logger = get_logger("analyze_source_reduced_schoyen", out / "analysis.log")
    seed = int(cfg.get("seed", 42))
    set_seed(seed, deterministic_algorithms=bool(cfg.get("runtime", {}).get("deterministic_algorithms", False)))
    configure_runtime(cfg)

    formal_path = Path(args.formal_topology).expanduser().resolve()
    topology_dir = Path(args.topology_dir).expanduser().resolve()
    phase_path = Path(args.phase_arrays).expanduser().resolve()
    fit_path = Path(args.fit_arrays).expanduser().resolve()
    val_path = Path(args.validation_arrays).expanduser().resolve()
    test_path = Path(args.test_arrays).expanduser().resolve()
    manifest_path = Path(args.split_manifest).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    for path in (formal_path, phase_path, fit_path, val_path, test_path, manifest_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    formal = read_json(formal_path)
    if formal.get("schema_version") != FORMAL_SCHEMA:
        raise ValueError(f"Expected formal schema {FORMAL_SCHEMA}; got {formal.get('schema_version')}")
    if formal.get("formal_topology_passed") is not True:
        raise RuntimeError("Schøyen Euclidean formal topology did not pass; reduced mechanism is not evaluable")
    manifest = read_json(manifest_path)
    if manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise RuntimeError("Source split manifest and navigation checkpoint differ")

    topology_arrays = topology_dir / "torus_analysis_arrays.npz"
    if not topology_arrays.is_file():
        raise FileNotFoundError(topology_arrays)
    units, family_labels = stored_topology_selection(topology_arrays)
    manifest_units = np.asarray(manifest.get("selected_unit_indices", []), dtype=np.int64)
    if not np.array_equal(units, manifest_units):
        raise RuntimeError("Schøyen selected units differ from source-split manifest")

    source_cfg = cfg.get("source", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    analysis_cfg = cfg.get("analysis", {}) or {}
    phase_cfg = cfg.get("schoyen_phase", {}) or {}
    burn_in = int(source_cfg.get("burn_in_steps", 50))
    min_step = float(source_cfg.get("min_step_displacement", 0.0005))
    phase_data = load_split(phase_path, units, burn_in, min_step)
    source_fit = load_split(fit_path, units, burn_in, min_step)
    source_val = load_split(val_path, units, burn_in, min_step)
    source_test = load_split(test_path, units, burn_in, min_step)
    trajectory_audit = audit_disjoint_trajectory_sets(
        {"phase_calibration": phase_data, "source_fit": source_fit, "source_validation": source_val, "source_test": source_test},
        fail_on_overlap=True,
    )

    arena_size = float(phase_cfg.get("arena_size", cfg.get("data", {}).get("arena_size", 2.2)))
    phase_encoder = build_schoyen_phase_encoder(
        phase_data,
        selected_unit_indices=units,
        arena_size=arena_size,
        resolution=int(phase_cfg.get("ratemap_resolution", 64)),
        smooth_sigma=float(phase_cfg.get("ratemap_smooth_sigma", 1.0)),
        family_labels=family_labels,
        max_fft_frequency=int(phase_cfg.get("max_fft_frequency", 16)),
        chart_dim=int(model_cfg.get("chart_dim", 8)),
        max_landmarks=int(phase_cfg.get("max_chart_landmarks", 8000)),
        seed=seed,
    )
    phase_validation = evaluate_schoyen_phase_encoder(phase_encoder, source_val)
    phase_test = evaluate_schoyen_phase_encoder(phase_encoder, source_test)
    logger.info("Schoyen phase chart validation RMSE=%.4f rad; test RMSE=%.4f rad", phase_validation["phase_rmse_rad"], phase_test["phase_rmse_rad"])

    # Reuse the mature reduced-dynamics/fibre machinery, but inject the new
    # Schøyen/position-Fourier phase encoder instead of cohomological landmarks.
    model = SourceReducedModelV3.fit(
        topology_data=phase_data,
        source_fit=source_fit,
        source_validation=source_val,
        selected_unit_indices=units,
        phase_encoder=phase_encoder,
        config=model_cfg,
        seed=seed,
    )
    test_eval = model.evaluate_test(
        source_test,
        seed=seed + 1000,
        n_bootstrap=int(analysis_cfg.get("bootstrap_samples", 2000)),
        min_rollout_steps=int(analysis_cfg.get("min_rollout_steps", 2)),
    )
    gates = evaluate_source_quality_gates(
        model,
        test_eval,
        thresholds=analysis_cfg.get("quality_gates", {}),
    )
    gates["formal_topology"] = {
        "schema_version": FORMAL_SCHEMA,
        "passed": True,
        "checks": formal.get("checks"),
    }
    gates["schoyen_phase_chart"] = {
        "selection_used_source_test": False,
        "validation": phase_validation,
        "test": phase_test,
        "diagnostics": phase_encoder.diagnostics,
    }
    # Formal-topology pass is the evaluability condition. Reduced-model gates
    # remain outcomes, not filters on whether action/path analyses are computed.
    gates["mechanism_evaluable"] = True

    checkpoint_sha = sha256_file(checkpoint)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "formal_topology_contract": str(formal_path),
        "formal_topology_contract_sha256": sha256_file(formal_path),
        "formal_topology_definition": "Schoyen selected candidate, Euclidean PH (1,2,1) gate",
        "phase_coordinate_definition": "TRUE-position Fourier phase of Schoyen-selected ensemble on independent calibration trajectories",
        "topology_dir": str(topology_dir),
        "topology_arrays_sha256": sha256_file(topology_arrays),
        "split_manifest": str(manifest_path),
        "split_manifest_sha256": sha256_file(manifest_path),
        "input_arrays": {
            "phase": {"path": str(phase_path), "sha256": sha256_file(phase_path)},
            "fit": {"path": str(fit_path), "sha256": sha256_file(fit_path)},
            "validation": {"path": str(val_path), "sha256": sha256_file(val_path)},
            "test": {"path": str(test_path), "sha256": sha256_file(test_path)},
        },
        "trajectory_fingerprints": trajectory_audit,
        "selected_unit_indices": units.tolist(),
        "kmeans3_family_labels_available": family_labels is not None,
        "selected_k_vectors": phase_encoder.diagnostics.get("selected_k_vectors"),
        "model_selection_used_source_test": False,
    }
    run_prov = save_run_provenance(out, "analyze_source_reduced_schoyen", cfg=cfg, extra=provenance)

    artifact = out / "source_reduced_schoyen_v1.pkl"
    _atomic_pickle({
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "model": model,
        "quality_gates": gates,
        "provenance": provenance,
    }, artifact)

    per_seed = flatten_test_summary_for_seed(test_eval, gates, seed=seed)
    per_seed.update({
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "formal_topology_passed": True,
        "phase_chart_validation_rmse_rad": phase_validation["phase_rmse_rad"],
        "phase_chart_test_rmse_rad": phase_test["phase_rmse_rad"],
        "selected_k_vectors": json.dumps(phase_encoder.diagnostics.get("selected_k_vectors")),
    })
    test_csv = out / "source_reduced_schoyen_test_metrics.csv"
    traj_csv = out / "source_reduced_schoyen_per_trajectory_metrics.csv"
    _write_csv([per_seed], test_csv)
    _write_csv(test_eval["trajectory_metrics"], traj_csv)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "stage": "source_reduced_schoyen_v1",
        "formal_topology": formal,
        "phase_chart": {
            "validation": phase_validation,
            "test": phase_test,
            "diagnostics": phase_encoder.diagnostics,
        },
        "model_selection": model.diagnostics,
        "source_test": test_eval,
        "quality_gates": gates,
        "provenance": provenance,
        "run_provenance": run_prov,
        "artifacts": {
            "model_pickle": str(artifact),
            "per_seed_metrics_csv": str(test_csv),
            "per_trajectory_metrics_csv": str(traj_csv),
        },
    }
    write_json(summary, out / "source_reduced_schoyen_summary.json")
    write_json({
        "schema_version": SCHEMA_VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "artifacts": {
            p.name: {"path": str(p.resolve()), "sha256": sha256_file(p)}
            for p in [artifact, test_csv, traj_csv, out / "source_reduced_schoyen_summary.json", out / "run_provenance.json", out / "config_used.yaml"]
            if p.exists()
        },
    }, out / "source_reduced_schoyen_artifact_manifest.json")
    print(json.dumps({
        "source_seed": seed,
        "classification": gates.get("classification"),
        "exact_2d": gates.get("exact_2d_gate", {}).get("passed"),
        "base_plus_fibre": gates.get("toroidal_base_plus_fibre_gate", {}).get("passed"),
        "phase_chart_test_rmse_rad": phase_test["phase_rmse_rad"],
    }, indent=2))


if __name__ == "__main__":
    main()
