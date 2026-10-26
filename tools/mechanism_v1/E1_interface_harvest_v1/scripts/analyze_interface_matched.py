#!/usr/bin/env python3
"""E1 addendum: performance-matched contrasts and the cross-experiment link.

Why this exists
---------------
The declared null family in ``analyze_interface_magnitude.py`` is evaluated at
the largest common budget.  At that point the navigation arm is at the accuracy
ceiling while the delayed-reconstruction arm is not yet converged, so the shape
metrics there are confounded with convergence state.  The scientifically correct
version compares each arm at MATCHED ACCURACY.  This script adds that contrast,
plus the source-level link to the Phase F 8x8 learned gain.

Contrasts added
---------------
P1  Performance-matched shape and magnitude.
    For every source, take navigation at the budget whose accuracy is closest to
    the delayed-reconstruction arm's accuracy at its own largest budget, then
    contrast antipodality, Procrustes residual, axis-angle deviation and
    magnitude.  Declared expectation: the three shape metrics null out and the
    magnitude contrast survives.

P2  Cross-experiment source-level table.
    Joins the 4x4 interface magnitude with the Phase F 8x8 learned scalar gain
    and the 8x8 accuracies, then reports Spearman and Pearson correlations.
    Everything here is EXPLORATORY: n = 10, the predictor ranges are narrow, and
    no multiplicity correction is applied.  Read the printed caveats.

Usage
-----
    "$PYTHON_BIN" analyze_interface_matched.py \
        --in       .../followups/mechanism_v1/E1_interface \
        --phasef   .../rerun_best_validation/aggregate_full/hard8x8_phaseF_followup \
        --out      .../followups/mechanism_v1/E1_interface
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

STATISTICS_SEED = 982451653
BOOTSTRAP_SAMPLES = 20000
SHAPE_METRICS = ("antipodality_index", "procrustes_residual", "axis_angle_deviation")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    tmp.replace(path)


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def exact_sign_flip_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    count = sum(
        1 for signs in itertools.product((1.0, -1.0), repeat=n)
        if abs(float(np.mean(np.asarray(signs) * values))) >= observed - 1e-12
    )
    return count / float(2 ** n)


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(STATISTICS_SEED)
    index = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    means = values[index].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(values: np.ndarray) -> np.ndarray:
        order = values.argsort()
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(values.size, dtype=np.float64)
        # average ties
        _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
        if counts.max() > 1:
            sums = np.zeros(counts.size)
            np.add.at(sums, inverse, ranks)
            ranks = (sums / counts)[inverse]
        return ranks
    rx, ry = rank(np.asarray(x, float)), rank(np.asarray(y, float))
    return float(np.corrcoef(rx, ry)[0, 1])


def permutation_p(x: np.ndarray, y: np.ndarray, statistic, draws: int = 200000) -> float:
    rng = np.random.default_rng(STATISTICS_SEED)
    observed = abs(statistic(x, y))
    y = np.asarray(y, float)
    count = 0
    for _ in range(draws):
        if abs(statistic(x, rng.permutation(y))) >= observed - 1e-12:
            count += 1
    return (count + 1) / (draws + 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="E1 addendum")
    parser.add_argument("--in", dest="input_dir", required=True, type=Path)
    parser.add_argument("--phasef", dest="phasef_dir", type=Path, default=None)
    parser.add_argument("--out", dest="output_dir", required=True, type=Path)
    args = parser.parse_args()

    rows = read_csv(args.input_dir / "source_level_interface.csv")
    table: dict[tuple[int, str, int], dict[str, float]] = {}
    for row in rows:
        key = (int(row["source_seed"]), row["arm"], int(row["budget"]))
        table[key] = {k: as_float(v) for k, v in row.items() if k not in ("arm",)}
    sources = sorted({key[0] for key in table})
    budgets = {
        arm: sorted({b for (_, a, b) in table if a == arm})
        for arm in {a for (_, a, _) in table}
    }
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- P1: performance-matched ---------------------------
    # Shape metrics move steeply with accuracy, so a single "nearest" match can
    # land just above or just below the control and flip the contrast. Report
    # three matchings and let the sensitivity be visible: the nearest navigation
    # budget below the control's accuracy, the nearest above, and the nearest
    # overall. A result that only holds for one of the three is not a result.
    reference_budget = max(budgets["ctl01_frozen"])

    def build_matching(mode: str) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for source in sources:
            reference = table.get((source, "ctl01_frozen", reference_budget))
            if reference is None or not math.isfinite(reference["test_accuracy"]):
                continue
            target_accuracy = reference["test_accuracy"]
            best_budget, best_gap = None, None
            for budget in budgets["nav_frozen"]:
                candidate = table.get((source, "nav_frozen", budget))
                if candidate is None or not math.isfinite(candidate["test_accuracy"]):
                    continue
                delta = candidate["test_accuracy"] - target_accuracy
                if mode == "below" and delta > 0:
                    continue
                if mode == "above" and delta < 0:
                    continue
                gap = abs(delta)
                if best_gap is None or gap < best_gap:
                    best_budget, best_gap = budget, gap
            if best_budget is None:
                continue
            navigation = table[(source, "nav_frozen", best_budget)]
            record: dict[str, Any] = {
                "matching": mode,
                "source_seed": source,
                "nav_budget": best_budget,
                "ctl01_budget": reference_budget,
                "nav_accuracy": navigation["test_accuracy"],
                "ctl01_accuracy": target_accuracy,
                "accuracy_gap": navigation["test_accuracy"] - target_accuracy,
            }
            for metric in SHAPE_METRICS + ("M_rel_codebook",):
                record[f"nav_{metric}"] = navigation.get(metric, float("nan"))
                record[f"ctl01_{metric}"] = reference.get(metric, float("nan"))
                record[f"diff_{metric}"] = record[f"nav_{metric}"] - record[f"ctl01_{metric}"]
            collected.append(record)
        return collected

    matchings = {mode: build_matching(mode) for mode in ("below", "above", "nearest")}
    matched_rows = [row for mode in ("below", "above", "nearest") for row in matchings[mode]]

    write_csv(
        out_dir / "performance_matched_rows.csv", matched_rows,
        ["matching", "source_seed", "nav_budget", "ctl01_budget", "nav_accuracy",
         "ctl01_accuracy", "accuracy_gap"]
        + [f"{side}_{metric}" for metric in SHAPE_METRICS + ("M_rel_codebook",)
           for side in ("nav", "ctl01", "diff")],
    )

    matched_contrasts: list[dict[str, Any]] = []
    for mode in ("below", "above", "nearest"):
        block = matchings[mode]
        if not block:
            continue
        for metric in SHAPE_METRICS + ("M_rel_codebook",):
            values = np.array([r[f"diff_{metric}"] for r in block], dtype=np.float64)
            low, high = bootstrap_ci(values)
            matched_contrasts.append({
                "contrast": f"{mode}_{metric}_nav_minus_ctl01",
                "family": f"performance_matched_{mode}",
                "matching": mode,
                "n_sources": int(np.isfinite(values).sum()),
                "nav_accuracy_mean": float(np.nanmean([r["nav_accuracy"] for r in block])),
                "ctl01_accuracy_mean": float(np.nanmean([r["ctl01_accuracy"] for r in block])),
                "nav_mean": float(np.nanmean([r[f"nav_{metric}"] for r in block])),
                "ctl01_mean": float(np.nanmean([r[f"ctl01_{metric}"] for r in block])),
                "mean_difference": float(np.nanmean(values)),
                "ci_low": low, "ci_high": high,
                "exact_p": exact_sign_flip_p(values),
                "n_negative": int(np.sum(values < 0)),
                "expectation": ("null (shape is task-imposed)" if metric in SHAPE_METRICS
                                else "negative (navigation needs less magnitude)"),
            })
    write_csv(out_dir / "performance_matched_contrasts.csv", matched_contrasts,
              ["contrast", "family", "matching", "n_sources", "nav_accuracy_mean",
               "ctl01_accuracy_mean", "nav_mean", "ctl01_mean", "mean_difference",
               "ci_low", "ci_high", "exact_p", "n_negative", "expectation"])

    # ---------------- P2: cross-experiment ------------------------------
    cross_rows: list[dict[str, Any]] = []
    correlations: list[dict[str, Any]] = []
    if args.phasef_dir is not None and args.phasef_dir.is_dir():
        diagnostics = read_csv(args.phasef_dir / "all_condition_diagnostics.csv")
        metrics = read_csv(args.phasef_dir / "source_level_metrics.csv")

        gains: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        for row in diagnostics:
            gain = as_float(row.get("gain"))
            if math.isfinite(gain):
                gains[row["condition"]][int(row["source_seed"])].append(gain)

        accuracy: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
        for row in metrics:
            accuracy[(row["condition"], row["split"])][int(row["source_seed"])] = \
                as_float(row["accuracy"])

        nav_max = max(budgets["nav_frozen"])
        ctl_max = max(budgets["ctl01_frozen"])
        for source in sources:
            navigation = table.get((source, "nav_frozen", nav_max), {})
            control = table.get((source, "ctl01_frozen", ctl_max), {})
            gain_nav = gains["navigation_trainable_scalar_gain"].get(source, [])
            gain_matched = gains["matched_random_trainable_scalar_gain"].get(source, [])
            cross_rows.append({
                "source_seed": source,
                "M4x4_nav": navigation.get("M_rel_codebook", float("nan")),
                "M4x4_ctl01": control.get("M_rel_codebook", float("nan")),
                "gain8x8_nav": float(np.mean(gain_nav)) if gain_nav else float("nan"),
                "gain8x8_matched": float(np.mean(gain_matched)) if gain_matched else float("nan"),
                "acc8x8_id_seen_short": accuracy[("navigation_trainable_scalar_gain", "id_seen_short")].get(source, float("nan")),
                "acc8x8_seen_long": accuracy[("navigation_trainable_scalar_gain", "seen_long")].get(source, float("nan")),
            })
        write_csv(out_dir / "cross_experiment_source_table.csv", cross_rows,
                  ["source_seed", "M4x4_nav", "M4x4_ctl01", "gain8x8_nav",
                   "gain8x8_matched", "acc8x8_id_seen_short", "acc8x8_seen_long"])

        pairs = [
            ("M4x4_nav", "gain8x8_nav"),
            ("M4x4_nav", "acc8x8_seen_long"),
            ("gain8x8_nav", "acc8x8_seen_long"),
            ("gain8x8_nav", "acc8x8_id_seen_short"),
            ("M4x4_ctl01", "gain8x8_matched"),
        ]
        for left, right in pairs:
            x = np.array([r[left] for r in cross_rows], dtype=np.float64)
            y = np.array([r[right] for r in cross_rows], dtype=np.float64)
            keep = np.isfinite(x) & np.isfinite(y)
            if keep.sum() < 4:
                continue
            x, y = x[keep], y[keep]
            correlations.append({
                "left": left, "right": right, "n": int(keep.sum()),
                "spearman_rho": spearman(x, y),
                "permutation_p": permutation_p(x, y, spearman, draws=20000),
                "pearson_r": float(np.corrcoef(x, y)[0, 1]),
                "status": "EXPLORATORY: n=10, narrow predictor range, no multiplicity control",
            })
        write_csv(out_dir / "cross_experiment_correlations.csv", correlations,
                  ["left", "right", "n", "spearman_rho", "permutation_p", "pearson_r", "status"])

    # ---------------- report --------------------------------------------
    lines = ["# E1 addendum: performance-matched contrasts and cross-experiment link\n"]
    lines.append("## Performance-matched contrasts\n")
    lines.append(
        f"Navigation is compared against the delayed-reconstruction arm at budget "
        f"{reference_budget}. Three matchings are reported because the shape metrics "
        f"move steeply with accuracy: the nearest navigation budget BELOW the control's "
        f"accuracy, the nearest ABOVE, and the nearest overall. A shape effect that "
        f"appears in only one matching is a convergence artifact, not a result.\n"
    )
    lines.append("| matching | nav acc | ctl01 acc | metric | n | nav | ctl01 | difference | 95% CI | exact p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for record in matched_contrasts:
        metric = record["contrast"].split("_", 1)[1].replace("_nav_minus_ctl01", "")
        lines.append(
            f"| {record['matching']} | {record['nav_accuracy_mean']:.4f} | "
            f"{record['ctl01_accuracy_mean']:.4f} | {metric} | {record['n_sources']} | "
            f"{record['nav_mean']:.4g} | {record['ctl01_mean']:.4g} | "
            f"{record['mean_difference']:+.4g} | "
            f"[{record['ci_low']:+.4g}, {record['ci_high']:+.4g}] | "
            f"{record['exact_p']:.6f} |"
        )
    lines.append("")

    if correlations:
        lines.append("## Cross-experiment source-level correlations (EXPLORATORY)\n")
        lines.append("| left | right | n | Spearman rho | permutation p | Pearson r |")
        lines.append("|---|---|---|---|---|---|")
        for record in correlations:
            lines.append(
                f"| {record['left']} | {record['right']} | {record['n']} | "
                f"{record['spearman_rho']:+.3f} | {record['permutation_p']:.4f} | "
                f"{record['pearson_r']:+.3f} |"
            )
        lines.append("")
        lines.append("These are exploratory. n = 10, the predictor ranges are narrow, and no "
                     "multiplicity correction is applied. A null here is uninformative rather "
                     "than evidence of absence. Note also that the 8x8 learned gain and the 8x8 "
                     "accuracies come from the SAME training run, so any association between "
                     "them is a within-run association and not a source-side prediction.\n")

    (out_dir / "INTERFACE_ADDENDUM.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "INTERFACE_ADDENDUM.json").write_text(json.dumps({
        "performance_matched_reference_budget": reference_budget,
        "performance_matched_contrasts": matched_contrasts,
        "cross_experiment_correlations": correlations,
    }, indent=2), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
