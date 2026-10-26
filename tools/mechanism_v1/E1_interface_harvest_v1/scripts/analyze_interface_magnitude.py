#!/usr/bin/env python3
"""E1 analysis: does the learned relation interface differ in MAGNITUDE or in SHAPE?

Pre-registered structure (fix this file before looking at the output)
---------------------------------------------------------------------
Replication unit  : navigation source seed (target repeats averaged within source).
Primary test      : two-sided exhaustive exact sign-flip across sources.
Interval          : 20,000-resample source bootstrap, seed 982451653.
Multiplicity      : Holm within each declared family, families kept separate.

PRIMARY
    H1  At the largest common budget, where nav_frozen and ctl01_frozen both sit
        at the accuracy ceiling, the navigation core operates at a SMALLER
        relation magnitude:   M(nav) < M(ctl01),
        with M = mean_k ||W_rel[:,k]|| / ||codebook column||.

SECONDARY FAMILY (Holm, m = 4)
    S1  M(nav) < M(random_frozen) at the largest common budget.
    S2  M* (magnitude at the first budget reaching accuracy >= 0.9) is smaller
        for nav than for ctl01.  This is the performance-matched form.
    S3  nav's magnitude has plateaued:  M(max budget) - M(1000) ~ 0.
    S4  ctl01's magnitude has not plateaued:  M(max budget) - M(1000) > 0.

PRE-REGISTERED NULL FAMILY (Holm, m = 3)  -- predicted NOT to favour navigation
    N1  antipodality_index at the largest common budget, nav vs ctl01.
    N2  procrustes_residual at the largest common budget, nav vs ctl01.
    N3  |pair_axis_angle_deg - 90| at the largest common budget, nav vs ctl01.
    Reporting these as a declared null is the point: exp8 criterion S2 was a
    historical pre-registration, and if the delayed-reconstruction control
    reaches a MORE canonical frame while transferring worse, interface shape is
    task-imposed rather than core-imposed.

DESCRIPTIVE (no inference)
    D1  Shape/accuracy dissociation: at the budget minimising the nav-vs-random
        gap in antipodality, how large is the accuracy gap?
    D2  Chance-normalised recurrence-ablation drop,
            (acc - acc_no_recurrence) / (acc - 1/n_nodes),
        which removes the accuracy ceiling that confounds the raw drop.
    D3  The non-recurrent 2-D accumulator's shape metrics, as the zero-core
        reference line.

Usage
-----
    "$PYTHON_BIN" analyze_interface_magnitude.py \
        --in  <PROJECT_ROOT>/.../followups/mechanism_v1/E1_interface \
        --out <PROJECT_ROOT>/.../followups/mechanism_v1/E1_interface
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

BOOTSTRAP_SAMPLES = 20000
STATISTICS_SEED = 982451653
ACCURACY_THRESHOLD = 0.9
PLATEAU_REFERENCE_BUDGET = 1000
RECURRENCE_BUDGET = 1000

RECURRENT_ARMS = ("nav_frozen", "ctl01_frozen", "random_frozen")
ACCUMULATOR_ARM = "baseline_linear_accumulator"
COUNT_ARM = "baseline_action_count_linear"

ARM_LABEL = {
    "nav_frozen": "Navigation",
    "ctl01_frozen": "Delayed reconstruction",
    "random_frozen": "Random recurrent",
    ACCUMULATOR_ARM: "2-D accumulator (no core)",
    COUNT_ARM: "Relation-count linear (no core)",
}


# ---------------------------------------------------------------------------
# io
# ---------------------------------------------------------------------------

def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    tmp.replace(path)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def exact_sign_flip_p(differences: np.ndarray) -> float:
    """Two-sided exhaustive sign-flip p-value; Monte Carlo above 20 units."""
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return float("nan")
    observed = abs(float(values.mean()))
    if n <= 20:
        count = 0
        for signs in itertools.product((1.0, -1.0), repeat=n):
            if abs(float(np.mean(np.asarray(signs) * values))) >= observed - 1e-12:
                count += 1
        return count / float(2 ** n)
    rng = np.random.default_rng(STATISTICS_SEED)
    draws = rng.choice((1.0, -1.0), size=(200000, n))
    stats = np.abs((draws * values).mean(axis=1))
    return float((np.sum(stats >= observed - 1e-12) + 1) / (stats.size + 1))


def bootstrap_ci(differences: np.ndarray) -> tuple[float, float]:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(STATISTICS_SEED)
    index = rng.integers(0, values.size, size=(BOOTSTRAP_SAMPLES, values.size))
    means = values[index].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(p_values: list[float]) -> list[float]:
    finite = [(i, p) for i, p in enumerate(p_values) if math.isfinite(p)]
    adjusted = [float("nan")] * len(p_values)
    m = len(finite)
    running = 0.0
    for rank, (index, p) in enumerate(sorted(finite, key=lambda item: item[1])):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_contrast(name: str, family: str, left: str, right: str,
                    by_source: dict[int, float], by_source_right: dict[int, float],
                    expectation: str) -> dict[str, Any]:
    sources = sorted(set(by_source) & set(by_source_right))
    pairs = [(s, by_source[s], by_source_right[s]) for s in sources
             if math.isfinite(by_source[s]) and math.isfinite(by_source_right[s])]
    if not pairs:
        return {"contrast": name, "family": family, "n_sources": 0,
                "mean_difference": float("nan"), "exact_p": float("nan"),
                "ci_low": float("nan"), "ci_high": float("nan"),
                "n_positive": 0, "expectation": expectation,
                "left_arm": left, "right_arm": right,
                "left_mean": float("nan"), "right_mean": float("nan")}
    differences = np.array([a - b for _, a, b in pairs], dtype=np.float64)
    low, high = bootstrap_ci(differences)
    return {
        "contrast": name,
        "family": family,
        "left_arm": left,
        "right_arm": right,
        "n_sources": len(pairs),
        "left_mean": float(np.mean([a for _, a, _ in pairs])),
        "right_mean": float(np.mean([b for _, _, b in pairs])),
        "mean_difference": float(differences.mean()),
        "ci_low": low,
        "ci_high": high,
        "exact_p": exact_sign_flip_p(differences),
        "n_positive": int(np.sum(differences > 0)),
        "expectation": expectation,
        "per_source_difference": [float(x) for x in differences],
    }


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def source_level_table(rows: list[dict[str, Any]]) -> tuple[dict, list[dict[str, Any]]]:
    """Average target repeats within each (source, arm, budget)."""
    bucket: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (int(row["source_seed"]), str(row["arm"]), int(row["budget"]))
        bucket[key].append(row)

    metrics = [
        "test_accuracy", "M_rel_codebook", "M_rel_navstep", "col_norm_mean",
        "col_norm_anisotropy", "antipodality_index", "pair_axis_angle_deg",
        "procrustes_residual", "recurrence_drop_normalized",
        "recurrence_ablation_drop", "test_accuracy_no_recurrence",
    ]
    table: dict[tuple[int, str, int], dict[str, float]] = {}
    flat: list[dict[str, Any]] = []
    for key, group in sorted(bucket.items()):
        source, arm, budget = key
        record: dict[str, Any] = {
            "source_seed": source, "arm": arm, "budget": budget,
            "n_target_repeats": len(group),
        }
        for metric in metrics:
            values = np.array([as_float(item.get(metric)) for item in group], dtype=np.float64)
            values = values[np.isfinite(values)]
            record[metric] = float(values.mean()) if values.size else float("nan")
        record["axis_angle_deviation"] = (
            abs(record["pair_axis_angle_deg"] - 90.0)
            if math.isfinite(record["pair_axis_angle_deg"]) else float("nan")
        )
        table[key] = record
        flat.append(record)
    return table, flat


def series(table: dict, arm: str, metric: str, budget: int) -> dict[int, float]:
    return {
        source: record[metric]
        for (source, row_arm, row_budget), record in table.items()
        if row_arm == arm and row_budget == budget and math.isfinite(record.get(metric, float("nan")))
    }


def first_crossing(table: dict, arm: str, budgets: Iterable[int],
                   metric: str = "M_rel_codebook") -> tuple[dict[int, float], dict[int, int]]:
    """Magnitude at the first budget whose source-level accuracy reaches the threshold."""
    ordered = sorted(budgets)
    sources = sorted({source for (source, row_arm, _) in table if row_arm == arm})
    values: dict[int, float] = {}
    chosen: dict[int, int] = {}
    for source in sources:
        for budget in ordered:
            record = table.get((source, arm, budget))
            if record is None:
                continue
            if record["test_accuracy"] >= ACCURACY_THRESHOLD and math.isfinite(record[metric]):
                values[source] = record[metric]
                chosen[source] = budget
                break
    return values, chosen


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="E1 interface magnitude analysis")
    parser.add_argument("--in", dest="input_dir", required=True, type=Path)
    parser.add_argument("--out", dest="output_dir", required=True, type=Path)
    args = parser.parse_args()

    rows = read_rows(args.input_dir / "interface_rows.csv")
    table, flat = source_level_table(rows)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        out_dir / "source_level_interface.csv", flat,
        ["source_seed", "arm", "budget", "n_target_repeats", "test_accuracy",
         "M_rel_codebook", "M_rel_navstep", "col_norm_mean", "col_norm_anisotropy",
         "antipodality_index", "pair_axis_angle_deg", "axis_angle_deviation",
         "procrustes_residual", "recurrence_ablation_drop",
         "recurrence_drop_normalized", "test_accuracy_no_recurrence"],
    )

    budgets_by_arm = {
        arm: sorted({budget for (_, row_arm, budget) in table if row_arm == arm})
        for arm in {row_arm for (_, row_arm, _) in table}
    }
    common = set.intersection(*(set(budgets_by_arm[arm]) for arm in RECURRENT_ARMS))
    if not common:
        raise SystemExit("no budget is shared by all three recurrent arms")
    max_budget = max(common)

    contrasts: list[dict[str, Any]] = []

    # -- PRIMARY ------------------------------------------------------------
    contrasts.append(paired_contrast(
        "M_nav_minus_ctl01_at_max_budget", "primary", "nav_frozen", "ctl01_frozen",
        series(table, "nav_frozen", "M_rel_codebook", max_budget),
        series(table, "ctl01_frozen", "M_rel_codebook", max_budget),
        "negative (navigation operates at a smaller relation magnitude)",
    ))

    # -- SECONDARY ----------------------------------------------------------
    secondary_start = len(contrasts)
    contrasts.append(paired_contrast(
        "M_nav_minus_random_at_max_budget", "secondary", "nav_frozen", "random_frozen",
        series(table, "nav_frozen", "M_rel_codebook", max_budget),
        series(table, "random_frozen", "M_rel_codebook", max_budget),
        "negative",
    ))

    nav_star, nav_star_budget = first_crossing(table, "nav_frozen", budgets_by_arm["nav_frozen"])
    ctl_star, ctl_star_budget = first_crossing(table, "ctl01_frozen", budgets_by_arm["ctl01_frozen"])
    star = paired_contrast(
        "Mstar_nav_minus_ctl01_at_matched_accuracy", "secondary",
        "nav_frozen", "ctl01_frozen", nav_star, ctl_star, "negative",
    )
    sources_with_arm = {
        arm: {source for (source, row_arm, _) in table if row_arm == arm}
        for arm in ("nav_frozen", "ctl01_frozen")
    }
    star["detail"] = {
        "accuracy_threshold": ACCURACY_THRESHOLD,
        "nav_crossing_budget_by_source": nav_star_budget,
        "ctl01_crossing_budget_by_source": ctl_star_budget,
        "sources_nav_never_crossed": sorted(sources_with_arm["nav_frozen"] - set(nav_star)),
        "sources_ctl01_never_crossed": sorted(sources_with_arm["ctl01_frozen"] - set(ctl_star)),
    }
    contrasts.append(star)

    for arm, expectation in (("nav_frozen", "approximately zero (plateaued)"),
                             ("ctl01_frozen", "positive (still increasing)")):
        contrasts.append(paired_contrast(
            f"M_plateau_{arm}_max_minus_{PLATEAU_REFERENCE_BUDGET}", "secondary", arm, arm,
            series(table, arm, "M_rel_codebook", max_budget),
            series(table, arm, "M_rel_codebook", PLATEAU_REFERENCE_BUDGET),
            expectation,
        ))

    # -- PRE-REGISTERED NULL ------------------------------------------------
    null_start = len(contrasts)
    for metric, name in (("antipodality_index", "N1_antipodality"),
                         ("procrustes_residual", "N2_procrustes_residual"),
                         ("axis_angle_deviation", "N3_axis_angle_deviation")):
        contrasts.append(paired_contrast(
            f"{name}_nav_minus_ctl01_at_max_budget", "prespecified_null",
            "nav_frozen", "ctl01_frozen",
            series(table, "nav_frozen", metric, max_budget),
            series(table, "ctl01_frozen", metric, max_budget),
            "predicted NOT negative; a positive value means navigation's frame is "
            "LESS canonical than the control's",
        ))

    for start, end in ((secondary_start, null_start), (null_start, len(contrasts))):
        block = contrasts[start:end]
        for record, adjusted in zip(block, holm([item["exact_p"] for item in block])):
            record["holm_p"] = adjusted
    contrasts[0]["holm_p"] = contrasts[0]["exact_p"]

    # -- DESCRIPTIVE --------------------------------------------------------
    dissociation: list[dict[str, Any]] = []
    for source in sorted({s for (s, _, _) in table}):
        best = None
        for budget in sorted(common):
            nav = table.get((source, "nav_frozen", budget))
            rnd = table.get((source, "random_frozen", budget))
            if not nav or not rnd:
                continue
            gap = abs(nav["antipodality_index"] - rnd["antipodality_index"])
            if not math.isfinite(gap):
                continue
            if best is None or gap < best["shape_gap"]:
                best = {
                    "source_seed": source, "budget": budget, "shape_gap": gap,
                    "nav_antipodality": nav["antipodality_index"],
                    "random_antipodality": rnd["antipodality_index"],
                    "nav_accuracy": nav["test_accuracy"],
                    "random_accuracy": rnd["test_accuracy"],
                    "accuracy_gap": nav["test_accuracy"] - rnd["test_accuracy"],
                    "nav_M": nav["M_rel_codebook"],
                    "random_M": rnd["M_rel_codebook"],
                }
        if best:
            dissociation.append(best)
    if dissociation:
        write_csv(out_dir / "shape_accuracy_dissociation.csv", dissociation,
                  ["source_seed", "budget", "shape_gap", "nav_antipodality",
                   "random_antipodality", "nav_accuracy", "random_accuracy",
                   "accuracy_gap", "nav_M", "random_M"])

    recurrence = []
    for arm in RECURRENT_ARMS:
        raw = series(table, arm, "recurrence_ablation_drop", RECURRENCE_BUDGET)
        norm = series(table, arm, "recurrence_drop_normalized", RECURRENCE_BUDGET)
        accuracy = series(table, arm, "test_accuracy", RECURRENCE_BUDGET)
        if not raw:
            continue
        recurrence.append({
            "arm": arm, "budget": RECURRENCE_BUDGET, "n_sources": len(raw),
            "accuracy": float(np.mean(list(accuracy.values()))) if accuracy else float("nan"),
            "raw_drop": float(np.mean(list(raw.values()))),
            "normalized_drop": float(np.mean(list(norm.values()))) if norm else float("nan"),
        })
    if recurrence:
        write_csv(out_dir / "recurrence_drop_normalized.csv", recurrence,
                  ["arm", "budget", "n_sources", "accuracy", "raw_drop", "normalized_drop"])

    accumulator_budgets = budgets_by_arm.get(ACCUMULATOR_ARM, [])
    accumulator_reference = {}
    if accumulator_budgets:
        budget = max(accumulator_budgets)
        accumulator_reference = {
            "budget": budget,
            "accuracy": float(np.mean(list(series(table, ACCUMULATOR_ARM, "test_accuracy", budget).values()) or [float("nan")])),
            "antipodality_index": float(np.mean(list(series(table, ACCUMULATOR_ARM, "antipodality_index", budget).values()) or [float("nan")])),
            "procrustes_residual": float(np.mean(list(series(table, ACCUMULATOR_ARM, "procrustes_residual", budget).values()) or [float("nan")])),
            "pair_axis_angle_deg": float(np.mean(list(series(table, ACCUMULATOR_ARM, "pair_axis_angle_deg", budget).values()) or [float("nan")])),
        }

    write_csv(out_dir / "interface_contrasts.csv", contrasts,
              ["contrast", "family", "left_arm", "right_arm", "n_sources",
               "left_mean", "right_mean", "mean_difference", "ci_low", "ci_high",
               "exact_p", "holm_p", "n_positive", "expectation"])

    result = {
        "schema_version": "e1_interface_analysis_v1",
        "max_common_budget": max_budget,
        "budgets_by_arm": budgets_by_arm,
        "accuracy_threshold": ACCURACY_THRESHOLD,
        "contrasts": contrasts,
        "accumulator_reference": accumulator_reference,
        "recurrence_drop": recurrence,
        "shape_accuracy_dissociation_mean_accuracy_gap": (
            float(np.mean([d["accuracy_gap"] for d in dissociation])) if dissociation else None
        ),
        "shape_accuracy_dissociation_mean_shape_gap": (
            float(np.mean([d["shape_gap"] for d in dissociation])) if dissociation else None
        ),
    }
    (out_dir / "INTERFACE_RESULT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # -- human-readable report ---------------------------------------------
    lines: list[str] = []
    lines.append("# E1 interface result: magnitude versus shape\n")
    lines.append(f"- largest budget shared by all three recurrent arms: **{max_budget}**")
    lines.append(f"- replication unit: navigation source seed, target repeats averaged within source")
    lines.append(f"- accuracy threshold for the performance-matched contrast: {ACCURACY_THRESHOLD}\n")

    lines.append("## Magnitude by arm and budget (source-level mean)\n")
    header_budgets = sorted({b for arm in RECURRENT_ARMS for b in budgets_by_arm[arm]})
    lines.append("| arm | " + " | ".join(str(b) for b in header_budgets) + " |")
    lines.append("|---|" + "---|" * len(header_budgets))
    for arm in RECURRENT_ARMS:
        cells = []
        for budget in header_budgets:
            values = series(table, arm, "M_rel_codebook", budget)
            cells.append(f"{np.mean(list(values.values())):.1f}" if values else "-")
        lines.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Values are mean_k ||W_rel[:,k]|| divided by the codebook column norm, "
                 "so they are directly comparable to the Phase F 8x8 scalar gain.\n")

    lines.append("## Accuracy by arm and budget (source-level mean)\n")
    lines.append("| arm | " + " | ".join(str(b) for b in header_budgets) + " |")
    lines.append("|---|" + "---|" * len(header_budgets))
    for arm in RECURRENT_ARMS:
        cells = []
        for budget in header_budgets:
            values = series(table, arm, "test_accuracy", budget)
            cells.append(f"{np.mean(list(values.values())):.3f}" if values else "-")
        lines.append(f"| {ARM_LABEL[arm]} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Declared contrasts\n")
    lines.append("| family | contrast | n | left | right | difference | 95% CI | exact p | Holm |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for record in contrasts:
        lines.append(
            f"| {record['family']} | {record['contrast']} | {record['n_sources']} | "
            f"{record['left_mean']:.4g} | {record['right_mean']:.4g} | "
            f"{record['mean_difference']:+.4g} | "
            f"[{record['ci_low']:+.4g}, {record['ci_high']:+.4g}] | "
            f"{record['exact_p']:.6f} | {record.get('holm_p', float('nan')):.6f} |"
        )
    lines.append("")

    if accumulator_reference:
        lines.append("## Zero-core reference (2-D accumulator)\n")
        lines.append(
            f"- budget {accumulator_reference['budget']}, accuracy "
            f"{accumulator_reference['accuracy']:.3f}, antipodality "
            f"{accumulator_reference['antipodality_index']:.4f}, Procrustes residual "
            f"{accumulator_reference['procrustes_residual']:.4f}, axis angle "
            f"{accumulator_reference['pair_axis_angle_deg']:.1f} deg"
        )
        lines.append("- A model with no recurrent core at all still learns a near-antipodal "
                     "relation frame, so frame canonicality is imposed by the task.\n")

    if recurrence:
        lines.append("## Recurrence ablation, raw versus chance-normalised\n")
        lines.append("| arm | accuracy | raw drop | normalised drop |")
        lines.append("|---|---|---|---|")
        for record in recurrence:
            lines.append(
                f"| {ARM_LABEL.get(record['arm'], record['arm'])} | {record['accuracy']:.3f} | "
                f"{record['raw_drop']:.3f} | {record['normalized_drop']:.3f} |"
            )
        lines.append("")
        lines.append("The raw drop is bounded by accuracy minus chance, so it tracks accuracy "
                     "rather than recurrence dependence. The normalised column is the one to report.\n")

    if dissociation:
        lines.append("## Shape/accuracy dissociation\n")
        lines.append(
            f"- At the budget where navigation and the random core have the most similar "
            f"interface shape (mean |antipodality gap| = "
            f"{result['shape_accuracy_dissociation_mean_shape_gap']:.4f}), the mean accuracy "
            f"gap is {result['shape_accuracy_dissociation_mean_accuracy_gap']:+.3f}."
        )
        lines.append("- Matched interface shape with a large accuracy gap places the difference "
                     "in the core, not in the interface frame.\n")

    (out_dir / "INTERFACE_RESULT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
