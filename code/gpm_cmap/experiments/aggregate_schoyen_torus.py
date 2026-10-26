"""Aggregate automatic Schøyen-style torus results across navigation source seeds."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--seeds", default="6101,6102,6103,6104,6105,6106,6107,6108,6109,6110")
    args = ap.parse_args()
    run = Path(args.run_root)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out = run / "aggregate" / "torus_auto"
    out.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    for seed in seeds:
        d = run / "sources" / f"nav_seed_{seed}" / "artifacts" / "torus_topology"
        summary_path = d / "torus_analysis_summary.json"
        if not summary_path.exists():
            rows.append({"seed": seed, "status": "missing"})
            continue
        z = read_json(summary_path)
        sel = z.get("selection") or {}
        ph = (z.get("persistent_cohomology") or {}).get("metrics") or {}
        eu = ph.get("euclidean") or {}
        co = ph.get("cosine") or {}
        rows.append({
            "seed": seed,
            "status": z.get("status"),
            "is_torus_like": z.get("is_torus_like"),
            "selected_clusters": ",".join(map(str, sel.get("selected_clusters") or [])),
            "n_selected_units": sel.get("n_selected_units"),
            "robust_min_gap_score": sel.get("robust_min_gap_score"),
            "euclidean_H1_2_to_3": eu.get("H1_second_to_third"),
            "euclidean_H2_1_to_2": eu.get("H2_first_to_second"),
            "cosine_H1_2_to_3": co.get("H1_second_to_third"),
            "cosine_H2_1_to_2": co.get("H2_first_to_second"),
            "cluster_coverage": (z.get("data") or {}).get("cluster_coverage_fraction"),
            "tda_coverage": (z.get("data") or {}).get("tda_coverage_fraction"),
            "tda_mean_pi_error": (z.get("data") or {}).get("tda_mean_pi_error"),
            "tda_final_pi_error": (z.get("data") or {}).get("tda_final_pi_error"),
            "artifact_dir": str(d),
        })
        cand = d / "candidate_topology.csv"
        if cand.exists():
            with cand.open("r", encoding="utf-8-sig", newline="") as f:
                for r in csv.DictReader(f):
                    r = dict(r); r["seed"] = seed; candidate_rows.append(r)

    fields = []
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with (out / "torus_auto_summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    if candidate_rows:
        fields2 = []
        for r in candidate_rows:
            for k in r:
                if k not in fields2: fields2.append(k)
        with (out / "torus_auto_candidates.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields2); w.writeheader(); w.writerows(candidate_rows)

    usable = [r for r in rows if r.get("robust_min_gap_score") is not None]
    if usable:
        x = np.arange(len(usable))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        vals = [float(r["robust_min_gap_score"]) for r in usable]
        ax.bar(x, vals)
        ax.axhline(1.5, linestyle="--", linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels([str(r["seed"]) for r in usable])
        ax.set_ylabel("Robust minimum H1/H2 gap across both metrics")
        ax.set_xlabel("Navigation seed")
        ax.set_title("Automatic hidden-layer torus discovery")
        fig.tight_layout(); fig.savefig(out / "torus_auto_robust_scores.png", dpi=220); plt.close(fig)

    print("Wrote:", out / "torus_auto_summary.csv")
    if candidate_rows: print("Wrote:", out / "torus_auto_candidates.csv")


if __name__ == "__main__":
    main()
