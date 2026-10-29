#!/usr/bin/env python3
"""Redraw the computational figures of the article from the figure source data.

    python figures/redraw_all_figures.py --aggregate data/aggregate_full --out outputs/figures
    python figures/redraw_all_figures.py --figures Figure7,SupplementaryFigureS12

The builders read only the frozen tables in <aggregate>/figure_source_data and
<aggregate>/hard8x8_phaseF_followup; no experiment, aggregation or statistical test is rerun.
Each builder writes its files to <out>/_builder_outputs under its internal name; the files are
then copied to <out> under the figure numbers of the article (Figure2 ... Figure7,
SupplementaryFigureS1 ... SupplementaryFigureS12) as defined in figure_numbering.py.
Figure 1 is an author-drawn schematic (figures/figure1) and is not generated.
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
import traceback
from pathlib import Path

import pandas as pd

FIGURES_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIGURES_ROOT.parent
sys.path.insert(0, str(FIGURES_ROOT))

from figure_numbering import PAPER_FIGURES, in_paper_order  # noqa: E402

COPY_EXTENSIONS = (".png", ".svg", ".pdf", ".tiff", ".eps")
REQUIRED_EXTENSIONS = (".png", ".svg", ".pdf", ".tiff")
PHASEF_FIGURES = {"Figure6", "SupplementaryFigureS5", "SupplementaryFigureS6"}


def die(message: str) -> None:
    raise RuntimeError(message)


def source_pairs_from_aggregate(aggregate: Path) -> list[dict]:
    """Infer the independent navigation source cohort from the aggregate."""
    candidates = [
        aggregate / "source_qualification.csv",
        aggregate / "figure_source_data" / "FigureA3" / "source_qualification.csv",
    ]
    for p in candidates:
        if not p.exists():
            continue
        q = pd.read_csv(p)
        if "source_seed" not in q.columns:
            continue
        q = q.drop_duplicates("source_seed").sort_values("source_seed")
        result = []
        for _, row in q.iterrows():
            item = {"source_seed": int(row["source_seed"])}
            if "control_seed" in q.columns and pd.notna(row.get("control_seed")):
                item["control_seed"] = int(row["control_seed"])
            else:
                item["control_seed"] = 7000 + len(result) + 1
            result.append(item)
        if result:
            return result

    nav = aggregate / "figure_source_data" / "Figure2" / "navigation_history.csv"
    if nav.exists():
        q = pd.read_csv(nav)
        if "source_seed" in q.columns:
            seeds = sorted({int(x) for x in q["source_seed"].dropna().tolist()})
            return [
                {"source_seed": s, "control_seed": 7000 + i + 1}
                for i, s in enumerate(seeds)
            ]

    die("Could not infer source_seed list from aggregate_full.")


def copy_formats(srcstem: Path, dststem: Path) -> dict[str, str]:
    copied = {}
    for ext in COPY_EXTENSIONS:
        src = srcstem.with_suffix(ext)
        if src.exists():
            dst = dststem.with_suffix(ext)
            shutil.copy2(src, dst)
            copied[ext.lstrip(".")] = str(dst.resolve())
    return copied


def validate_final_outputs(figures_dir: Path, names) -> None:
    missing = []
    for stem in names:
        for ext in REQUIRED_EXTENSIONS:
            p = figures_dir / f"{stem}{ext}"
            if not p.exists() or p.stat().st_size == 0:
                missing.append(str(p))
    if missing:
        die("Final figure validation failed. Missing/empty files:\n  " + "\n  ".join(missing))


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aggregate", type=Path, default=REPOSITORY_ROOT / "data" / "aggregate_full",
                    help="figure source data (default: data/aggregate_full of this repository)")
    ap.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "outputs" / "figures",
                    help="output directory (default: outputs/figures); must not lie inside the source data")
    ap.add_argument("--figures", default=None,
                    help="comma-separated figure names, e.g. Figure6,SupplementaryFigureS12 (default: all)")
    args = ap.parse_args()

    known = [figure.number for figure in in_paper_order()]
    wanted = None if not args.figures else {x.strip() for x in args.figures.split(",") if x.strip()}
    if wanted:
        unknown = wanted - set(known)
        if unknown:
            die(f"Unknown figure names: {sorted(unknown)}. Valid names: {', '.join(known)}")
    selected = [figure for figure in PAPER_FIGURES if wanted is None or figure.number in wanted]

    aggregate = args.aggregate.resolve()
    source_data = aggregate / "figure_source_data"
    figures_dir = args.out.resolve()
    raw_dir = figures_dir / "_builder_outputs"

    print("=" * 76)
    print("Redrawing the article figures from the figure source data")
    print(f"Aggregate  : {aggregate}")
    print(f"Output     : {figures_dir}")
    print("=" * 76)

    if is_within(figures_dir, aggregate) or is_within(figures_dir, REPOSITORY_ROOT / "data"):
        die(f"--out {figures_dir} lies inside the figure source data; choose a separate folder such as outputs/figures")
    if not aggregate.is_dir():
        die(f"Missing directory: {aggregate}")
    if not source_data.is_dir():
        die(f"Missing source-data directory: {source_data}")
    if any(figure.number in PHASEF_FIGURES for figure in selected) and not (aggregate / "hard8x8_phaseF_followup").is_dir():
        die(f"Missing 8 x 8 (Phase F) tables: {aggregate / 'hard8x8_phaseF_followup'}")
    for figure in selected:
        for rel in figure.source_data:
            if not (aggregate / rel).exists():
                die(f"{figure.number}: missing source data {aggregate / rel}")

    from paper_figures.common import FigureContext, write_json

    context_builders = [figure for figure in selected if figure.function == "build"]
    source_pairs = source_pairs_from_aggregate(aggregate) if context_builders else []
    n_sources = len(source_pairs)
    print(f"[info] Navigation source cohort: n={n_sources}")

    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "config": {
            "run_id": "redraw_all_figures",
            "checkpoint_policy": "best_validation",
            "schoyen_topology": {
                "h1_gap_threshold": 1.5,
                "h2_gap_threshold": 1.5,
            },
        },
        "source_pairs": source_pairs,
        "run_root": ".",
    }
    ctx = FigureContext(
        project_root=FIGURES_ROOT,
        plan_path=FIGURES_ROOT / "_local_plot_context.json",
        aggregate_dir=aggregate,
        run_dir=FIGURES_ROOT,
        source_data_dir=source_data,
        figures_dir=raw_dir,
        plan=plan,
    )

    # The 8 x 8 and mechanism builders read these two variables when they are imported.
    os.environ["NRT_FIGURE_AGGREGATE_DIR"] = str(aggregate)
    os.environ["NRT_FIGURE_OUTPUT_DIR"] = str(raw_dir)
    for figure in selected:
        print(f"\n[build] {figure.builder_output}")
        module = importlib.import_module(figure.builder)
        if figure.function == "build":
            module.build(ctx)
        else:
            getattr(module, figure.function)()
        print(f"[done] {figure.builder_output}")

    final_outputs = {}
    print("\n[number] Applying the figure numbers of the article")
    for figure in in_paper_order(selected):
        copied = copy_formats(raw_dir / figure.builder_output, figures_dir / figure.number)
        if not copied:
            die(f"No output formats were found for builder output: {raw_dir / figure.builder_output}")
        final_outputs[figure.number] = copied
        print(f"  {figure.builder_output:24s} -> {figure.number}")

    validate_final_outputs(figures_dir, [figure.number for figure in selected])

    manifest = {
        "run_id": plan["config"]["run_id"],
        "n_navigation_sources": n_sources,
        "checkpoint_policy": "best_validation",
        "external_figure1": "author-drawn schematic; not generated by code",
        "source_data_dir": str(source_data),
        "hard8_followup_dir": str(aggregate / "hard8x8_phaseF_followup"),
        "figures_dir": str(figures_dir),
        "raw_builder_output_dir": str(raw_dir),
        "builder_outputs": {figure.number: figure.builder_output for figure in in_paper_order(selected)},
        "figures": final_outputs,
    }
    write_json(figures_dir / "figure_manifest.json", manifest)

    print("\n" + "=" * 76)
    print("SUCCESS")
    print(f"Figures written to {figures_dir}: " + ", ".join(figure.number for figure in in_paper_order(selected)))
    print("Figure 1 is an author-drawn schematic and is not generated.")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("\n" + "!" * 76, file=sys.stderr)
        print(f"FAILED: {exc}", file=sys.stderr)
        print("!" * 76, file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
