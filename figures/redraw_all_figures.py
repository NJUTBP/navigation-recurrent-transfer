#!/usr/bin/env python3
"""One-click redraw for the CURRENT CTP manuscript figures (including mechanism Figure 7).

Put this file next to these folders::

    ROOT/
      redraw_all_figures.py
      redraw_all_figures.bat
      aggregate_full/
        figure_source_data/
        hard8x8_phaseF_followup/
      paper_figures/
      hard8_phaseF/

Then run ``python redraw_all_figures.py`` or double-click the BAT file.

The script redraws directly from the frozen source-data tables in
``aggregate_full``.  It does NOT call ``prepare_source_data.py`` and therefore
does not recompute experiments or statistics.

Current manuscript mapping
--------------------------
Main:
    Figure 2 <- paper_figures.make_figure2_navigation
    Figure 3 <- paper_figures.make_figure3_frozen_transfer
    Figure 4 <- paper_figures.make_figure4_interface
    Figure 5 <- paper_figures.make_figure5_symbols
    Figure 6 <- hard8_phaseF.make_figure6
    Figure 7 <- hard8_phaseF.make_figure7 (v2.6: panels C and D swapped)

Supplement (ordered by first citation in the current manuscript):
    S1 <- FigureA1 delayed-reconstruction pretraining
    S2 <- FigureA3 topology / reduced-dynamics / path-invariance tests
    S3 <- FigureA4 topology visualization
    S4 <- FigureA2 relation/endpoint mapping controls
    S5 <- hard8_phaseF.make_figureS5 (extended 8×8 transfer controls; v2.6 adds panel E)
    S6 <- hard8_phaseF.make_figureS6 (optimization / split diagnostics)
    S7 <- hard8_phaseF.make_figureS7 (extended optimization / gain / additive controls)
    S8 <- hard8_phaseF.make_figureS8 (multi-step state maintenance)
    S9 <- hard8_phaseF.make_figureS10 (navigation-interface refitting)
    S10 <- FigureA6 grid-like-unit-fraction / transfer associations
    S11 <- hard8_phaseF.make_figureS9 (prespecified source-measure analyses)
    S12 <- hard8_phaseF.make_figureS12 (rescaling interventions by path length and source; revision)

v2.3: S9-S11 were renumbered so that Supplementary Figures are numbered in the
order of their first citation in the Main Text.  Builder file names and the
frozen data interface are unchanged; only the final output names differ.

Figure 1 is author-drawn and is intentionally not generated.

v2.6 (paths written L_p): Figure 2A axis unit; Figures 6, 7, S6, S7 and S8 path-length
labels; Figure 7 C/D order; Supplementary Figure S5 panel E; S11 axis wording.  For a
submission copy of the source data under the final figure numbers run
``python export_source_data_final.py``.

v2.9: panel letters are lowercase (a, b, c) in every generated figure, as required
by Nonlinear Dynamics; switch PANEL_LETTER_CASE in the two nature_style.py files to
restore uppercase.
The superseded pre-Phase-F FigureA5 builder is intentionally NOT used.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# Internal builder names.  These are not manuscript figure numbers.
BASE_BUILDERS = [
    ("Figure2", "paper_figures.make_figure2_navigation"),
    ("Figure3", "paper_figures.make_figure3_frozen_transfer"),
    ("Figure4", "paper_figures.make_figure4_interface"),
    ("Figure5", "paper_figures.make_figure5_symbols"),
    ("FigureA1", "paper_figures.make_figureA1_control"),
    ("FigureA2", "paper_figures.make_figureA2_mapping_controls"),
    ("FigureA3", "paper_figures.make_figureA3_mechanism"),
    ("FigureA4", "paper_figures.make_figureA4_topology"),
    ("FigureA6", "paper_figures.make_figureA6_associations"),
]

# Builder output -> current manuscript output.
FINAL_MAPPING = {
    "Figure2": "Figure2",
    "Figure3": "Figure3",
    "Figure4": "Figure4",
    "Figure5": "Figure5",
    "Figure6": "Figure6",
    "Figure7": "Figure7",
    "FigureA1": "SupplementaryFigureS1",
    "FigureA3": "SupplementaryFigureS2",
    "FigureA4": "SupplementaryFigureS3",
    "FigureA2": "SupplementaryFigureS4",
    "SupplementaryFigureS5": "SupplementaryFigureS5",
    "SupplementaryFigureS6": "SupplementaryFigureS6",
    "SupplementaryFigureS7": "SupplementaryFigureS7",
    "SupplementaryFigureS8": "SupplementaryFigureS8",
    "SupplementaryFigureS9": "SupplementaryFigureS11",   # v2.3: source-measure analyses
    "SupplementaryFigureS10": "SupplementaryFigureS9",   # v2.3: navigation-interface refitting
    "FigureA6": "SupplementaryFigureS10",                # v2.3: grid-fraction associations
    "SupplementaryFigureS12": "SupplementaryFigureS12",  # revision: rescaling by path length and source
}

# Phase-F / mechanism builders: builder output name -> (module, function)
PHASEF_BUILDERS = [
    ("Figure6", "make_figure6", "make_figure6"),
    ("SupplementaryFigureS5", "make_figureS5", "make_s5"),
    ("SupplementaryFigureS6", "make_figureS6", "make_s6"),
    ("Figure7", "make_figure7", "make_figure7"),
    ("SupplementaryFigureS7", "make_figureS7", "make_s7"),
    ("SupplementaryFigureS8", "make_figureS8", "make_s8"),
    ("SupplementaryFigureS9", "make_figureS9", "make_s9"),
    ("SupplementaryFigureS10", "make_figureS10", "make_s10"),
    ("SupplementaryFigureS12", "make_figureS12", "make_s12"),
]

COPY_EXTENSIONS = (".png", ".svg", ".pdf", ".tiff", ".eps")
REQUIRED_EXTENSIONS = (".png", ".svg", ".pdf", ".tiff")


def die(message: str) -> None:
    raise RuntimeError(message)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        die(f"Missing required file: {path}")
    return pd.read_csv(path)


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


def archive_legacy_top_level_builder_outputs(figures_dir: Path) -> None:
    """Move stale pre-v12 builder-name files away from the manuscript-facing folder."""
    archive = figures_dir / "_legacy_builder_names"
    legacy_stems = ["FigureA1", "FigureA2", "FigureA3", "FigureA4", "FigureA5", "FigureA6"]
    moved = 0
    for stem in legacy_stems:
        for ext in COPY_EXTENSIONS:
            src = figures_dir / f"{stem}{ext}"
            if src.exists():
                archive.mkdir(parents=True, exist_ok=True)
                dst = archive / src.name
                # Keep the most recent archived copy rather than overwriting silently.
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))
                moved += 1
    if moved:
        print(f"[cleanup] Archived {moved} legacy builder-name figure files in {archive}")


def validate_final_outputs(figures_dir: Path, stems=None) -> None:
    missing = []
    for stem in (stems if stems is not None else FINAL_MAPPING.values()):
        for ext in REQUIRED_EXTENSIONS:
            p = figures_dir / f"{stem}{ext}"
            if not p.exists() or p.stat().st_size == 0:
                missing.append(str(p))
    if missing:
        die("Final figure validation failed. Missing/empty files:\n  " + "\n  ".join(missing))


def main() -> int:
    root = Path(__file__).resolve().parent

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--aggregate",
        type=Path,
        default=root / "aggregate_full",
        help="aggregate_full directory (default: sibling aggregate_full)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="final manuscript figure directory (default: <aggregate>/figures)",
    )
    ap.add_argument(
        "--figures",
        default=None,
        help="comma-separated final figure names to build (e.g. Figure6,SupplementaryFigureS12); default: all",
    )
    args = ap.parse_args()
    wanted = None if not args.figures else {x.strip() for x in args.figures.split(",") if x.strip()}
    if wanted:
        unknown = wanted - set(FINAL_MAPPING.values())
        if unknown:
            die(f"Unknown figure names: {sorted(unknown)}")
    selected = {b for b, f in FINAL_MAPPING.items() if wanted is None or f in wanted}

    aggregate = args.aggregate.resolve()
    package_dir = root / "paper_figures"
    hard8_dir = root / "hard8_phaseF"
    source_data = aggregate / "figure_source_data"
    figures_dir = (args.out or (aggregate / "figures")).resolve()
    raw_dir = figures_dir / "_builder_outputs_current_v14"

    print("=" * 76)
    print("CURRENT CTP manuscript figure redraw — one click")
    print(f"Root       : {root}")
    print(f"Aggregate  : {aggregate}")
    print(f"Output     : {figures_dir}")
    print("=" * 76)

    if not aggregate.is_dir():
        die(f"Missing directory: {aggregate}")
    if not source_data.is_dir():
        die(f"Missing source-data directory: {source_data}")
    needs_phasef = bool(selected & {"Figure6", "SupplementaryFigureS5", "SupplementaryFigureS6"})
    if needs_phasef and not (aggregate / "hard8x8_phaseF_followup").is_dir():
        die(
            "Missing final Phase-F data directory: "
            f"{aggregate / 'hard8x8_phaseF_followup'}"
        )
    if not package_dir.is_dir() or not (package_dir / "__init__.py").exists():
        die(f"Missing/current paper_figures package: {package_dir}")
    if not hard8_dir.is_dir() or not (hard8_dir / "__init__.py").exists():
        die(f"Missing hard8_phaseF package: {hard8_dir}")

    sys.path.insert(0, str(root))
    from paper_figures.common import FigureContext, write_json

    base_selected = [(n, m) for n, m in BASE_BUILDERS if n in selected]
    source_pairs = source_pairs_from_aggregate(aggregate) if base_selected else []
    n_sources = len(source_pairs)
    print(f"[info] Navigation source cohort: n={n_sources}")

    figures_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_legacy_top_level_builder_outputs(figures_dir)

    plan = {
        "config": {
            "run_id": "local_current_v14_one_click",
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
        project_root=root,
        plan_path=root / "_local_plot_context.json",
        aggregate_dir=aggregate,
        run_dir=root,
        source_data_dir=source_data,
        figures_dir=raw_dir,
        plan=plan,
    )

    outputs: dict[str, dict[str, str]] = {}

    # Current non-Phase-F builders.
    for name, module_name in base_selected:
        print(f"\n[build] {name}")
        module = __import__(module_name, fromlist=["build"])
        paths = module.build(ctx)
        outputs[name] = {k: str(Path(v).resolve()) for k, v in paths.items()}
        print(f"[done] {name}")

    # Phase-F (Figure 6, S5, S6) and mechanism (Figure 7, S7-S12) builders.  These modules read
    # CTP_AGGREGATE_DIR / CTP_FIGURE_OUT at import time, so they are imported only after setting them.
    os.environ["CTP_AGGREGATE_DIR"] = str(aggregate)
    os.environ["CTP_FIGURE_OUT"] = str(raw_dir)
    import importlib
    for name, module_name, func in PHASEF_BUILDERS:
        if name not in selected:
            continue
        print(f"\n[build] {name}")
        module = importlib.import_module(f"hard8_phaseF.{module_name}")
        getattr(module, func)()
        print(f"[done] {name}")

    # Copy builder outputs to manuscript-facing current figure names.
    final_outputs = {}
    print("\n[number] Applying current manuscript numbering")
    for src, dst in FINAL_MAPPING.items():
        if src not in selected:
            continue
        copied = copy_formats(raw_dir / src, figures_dir / dst)
        if not copied:
            die(f"No output formats were found for builder output: {raw_dir / src}")
        final_outputs[dst] = copied
        print(f"  {src:24s} -> {dst}")

    validate_final_outputs(figures_dir, [FINAL_MAPPING[s] for s in selected])

    manifest = {
        "run_id": plan["config"]["run_id"],
        "n_navigation_sources": n_sources,
        "checkpoint_policy": "best_validation",
        "external_figure1": "author-drawn schematic; not generated by code",
        "superseded_builder": "pre-Phase-F FigureA5 is intentionally not used",
        "source_data_dir": str(source_data.resolve()),
        "hard8_followup_dir": str((aggregate / "hard8x8_phaseF_followup").resolve()),
        "figures_dir": str(figures_dir.resolve()),
        "raw_builder_output_dir": str(raw_dir.resolve()),
        "current_numbering": {
            "main": ["Figure1 external", "Figure2", "Figure3", "Figure4", "Figure5", "Figure6", "Figure7"],
            "supplement": [
                "SupplementaryFigureS1",
                "SupplementaryFigureS2",
                "SupplementaryFigureS3",
                "SupplementaryFigureS4",
                "SupplementaryFigureS5",
                "SupplementaryFigureS6",
                "SupplementaryFigureS7",
                "SupplementaryFigureS8",
                "SupplementaryFigureS9",
                "SupplementaryFigureS10",
                "SupplementaryFigureS11",
                "SupplementaryFigureS12",
            ],
        },
        "figures": final_outputs,
    }
    write_json(figures_dir / "figure_manifest_current_v14.json", manifest)

    print("\n" + "=" * 76)
    print("SUCCESS")
    print("Current manuscript figures were rebuilt in:")
    print(f"  {figures_dir}")
    print("Main: Figure2–Figure7")
    print("Supplement: SupplementaryFigureS1–SupplementaryFigureS12")
    print("Figure1 is intentionally not generated.")
    print("The obsolete pre-Phase-F FigureA5 is intentionally not generated.")
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
