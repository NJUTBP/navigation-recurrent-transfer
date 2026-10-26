#!/usr/bin/env python3
"""Export the frozen figure source data under the FINAL manuscript figure numbers.

The plotting builders read internal folder names (for example
``figure_source_data/SupplementaryFigureS8`` holds the data drawn as
Supplementary Figure S7).  This script does not touch those folders.  It writes a
separate submission copy in which every folder is named after the figure it
supports::

    aggregate_full/source_data_final/Figure2 ... Figure7
    aggregate_full/source_data_final/SupplementaryFigureS1 ... SupplementaryFigureS11

Figure 1 is an author-drawn schematic and has no source data.
In Figure 7 the two tables whose panel letters changed in v2.6 are renamed to
the final letters (C = path-length profile, D = autonomous retention).

Usage:  python export_source_data_final.py [--aggregate PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MAPPING = {
    "Figure2": (["figure_source_data/Figure2"], "Navigation pretraining (builder Figure2)"),
    "Figure3": (["figure_source_data/Figure3"], "Frozen-core transfer on the 4 x 4 task"),
    "Figure4": (["figure_source_data/Figure4"], "Relation-input transformations"),
    "Figure5": (["figure_source_data/Figure5"], "Arbitrary relation symbols"),
    "Figure6": (["figure_source_data/FigureA5"], "Extended 8 x 8 task (tables shared with S5 and S6)"),
    "Figure7": (["figure_source_data/Figure7"], "Recurrent perturbation and rescaling (C/D renamed to final letters)"),
    "SupplementaryFigureS1": (["figure_source_data/FigureA1"], "Delayed-reconstruction pretraining"),
    "SupplementaryFigureS2": (["figure_source_data/FigureA3"], "Topology, reduced-dynamics and path-invariance tests"),
    "SupplementaryFigureS3": (["figure_source_data/FigureA4"], "Torus-like topology visualization"),
    "SupplementaryFigureS4": (["figure_source_data/FigureA2"], "Relation and endpoint mapping controls"),
    "SupplementaryFigureS5": (["figure_source_data/FigureA5"], "Extended 8 x 8 controls (panel E: all_per_length.csv, held-out splits)"),
    "SupplementaryFigureS6": (["figure_source_data/FigureA5", "hard8x8_phaseF_followup/all_dataset_audit.csv"],
                              "Optimization, shortcut-baseline and split diagnostics"),
    "SupplementaryFigureS7": (["figure_source_data/SupplementaryFigureS8"], "Extended optimization, gain calibration and additive controls"),
    "SupplementaryFigureS8": (["figure_source_data/SupplementaryFigureS9"], "Multi-step state maintenance and error accumulation"),
    "SupplementaryFigureS9": (["figure_source_data/SupplementaryFigureS11"], "Refitting of navigation-specific mappings"),
    "SupplementaryFigureS10": (["figure_source_data/FigureA6"], "Grid-like unit fraction versus transfer"),
    "SupplementaryFigureS11": (["figure_source_data/SupplementaryFigureS10"], "Prespecified source-measure analyses"),
    "SupplementaryFigureS12": (["figure_source_data/SupplementaryFigureS12"], "Rescaling interventions by path length and source"),
}
FIGURE7_RENAME = {
    "panelC_retention_dose.csv": "panelD_retention_dose.csv",
    "panelD_horizon_profile.csv": "panelC_horizon_profile.csv",
}
FIGURE7_NOTE = (
    "Final panel letters (manuscript v2.6):\n"
    "  C = path-length profile at rho = 0, 0.05, 0.10 -> panelC_horizon_profile.csv\n"
    "  D = autonomous retention <kappa>(rho)           -> panelD_retention_dose.csv\n"
    "The provenance files PANEL_CONTRACT.json, SOURCE_MANIFEST.json and\n"
    "figure7_statistics.json were written by the frozen pipeline and keep the\n"
    "builder-era letters: their 'panelC' is final panel D and their 'panelD' is\n"
    "final panel C.\n"
)


def main() -> int:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aggregate", type=Path, default=root / "aggregate_full")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    agg = args.aggregate.resolve()
    out = (args.out or agg / "source_data_final").resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    manifest = {}
    for final, (sources, note) in MAPPING.items():
        dst = out / final
        dst.mkdir()
        for rel in sources:
            src = agg / rel
            if not src.exists():
                raise FileNotFoundError(f"{final}: missing {src}")
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst / src.name)
        if final == "Figure7":
            for old, new in FIGURE7_RENAME.items():
                (dst / old).rename(dst / new)
            (dst / "PANEL_LETTERS_NOTE.txt").write_text(FIGURE7_NOTE, encoding="utf-8")
        files = sorted(str(p.relative_to(dst)).replace("\\", "/") for p in dst.rglob("*") if p.is_file())
        manifest[final] = {"sources": sources, "note": note, "n_files": len(files), "files": files}
    prov = out / "_provenance"
    prov.mkdir()
    for name in ("SOURCE_DATA_MANIFEST.json", "MECHANISM_FIGURE_SOURCE_FREEZE.json"):
        p = agg / "figure_source_data" / name
        if p.exists():
            shutil.copy2(p, prov / name)
    (out / "EXPORT_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    lines = ["Source data organized by FINAL manuscript figure number.", "",
             f"{'Final figure':26s} {'Builder data':58s} Content"]
    for final, (sources, note) in MAPPING.items():
        lines.append(f"{final:26s} {' + '.join(sources):58s} {note}")
    lines += ["", "Figure 1 is an author-drawn schematic and has no source data."]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exported {len(MAPPING)} figures to {out}")
    for final, m in manifest.items():
        print(f"  {final:24s} {m['n_files']:4d} files  <- {' + '.join(m['sources'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
