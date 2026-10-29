#!/usr/bin/env python3
"""Copy the figure source data into folders named by the figure numbers of the article.

The builders read folders named by their internal identifiers (for example
figure_source_data/SupplementaryFigureS8 holds the data of Supplementary Figure S7).  This script
leaves those folders untouched and writes a separate copy in which every folder is named after the
figure it supports (Figure2 ... Figure7, SupplementaryFigureS1 ... SupplementaryFigureS12), using
the mapping in figure_numbering.py.  Figure 1 is an author-drawn schematic and has no source data.
In the Figure 7 copy, the two tables whose file names carry the earlier panel letters are renamed
to the final letters (C = path-length profile, D = autonomous retention).

    python figures/export_source_data_final.py --aggregate data/aggregate_full --out outputs/source_data_final
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

FIGURES_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = FIGURES_ROOT.parent
sys.path.insert(0, str(FIGURES_ROOT))

from figure_numbering import in_paper_order  # noqa: E402

FIGURE7_RENAME = {
    "panelC_retention_dose.csv": "panelD_retention_dose.csv",
    "panelD_horizon_profile.csv": "panelC_horizon_profile.csv",
}
FIGURE7_NOTE = (
    "Final panel letters:\n"
    "  C = path-length profile at rho = 0, 0.05, 0.10 -> panelC_horizon_profile.csv\n"
    "  D = autonomous retention <kappa>(rho)           -> panelD_retention_dose.csv\n"
    "The provenance files PANEL_CONTRACT.json, SOURCE_MANIFEST.json and\n"
    "figure7_statistics.json were written by the frozen pipeline and keep the\n"
    "builder-era letters: their 'panelC' is final panel D and their 'panelD' is\n"
    "final panel C.\n"
)
EXPORT_MARKER = "EXPORT_MANIFEST.json"


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
    ap.add_argument("--out", type=Path, default=REPOSITORY_ROOT / "outputs" / "source_data_final",
                    help="new export folder (default: outputs/source_data_final)")
    args = ap.parse_args()
    agg = args.aggregate.resolve()
    out = args.out.resolve()
    if not (agg / "figure_source_data").is_dir():
        raise SystemExit(f"Missing figure source data: {agg / 'figure_source_data'}")
    if is_within(out, agg) or is_within(agg, out):
        raise SystemExit(f"--out {out} overlaps the figure source data {agg}; choose a separate folder")
    if out.exists():
        if not (out / EXPORT_MARKER).is_file():
            raise SystemExit(f"{out} exists and is not a previous export (no {EXPORT_MARKER}); "
                             "remove it or choose another --out")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    manifest = {}
    for figure in in_paper_order():
        dst = out / figure.number
        dst.mkdir()
        for rel in figure.source_data:
            src = agg / rel
            if not src.exists():
                raise FileNotFoundError(f"{figure.number}: missing {src}")
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst / src.name)
        if figure.number == "Figure7":
            for old, new in FIGURE7_RENAME.items():
                (dst / old).rename(dst / new)
            (dst / "PANEL_LETTERS_NOTE.txt").write_text(FIGURE7_NOTE, encoding="utf-8")
        files = sorted(str(p.relative_to(dst)).replace("\\", "/") for p in dst.rglob("*") if p.is_file())
        manifest[figure.number] = {"sources": list(figure.source_data), "note": figure.content,
                                   "n_files": len(files), "files": files}
    prov = out / "_provenance"
    prov.mkdir()
    for name in ("SOURCE_DATA_MANIFEST.json", "MECHANISM_FIGURE_SOURCE_FREEZE.json"):
        p = agg / "figure_source_data" / name
        if p.exists():
            shutil.copy2(p, prov / name)
    (out / EXPORT_MARKER).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    lines = ["Source data organized by the figure numbers of the article.", "",
             f"{'Figure':26s} {'Builder data':58s} Content"]
    for figure in in_paper_order():
        lines.append(f"{figure.number:26s} {' + '.join(figure.source_data):58s} {figure.content}")
    lines += ["", "Figure 1 is an author-drawn schematic and has no source data."]
    (out / "README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Exported {len(manifest)} figures to {out}")
    for number, entry in manifest.items():
        print(f"  {number:24s} {entry['n_files']:4d} files  <- {' + '.join(entry['sources'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
