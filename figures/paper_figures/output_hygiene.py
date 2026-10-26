"""Remove stale paper-figure outputs before a clean redraw.

Figure1 is external and is never deleted.  Only outputs owned by this
package are removed.
"""
from __future__ import annotations
from pathlib import Path
import shutil

MANAGED=["Figure2","Figure3","Figure4","Figure5","FigureA1","FigureA2","FigureA3","FigureA4","FigureA5","FigureA6"]
STALE_DIRS=["Figure1",*MANAGED]
EXTS=[".pdf",".svg",".png",".tiff",".eps"]

def clean_managed_outputs(ctx) -> None:
    figures=Path(ctx.figures_dir);source=Path(ctx.source_data_dir)
    figures.mkdir(parents=True,exist_ok=True);source.mkdir(parents=True,exist_ok=True)
    # Remove historical per-panel directories carrying obsolete numbering.
    for name in STALE_DIRS:
        p=figures/name
        if p.is_dir(): shutil.rmtree(p)
    # Remove only generated top-level result figures; preserve external Figure1.
    for name in MANAGED:
        for ext in EXTS:
            p=figures/(name+ext)
            if p.exists(): p.unlink()
    for p in [figures/"manifests", source]:
        if p.exists() and p==figures/"manifests": shutil.rmtree(p)
    # Figure-specific source-data are always regenerated from the aggregate.
    for name in STALE_DIRS:
        p=source/name
        if p.is_dir(): shutil.rmtree(p)
    for fn in [figures/"figure_manifest.json",source/"SOURCE_DATA_MANIFEST.json"]:
        if fn.exists(): fn.unlink()
