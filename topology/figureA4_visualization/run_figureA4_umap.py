#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, socket, sys, time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description="Stage current FigureA4 source data and redraw FigureA4 only.")
    ap.add_argument("--project", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--aggregate", required=True)
    args=ap.parse_args()
    project=Path(args.project).resolve(); plan=Path(args.plan).resolve(); aggregate=Path(args.aggregate).resolve()
    sys.path.insert(0, str(project))
    from paper_figures.common import make_context
    from paper_figures.prepare_source_data import prepare
    from paper_figures.make_figureA4_topology import build
    print(f"[FigureA4] host={socket.gethostname()}", flush=True)
    print(f"[FigureA4] project={project}", flush=True)
    print(f"[FigureA4] aggregate={aggregate}", flush=True)
    print(f"[FigureA4] restarts={os.environ.get('FIGA4_UMAP_RESTARTS','8')}", flush=True)
    print(f"[FigureA4] force={os.environ.get('FIGA4_FORCE_RECOMPUTE','0')}", flush=True)
    print(f"[FigureA4] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','unset')}", flush=True)
    t=time.time(); ctx=make_context(project, plan, aggregate); prepare(ctx); paths=build(ctx)
    print(f"[FigureA4] done in {(time.time()-t)/60:.1f} min", flush=True)
    for k,v in paths.items(): print(f"  {k}: {v}", flush=True)

if __name__ == '__main__': main()
