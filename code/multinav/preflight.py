"""Fail fast on configuration, executable, dependency and source-tree problems."""
from __future__ import annotations
import argparse, importlib.util, platform, sys
from pathlib import Path
from .common import *
from .plan import validate

MODULES = ['numpy','scipy','pandas','yaml','torch','matplotlib','sklearn','networkx','PIL','tqdm','ripser','umap']

def check(root, config, *, gpu=False, allow_missing_topology=False):
    cfg=read_yaml(config); validate(cfg)
    missing=[x for x in MODULES if importlib.util.find_spec(x) is None]
    if allow_missing_topology:
        missing=[x for x in missing if x not in {'ripser','umap'}]
    problems=[]
    if sys.version_info<(3,10): problems.append('Python >=3.10 is required by this project')
    if missing: problems.append('Missing modules: '+', '.join(missing))
    import numpy as np
    # The supplied Exp8 calls np.trapezoid; do not silently change its numerical implementation.
    if not hasattr(np,'trapezoid'): problems.append('Original Exp8 needs NumPy >=2.0 (np.trapezoid)')
    if gpu:
        import torch
        if cfg['device'].startswith('cuda') and not torch.cuda.is_available(): problems.append('CUDA unavailable on this worker')
    report=dict(time_utc=now(), python=sys.executable, python_version=sys.version, platform=platform.platform(),
                missing_modules=missing, problems=problems, passed=not problems)
    if not problems:
        import torch, scipy, pandas, matplotlib
        report['versions']={x:y.__version__ for x,y in [('numpy',np),('torch',torch),('scipy',scipy),('pandas',pandas),('matplotlib',matplotlib)]}
        report['cuda_runtime']=torch.version.cuda
        report['cuda_available']=torch.cuda.is_available()
        if gpu and torch.cuda.is_available(): report['gpu_name']=torch.cuda.get_device_name(0)
    return report

def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--project'); ap.add_argument('--config',default='multinav/config.yaml')
    ap.add_argument('--gpu',action='store_true'); ap.add_argument('--output'); args=ap.parse_args()
    root=project_root(args.project); report=check(root,root/args.config,gpu=args.gpu)
    if args.output: write_json(root/args.output,report)
    import json; print(json.dumps(report,indent=2,ensure_ascii=False)); raise SystemExit(0 if report['passed'] else 2)

if __name__=='__main__': main()
