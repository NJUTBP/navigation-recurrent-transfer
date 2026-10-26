#!/usr/bin/env python3
"""Standalone Figure-A3 rerun on completed best-validation + Schøyen cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import yaml


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def read_yaml(path: Path):
    obj = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(obj, dict):
        raise TypeError(path)
    return obj


def write_yaml(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(obj, sort_keys=False, allow_unicode=True), encoding="utf-8")


def sha(path: Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def source_pair(plan: dict, seed: int):
    pair=next((x for x in plan['source_pairs'] if int(x['source_seed'])==int(seed)),None)
    if pair is None: raise ValueError(f"Source {seed} not in frozen plan")
    return pair


class Context:
    def __init__(self, run_root: Path, result_root: Path, code_root: Path, seed: int, python_bin: str):
        self.run_root=run_root.resolve(); self.result_root=result_root.resolve(); self.code_root=code_root.resolve()
        self.seed=int(seed); self.python_bin=python_bin
        self.plan=read_json(self.run_root/'plan.json'); self.pair=source_pair(self.plan,self.seed)
        self.old=self.run_root/'sources'/f'nav_seed_{self.seed}'
        self.new=self.result_root/'sources'/f'nav_seed_{self.seed}'
        self.new.mkdir(parents=True,exist_ok=True)
        self.nav=self.old/'artifacts'/'navigation'; self.nav_ckpt=self.nav/'best.pt'; self.nav_cfg=self.nav/'config_used.yaml'
        self.topology=self.new/'artifacts'/'torus_topology'
        self.threads=int(self.plan.get('config',{}).get('threads',8))

    def artifact(self,name): return self.new/'artifacts'/name
    def state(self,name): return self.new/'state'/f'{name}.json'
    def config_path(self,name,try_no): return self.new/'configs'/f'{name}_{try_no:03d}.yaml'
    def log_path(self,name,try_no): return self.new/'logs'/f'{name}_{try_no:03d}.log'

    def env(self, seed=None):
        e=os.environ.copy(); e['PYTHONPATH']=str(self.code_root)+os.pathsep+e.get('PYTHONPATH','')
        e['MPLBACKEND']='Agg'; e['PYTHONUNBUFFERED']='1'; e['PYTHONHASHSEED']=str(self.seed if seed is None else seed)
        return e

    def mark(self,name,status,**extra):
        write_json(self.state(name),dict(stage=name,status=status,updated_utc=now(),**extra))

    def run_stage(self,name:str, command_builder:Callable[[Path,int],list[str]], required:Iterable[str], *, config:Optional[dict]=None):
        state=self.state(name)
        art=self.artifact(name)
        if state.is_file():
            z=read_json(state)
            if z.get('status')=='done' and art.exists() and all(list(art.glob(p)) for p in required):
                print(f'[reuse] {name}',flush=True); return True
        tries=self.new/'attempts'/name; tries.mkdir(parents=True,exist_ok=True)
        nums=[int(x.name.split('_')[-1]) for x in tries.glob('try_*') if x.is_dir() and x.name.split('_')[-1].isdigit()]
        n=max(nums+[0])+1; out=tries/f'try_{n:03d}'; out.mkdir()
        cfg_path=None
        if config is not None:
            config=dict(config); config['output_dir']=str(out)
            cfg_path=self.config_path(name,n); write_yaml(cfg_path,config)
        cmd=command_builder(out,n)
        log=self.log_path(name,n); log.parent.mkdir(parents=True,exist_ok=True)
        self.mark(name,'running',attempt=str(out),config=None if cfg_path is None else str(cfg_path),command=cmd,started_utc=now())
        print(f'[run] {name} -> {log}',flush=True)
        try:
            with log.open('w',encoding='utf-8') as stream:
                subprocess.run(cmd,cwd=self.run_root,env=self.env(),stdout=stream,stderr=subprocess.STDOUT,check=True)
            for pattern in required:
                found=list(out.glob(pattern))
                if not found or any(not x.is_file() or x.stat().st_size==0 for x in found):
                    raise RuntimeError(f'Missing required artifact {pattern}')
            art.parent.mkdir(parents=True,exist_ok=True)
            tmp=art.with_name(art.name+'.new')
            if tmp.exists() or tmp.is_symlink(): tmp.unlink()
            tmp.symlink_to(os.path.relpath(out,art.parent)); os.replace(tmp,art)
            self.mark(name,'done',attempt=str(out),ended_utc=now(),outputs=[str(x) for p in required for x in out.glob(p)])
            return True
        except Exception as exc:
            self.mark(name,'failed',attempt=str(out),ended_utc=now(),error=repr(exc),traceback=traceback.format_exc())
            print(f'[FAILED] {name}: {exc}',flush=True); return False


def preflight(ctx: Context):
    req=[ctx.nav_ckpt,ctx.nav_cfg,ctx.topology/'torus_analysis_summary.json',ctx.topology/'torus_analysis_arrays.npz',ctx.code_root/'gpm_cmap'/'experiments'/'derive_schoyen_formal_topology.py',ctx.code_root/'gpm_cmap'/'experiments'/'analyze_source_reduced_schoyen.py']
    for target in ctx.pair['factorial_target_seeds']:
        req.append(ctx.new/'artifacts'/f'factorial_t{target}'/'checkpoint_manifest.json')
    missing=[str(x) for x in req if not x.exists()]
    if missing: raise FileNotFoundError('Missing required inputs:\n  '+'\n  '.join(missing))
    identity=ctx.new/'source_identity.json'
    if identity.exists():
        z=read_json(identity)
        if z.get('checkpoint_policy')!='best_validation': raise RuntimeError('rerun source_identity is not best_validation')
        if z.get('navigation_checkpoint_sha256')!=sha(ctx.nav_ckpt): raise RuntimeError('best.pt hash differs from rerun source identity')
    print(f'[preflight ok] source={ctx.seed} best.pt={sha(ctx.nav_ckpt)[:12]} factorial_targets={ctx.pair["factorial_target_seeds"]}')


def prepare(ctx: Context):
    preflight(ctx)
    name='formal_topology_schoyen'
    ok=ctx.run_stage(name,lambda out,n:[ctx.python_bin,'-u','-m','gpm_cmap.experiments.derive_schoyen_formal_topology','--topology-dir',str(ctx.topology),'--output-dir',str(out),'--source-seed',str(ctx.seed),'--checkpoint',str(ctx.nav_ckpt)],['formal_topology_schoyen.json','formal_topology_components.csv'])
    if not ok: return 2
    formal=read_json(ctx.artifact(name)/'formal_topology_schoyen.json')
    if formal.get('formal_topology_passed') is not True:
        ctx.mark('source_splits_schoyen','not_evaluable',reason='Schoyen Euclidean formal topology did not pass')
        print(f'[scientific nonpass] source={ctx.seed} formal topology',flush=True); return 0
    name='source_splits_schoyen'
    ok=ctx.run_stage(name,lambda out,n:[ctx.python_bin,'-u','-m','gpm_cmap.experiments.export_schoyen_source_splits','--config',str(ctx.nav_cfg),'--checkpoint',str(ctx.nav_ckpt),'--topology-dir',str(ctx.topology),'--output-dir',str(out),'--phase-trajectories','64','--fit-trajectories','128','--validation-trajectories','64','--test-trajectories','64','--seq-len','500','--batch-size','16','--device','cuda'],['source_phase_arrays.npz','source_fit_arrays.npz','source_validation_arrays.npz','source_test_arrays.npz','source_reduced_schoyen_split_manifest.json'])
    return 0 if ok else 2


def reduced(ctx: Context):
    preflight(ctx)
    formal=ctx.artifact('formal_topology_schoyen')/'formal_topology_schoyen.json'
    splits=ctx.artifact('source_splits_schoyen')
    if not formal.is_file(): raise FileNotFoundError(formal)
    if read_json(formal).get('formal_topology_passed') is not True:
        ctx.mark('source_reduced_schoyen','not_evaluable',reason='Schoyen Euclidean formal topology did not pass'); return 0
    for f in ['source_phase_arrays.npz','source_fit_arrays.npz','source_validation_arrays.npz','source_test_arrays.npz','source_reduced_schoyen_split_manifest.json']:
        if not (splits/f).is_file(): raise FileNotFoundError(splits/f)
    cfg=read_yaml(ctx.code_root/'gpm_cmap'/'configs'/'source_reduced_schoyen.yaml')
    cfg['seed']=ctx.seed; cfg['device']='cpu'; cfg.setdefault('runtime',{})['cpu_num_threads']=ctx.threads
    navcfg=read_yaml(ctx.nav_cfg); cfg.setdefault('schoyen_phase',{})['arena_size']=float(navcfg['data']['arena_size'])
    name='source_reduced_schoyen'
    def command(out,n):
        cp=ctx.config_path(name,n)
        return [ctx.python_bin,'-u','-m','gpm_cmap.experiments.analyze_source_reduced_schoyen','--config',str(cp),'--formal-topology',str(formal),'--topology-dir',str(ctx.topology),'--phase-arrays',str(splits/'source_phase_arrays.npz'),'--fit-arrays',str(splits/'source_fit_arrays.npz'),'--validation-arrays',str(splits/'source_validation_arrays.npz'),'--test-arrays',str(splits/'source_test_arrays.npz'),'--split-manifest',str(splits/'source_reduced_schoyen_split_manifest.json'),'--checkpoint',str(ctx.nav_ckpt),'--output-dir',str(out)]
    ok=ctx.run_stage(name,command,['source_reduced_schoyen_v1.pkl','source_reduced_schoyen_summary.json','source_reduced_schoyen_test_metrics.csv'],config=cfg)
    return 0 if ok else 2


def mechanism(ctx: Context):
    preflight(ctx)
    reduced_dir=ctx.artifact('source_reduced_schoyen'); reduced_art=reduced_dir/'source_reduced_schoyen_v1.pkl'
    if not reduced_art.is_file():
        state=ctx.state('source_reduced_schoyen')
        if state.is_file() and read_json(state).get('status')=='not_evaluable':
            for target in ctx.pair['factorial_target_seeds']:
                for kind in ('action','path'): ctx.mark(f'{kind}_schoyen_t{target}','not_evaluable',reason='No Schoyen formal/reduced source')
            return 0
        raise FileNotFoundError(reduced_art)
    navsha=sha(ctx.nav_ckpt)
    all_ok=True
    for target in ctx.pair['factorial_target_seeds']:
        target=int(target); manifest=ctx.new/'artifacts'/f'factorial_t{target}'/'checkpoint_manifest.json'
        for kind,template in [('action','action_causality'),('path','path_invariance')]:
            cfg=read_yaml(ctx.code_root/'gpm_cmap'/'configs'/f'{template}.yaml')
            cfg['seed']=ctx.seed; cfg['device']='cuda'; cfg['source_reduced_artifact']=str(reduced_art)
            cfg['target']['checkpoint_manifest']=str(manifest); cfg['multinav_navigation_sha256']=navsha
            cfg.setdefault('runtime',{})['cpu_num_threads']=ctx.threads; cfg['runtime']['allow_device_fallback']=False
            if kind=='path': cfg['target']['conditions']=[x for x in cfg['target']['conditions'] if 'matched_random' not in x]
            name=f'{kind}_schoyen_t{target}'
            required=['action_causality_per_seed.csv','action_causality_summary.json'] if kind=='action' else ['path_invariance_per_seed.csv','path_invariance_summary.json']
            def command(out,n,kind=kind,name=name):
                cp=ctx.config_path(name,n)
                return [ctx.python_bin,'-u','-m','multinav.stage',kind,'--config',str(cp)]
            ok=ctx.run_stage(name,command,required,config=cfg); all_ok=all_ok and ok
    return 0 if all_ok else 2


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['preflight','prepare','reduced','mechanism'])
    ap.add_argument('--run-root',required=True); ap.add_argument('--result-root',required=True); ap.add_argument('--code-root',required=True); ap.add_argument('--seed',required=True,type=int); ap.add_argument('--python-bin',default=sys.executable)
    a=ap.parse_args(); ctx=Context(Path(a.run_root),Path(a.result_root),Path(a.code_root),a.seed,a.python_bin)
    try:
        if a.mode=='preflight': preflight(ctx); code=0
        elif a.mode=='prepare': code=prepare(ctx)
        elif a.mode=='reduced': code=reduced(ctx)
        else: code=mechanism(ctx)
    except Exception:
        traceback.print_exc(); code=2
    raise SystemExit(code)

if __name__=='__main__': main()
