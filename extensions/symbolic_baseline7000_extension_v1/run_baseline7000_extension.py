#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, hashlib, json, os, subprocess, sys, shutil
from pathlib import Path
import yaml

ARMS=("baseline_action_count_linear","baseline_linear_accumulator")
BUDGET=7000

def load_json(p):
    with Path(p).open('r',encoding='utf-8') as f: return json.load(f)
def write_json(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f: json.dump(x,f,indent=2,sort_keys=True); f.write('\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def load_yaml(p):
    with Path(p).open('r',encoding='utf-8') as f: return yaml.safe_load(f)
def write_yaml(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f: yaml.safe_dump(x,f,sort_keys=False)

def project(args): return Path(args.project).resolve()
def result_root(root): return root/'rerun_best_validation'
def source_dir(root,seed): return result_root(root)/'sources'/('nav_seed_%d'%seed)
def original_symbolic_dir(root,seed,target): return source_dir(root,seed)/'artifacts'/('symbolic_t%d'%target)
def ext_name(target): return 'symbolic_baseline7000_t%d'%target
def ext_artifact(root,seed,target): return source_dir(root,seed)/'artifacts'/ext_name(target)

def plan_pairs(root):
    p=root/'plan.json'
    if not p.is_file(): raise RuntimeError('Missing plan.json: %s'%p)
    return load_json(p)['source_pairs']

def targets_for(root,seed):
    pair=next((p for p in plan_pairs(root) if int(p['source_seed'])==int(seed)),None)
    if pair is None: raise RuntimeError('Source %d not found in plan'%seed)
    return [int(x) for x in pair['main_target_seeds']]

def config_path(sym):
    p=sym/'config_used.yaml'
    if p.is_file(): return p
    raise RuntimeError('Missing original symbolic config: %s'%p)

def validate_original(sym,target):
    rows=sym/'symbolic_adapter_rows.json'
    split=sym/'split_audits'/('seed_%d.json'%target)
    if not rows.is_file(): raise RuntimeError('Missing %s'%rows)
    if not split.is_file(): raise RuntimeError('Missing %s'%split)
    z=load_json(rows)
    recurrent=[r for r in z if int(r.get('budget',-1))==7000 and r.get('arm') in {'nav_frozen','ctl01_frozen','random_frozen'}]
    if len(recurrent)<2:
        raise RuntimeError('Original symbolic target %d lacks recurrent 7000 rows'%target)
    existing=[r for r in z if int(r.get('budget',-1))==7000 and r.get('arm') in ARMS]
    if existing:
        print('[already present] %s has %d baseline 7000 rows'%(sym,len(existing)))
    return z

def cmd_preflight(args):
    root=project(args)
    code=root/'code'/'gpm_cmap'/'experiments'/'exp8_symbolic_adapter.py'
    if not code.is_file(): raise RuntimeError('Missing frozen Exp8 code: %s'%code)
    text=code.read_text(encoding='utf-8')
    if '--controls-only' not in text:
        raise RuntimeError('Frozen exp8_symbolic_adapter.py does not support --controls-only; do not submit extension')
    n=0
    for pair in plan_pairs(root):
        seed=int(pair['source_seed'])
        for target in [int(x) for x in pair['main_target_seeds']]:
            sym=original_symbolic_dir(root,seed,target)
            validate_original(sym,target); cfg=load_yaml(config_path(sym))
            if int(cfg.get('seed',-1))!=target:
                raise RuntimeError('Config seed mismatch source=%d target=%d observed=%s'%(seed,target,cfg.get('seed')))
            budgets=[int(x) for x in cfg.get('episode_budgets',[])]
            if 7000 not in budgets: raise RuntimeError('Original symbolic config does not include 7000: %s'%sym)
            n+=1
    if n!=30: raise RuntimeError('Expected 30 symbolic targets, found %d'%n)
    print('[preflight ok] 10 sources / 30 targets; recurrent 7000 rows exist; controls-only supported')

def prepare_extension_config(sym,out_cfg,out_dir):
    cfg=load_yaml(config_path(sym))
    cfg['output_dir']=str(out_dir)
    cfg['n_seeds']=1
    cfg['episode_budgets']=[7000]
    cfg['low_budget_max']=200
    controls=dict(cfg.get('controls') or {})
    controls['include_action_count_linear']=True
    controls['include_linear_accumulator']=True
    controls['baseline_budgets']=[7000]
    controls['recurrence_ablation_budgets']=[]
    cfg['controls']=controls
    write_yaml(out_cfg,cfg)
    return cfg

def compare_split(original,extension):
    a=load_json(original); b=load_json(extension)
    if a!=b:
        # A strict equality failure means the new rows are not on the exact original split.
        raise RuntimeError('Extension split audit differs from original: %s vs %s'%(original,extension))

def validate_extension(out,target):
    rows=load_json(out/'symbolic_adapter_rows.json')
    good=[r for r in rows if int(r.get('budget',-1))==7000 and r.get('arm') in ARMS and int(r.get('seed',-1))==target]
    if len(good)!=2 or set(r['arm'] for r in good)!=set(ARMS):
        raise RuntimeError('Expected exactly two baseline 7000 rows for target %d; got %s'%(target,[(r.get('arm'),r.get('budget'),r.get('seed')) for r in rows]))
    return good

def cmd_run_source(args):
    root=project(args); seed=int(args.source_seed); py=Path(args.python).resolve()
    if not py.is_file(): raise RuntimeError('Python executable not found: %s'%py)
    base=source_dir(root,seed); code=root/'code';
    for target in targets_for(root,seed):
        sym=original_symbolic_dir(root,seed,target); validate_original(sym,target)
        public=ext_artifact(root,seed,target)
        if public.exists() or public.is_symlink():
            try:
                validate_extension(public,target)
                compare_split(sym/'split_audits'/('seed_%d.json'%target), public/'split_audits'/('seed_%d.json'%target))
                print('[reuse] source=%d target=%d'%(seed,target)); continue
            except Exception:
                raise RuntimeError('Existing extension artifact is invalid: %s'%public)
        tries=base/'attempts'/ext_name(target); tries.mkdir(parents=True,exist_ok=True)
        nums=[]
        for p in tries.glob('try_*'):
            try: nums.append(int(p.name.split('_')[-1]))
            except Exception: pass
        out=tries/('try_%03d'%(max(nums+[0])+1)); out.mkdir()
        cfg_dir=base/'configs'; cfg_dir.mkdir(parents=True,exist_ok=True)
        cfg_path=cfg_dir/('%s.yaml'%ext_name(target))
        prepare_extension_config(sym,cfg_path,out)
        log=base/'logs'/('%s.log'%ext_name(target)); log.parent.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy(); env['PYTHONPATH']=str(code)+os.pathsep+env.get('PYTHONPATH',''); env['MPLBACKEND']='Agg'; env['PYTHONHASHSEED']=str(target)
        cmd=[str(py),'-u','-m','gpm_cmap.experiments.exp8_symbolic_adapter','--config',str(cfg_path),'--n_seeds','1','--controls-only']
        print('[run] source=%d target=%d -> %s'%(seed,target,log),flush=True)
        with log.open('w',encoding='utf-8') as stream:
            subprocess.run(cmd,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
        rows=validate_extension(out,target)
        compare_split(sym/'split_audits'/('seed_%d.json'%target),out/'split_audits'/('seed_%d.json'%target))
        manifest={
            'schema':'symbolic_baseline7000_extension_v1','source_seed':seed,'target_seed':target,'budget':7000,
            'arms':list(ARMS),'original_symbolic_artifact':str(sym.relative_to(root)),
            'original_config_sha256':sha(config_path(sym)),'extension_config':str(cfg_path.relative_to(root)),
            'rows_sha256':sha(out/'symbolic_adapter_rows.json'),
            'split_audit_sha256':sha(out/'split_audits'/('seed_%d.json'%target)),
            'original_split_audit_sha256':sha(sym/'split_audits'/('seed_%d.json'%target)),
            'split_audit_exact_match':True,
        }
        write_json(out/'baseline7000_extension_manifest.json',manifest)
        public.parent.mkdir(parents=True,exist_ok=True); tmp=public.with_name(public.name+'.new')
        if tmp.exists() or tmp.is_symlink(): tmp.unlink()
        tmp.symlink_to(os.path.relpath(out,public.parent)); os.replace(tmp,public)
        state=base/'state'/('%s.json'%ext_name(target)); write_json(state,dict(status='done',stage=ext_name(target),attempt=str(out.relative_to(root)),artifact=str(public.relative_to(root)),manifest=manifest))
        print('[done] source=%d target=%d action_count=%.4f accumulator=%.4f'%(seed,target,next(r['test_accuracy'] for r in rows if r['arm']==ARMS[0]),next(r['test_accuracy'] for r in rows if r['arm']==ARMS[1])))

def cmd_verify(args):
    root=project(args); n=0
    for pair in plan_pairs(root):
        seed=int(pair['source_seed'])
        for target in [int(x) for x in pair['main_target_seeds']]:
            out=ext_artifact(root,seed,target); validate_extension(out,target)
            compare_split(original_symbolic_dir(root,seed,target)/'split_audits'/('seed_%d.json'%target),out/'split_audits'/('seed_%d.json'%target)); n+=1
    if n!=30: raise RuntimeError('Expected 30 extension artifacts, found %d'%n)
    print('[verify ok] 30 targets x 2 baseline rows at 7000; all split audits exactly match originals')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',default='.'); ap.add_argument('--python',default=sys.executable)
    sub=ap.add_subparsers(dest='cmd',required=True)
    sub.add_parser('preflight')
    p=sub.add_parser('run-source'); p.add_argument('--source-seed',type=int,required=True)
    sub.add_parser('verify')
    args=ap.parse_args(); {'preflight':cmd_preflight,'run-source':cmd_run_source,'verify':cmd_verify}[args.cmd](args)
if __name__=='__main__': main()
