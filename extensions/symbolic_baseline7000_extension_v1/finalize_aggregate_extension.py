#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, shutil, hashlib, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ARMS=['baseline_action_count_linear','baseline_linear_accumulator']

def load_json(p):
    with Path(p).open('r',encoding='utf-8') as f:return json.load(f)
def write_json(p,x):
    with Path(p).open('w',encoding='utf-8') as f:json.dump(x,f,indent=2,sort_keys=True);f.write('\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def pairs(root):return load_json(root/'plan.json')['source_pairs']
def artifact(root,seed,target):return root/'rerun_best_validation'/'sources'/('nav_seed_%d'%seed)/'artifacts'/('symbolic_baseline7000_t%d'%target)
def rel(p,root):return str(Path(p).resolve().relative_to(root.resolve()))

def extension_rows(root):
    rows=[]
    for pair in pairs(root):
        s=int(pair['source_seed'])
        for t in [int(x) for x in pair['main_target_seeds']]:
            a=artifact(root,s,t); data=load_json(a/'symbolic_adapter_rows.json')
            good=[r for r in data if r.get('arm') in ARMS and int(r.get('budget',-1))==7000 and int(r.get('seed',-1))==t]
            if len(good)!=2:raise RuntimeError('Invalid extension rows %s'%a)
            for r in good:
                rows.append(dict(source_seed=s,target_seed=t,experiment='symbolic',metric='accuracy',arm=r['arm'],budget=7000,transform=np.nan,split=np.nan,x=-1,value=float(r['test_accuracy']),artifact=rel(a/'symbolic_adapter_rows.json',root)))
    if len(rows)!=60:raise RuntimeError('Expected 60 target-level extension rows, got %d'%len(rows))
    return pd.DataFrame(rows)

def source_rows(ext):
    out=[]
    for (arm,s),q in ext.groupby(['arm','source_seed'],sort=True):
        vals=q['value'].to_numpy(float)
        out.append(dict(experiment='symbolic',metric='accuracy',arm=arm,budget=7000,transform=np.nan,split=np.nan,x=-1,source_seed=int(s),n_targets=len(vals),complete=len(vals)==3,value=float(vals.mean())))
    z=pd.DataFrame(out)
    if len(z)!=20 or not z.complete.all():raise RuntimeError('Extension source means incomplete')
    return z

def summary_rows(src):
    out=[]
    for arm,q in src.groupby('arm',sort=True):
        v=q.value.to_numpy(float); n=len(v); mean=float(v.mean()); sem=float(v.std(ddof=1)/np.sqrt(n)); h=float(stats.t.ppf(.975,n-1)*sem)
        out.append(dict(experiment='symbolic',metric='accuracy',arm=arm,budget=7000,transform=np.nan,split=np.nan,x=-1,n=n,mean=mean,sem=sem,ci_low=mean-h,ci_high=mean+h,n_sources_planned=10,scope='all_planned_sources',coverage_complete=n==10))
    return pd.DataFrame(out)

def contrast_rows(src,old_source_means):
    nav=old_source_means[(old_source_means.experiment=='symbolic')&(old_source_means.metric=='accuracy')&(old_source_means.arm=='nav_frozen')&(old_source_means.budget==7000)][['source_seed','value']].rename(columns={'value':'nav'})
    rows=[]; detail=[]
    for arm in ARMS:
        b=src[src.arm==arm][['source_seed','value']].rename(columns={'value':'baseline'})
        m=nav.merge(b,on='source_seed',how='inner'); m['difference']=m.nav-m.baseline
        if len(m)!=10:raise RuntimeError('Cannot form 10-source 7000 contrast for %s'%arm)
        v=m.difference.to_numpy(float); mean=float(v.mean()); sem=float(v.std(ddof=1)/np.sqrt(10)); h=float(stats.t.ppf(.975,9)*sem)
        rows.append(dict(scope='secondary_descriptive_extension',family='F4_accuracy_7000_extension',contrast='nav__minus__'+arm,n_sources=10,mean_difference=mean,ci_low=mean-h,ci_high=mean+h,p_t=float(stats.ttest_1samp(v,0).pvalue),positive_sources=int((v>0).sum()),note='Protocol extension added after original baseline budget ended at 1000.'))
        for r in m.itertuples(index=False):detail.append(dict(source_seed=int(r.source_seed),contrast='nav__minus__'+arm,nav_7000=float(r.nav),baseline_7000=float(r.baseline),difference=float(r.difference)))
    return pd.DataFrame(rows),pd.DataFrame(detail)

def next_attempt(result):
    attempts=result/'aggregate_full_attempts'; attempts.mkdir(parents=True,exist_ok=True)
    nums=[]
    for p in attempts.glob('try_*'):
        try:nums.append(int(p.name.split('_')[-1]))
        except Exception:pass
    return attempts/('try_%03d'%(max(nums+[0])+1))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',default='.');ap.add_argument('--python',default=sys.executable);args=ap.parse_args()
    root=Path(args.project).resolve();result=root/'rerun_best_validation';public=result/'aggregate_full'
    if not public.exists() and not public.is_symlink():raise RuntimeError('Missing aggregate_full')
    current=public.resolve(); new=next_attempt(result); shutil.copytree(current,new,symlinks=True)
    ext=extension_rows(root); src=source_rows(ext)
    target=pd.read_csv(new/'target_level_metrics.csv'); target=target[~((target.experiment=='symbolic')&(target.metric=='accuracy')&(target.arm.isin(ARMS))&(target.budget==7000))]; target=pd.concat([target,ext],ignore_index=True); target.to_csv(new/'target_level_metrics.csv',index=False)
    means=pd.read_csv(new/'source_means.csv'); means=means[~((means.experiment=='symbolic')&(means.metric=='accuracy')&(means.arm.isin(ARMS))&(means.budget==7000))]; means=pd.concat([means,src],ignore_index=True); means.to_csv(new/'source_means.csv',index=False)
    summary=pd.read_csv(new/'source_level_summary.csv'); summary=summary[~((summary.experiment=='symbolic')&(summary.metric=='accuracy')&(summary.arm.isin(ARMS))&(summary.budget==7000))]; summary=pd.concat([summary,summary_rows(src)],ignore_index=True); summary.to_csv(new/'source_level_summary.csv',index=False)
    contrasts,details=contrast_rows(src,means)
    contrasts.to_csv(new/'symbolic_baseline7000_extension_contrasts.csv',index=False);details.to_csv(new/'symbolic_baseline7000_extension_by_source.csv',index=False)
    # Preserve original confirmatory contrast files.  This extension is explicitly secondary/descriptive.
    audit=load_json(new/'audit.json'); audit['symbolic_baseline7000_extension']=dict(status='complete',n_new_target_rows=60,n_new_source_rows=20,arms=ARMS,budget=7000,interpretation='secondary_descriptive_extension; original 1000-trial confirmatory families unchanged');audit['n_target_metric_rows']=len(target);audit['n_source_metric_rows']=len(means);write_json(new/'audit.json',audit)
    checks_path=new/'input_artifact_checksums.json'; checks=load_json(checks_path) if checks_path.is_file() else {}
    for pair in pairs(root):
        s=int(pair['source_seed'])
        for t in [int(x) for x in pair['main_target_seeds']]:
            p=artifact(root,s,t)/'symbolic_adapter_rows.json'; checks[rel(p,root)]=sha(p)
    write_json(checks_path,checks)
    manifest=dict(schema='symbolic_baseline7000_aggregate_extension_v1',source_aggregate=str(current),new_aggregate=str(new),budget=7000,arms=ARMS,n_target_rows=60,n_source_rows=20,original_confirmatory_families_unchanged=True)
    write_json(new/'symbolic_baseline7000_extension_manifest.json',manifest)
    env=os.environ.copy();env['PYTHONPATH']=str(root)+os.pathsep+env.get('PYTHONPATH','');env['MPLBACKEND']='Agg'
    py=str(Path(args.python).resolve())
    subprocess.run([py,'-u','-m','paper_figures.make_all_figures','--project',str(root),'--plan',str(root/'plan.json'),'--aggregate',str(new)],cwd=root,env=env,check=True)
    tmp=public.with_name(public.name+'.new')
    if tmp.exists() or tmp.is_symlink():tmp.unlink()
    tmp.symlink_to(os.path.relpath(new,public.parent));os.replace(tmp,public)
    print('[finalized] new aggregate_full -> %s'%new)
    print('[added] 60 target rows + 20 source rows for two 7000-trial baselines')
    print('[stats] %s'%(new/'symbolic_baseline7000_extension_contrasts.csv'))
if __name__=='__main__':main()
