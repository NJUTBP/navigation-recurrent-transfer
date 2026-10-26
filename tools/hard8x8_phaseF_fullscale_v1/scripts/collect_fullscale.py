#!/usr/bin/env python3
from __future__ import annotations
import os

import csv
import json
import math
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT=Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[3])))
PACKAGE_ROOT=PROJECT_ROOT/'tools'/'hard8x8_phaseF_fullscale_v1'
PROTOCOL=json.loads((PACKAGE_ROOT/'PROTOCOL.json').read_text())
OUTPUT_ROOT=PROJECT_ROOT/'rerun_best_validation'/'hard8x8_diagnostics'/'audit_v1'/'phaseF_fullscale_v1'
AGG=OUTPUT_ROOT/'aggregate'
GROUPS=['fixed_scale','learned_interface','full_finetune']


def read_csv(path:Path):
    if not path.is_file() or path.stat().st_size==0: return []
    with path.open(newline='',encoding='utf-8') as f: return list(csv.DictReader(f))


def write_csv(path:Path, rows:list[dict[str,Any]]):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: path.write_text(''); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def exact_signflip(values):
    x=np.asarray(values,dtype=float); obs=abs(float(x.mean())); n=len(x)
    vals=[]
    for mask in range(1<<n):
        s=np.asarray([1.0 if mask&(1<<i) else -1.0 for i in range(n)])
        vals.append(abs(float(np.mean(s*x))))
    vals=np.asarray(vals)
    return float(np.mean(vals >= obs-1e-15))


def bootstrap_ci(values,n=20000,seed=20260917):
    x=np.asarray(values,dtype=float); rng=np.random.default_rng(int(seed))
    boot=np.empty(int(n),dtype=float)
    for i in range(int(n)):
        boot[i]=rng.choice(x,size=len(x),replace=True).mean()
    return float(np.quantile(boot,.025)),float(np.quantile(boot,.975))


def holm(ps):
    order=sorted(range(len(ps)),key=lambda i:ps[i]); out=[None]*len(ps); running=0.0; m=len(ps)
    for rank,i in enumerate(order):
        val=min(1.0,(m-rank)*ps[i]); running=max(running,val); out[i]=running
    return out


def mean_rows(rows, key_fields, numeric_fields):
    groups=defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in key_fields)].append(r)
    out=[]
    for key,grp in sorted(groups.items()):
        row={k:v for k,v in zip(key_fields,key)}
        row['n_target_repeats']=len(grp)
        for f in numeric_fields:
            vals=[float(x[f]) for x in grp if x.get(f,'') not in ('',None)]
            row[f]=float(np.mean(vals)) if vals else ''
        out.append(row)
    return out


def contrast(source_metrics,left,right,split):
    by=defaultdict(dict)
    for r in source_metrics:
        if r['split']==split and r['condition'] in {left,right}:
            by[int(r['source_seed'])][r['condition']]=float(r['accuracy'])
    diffs=[]; per={}
    for s in sorted(by):
        if left in by[s] and right in by[s]:
            d=by[s][left]-by[s][right]; diffs.append(d); per[s]=d
    if len(diffs)!=10: raise RuntimeError(f'contrast {left}-{right} {split} has n={len(diffs)}')
    lo,hi=bootstrap_ci(diffs,PROTOCOL['bootstrap_samples'],stable_seed(left,right,split))
    return {'split':split,'left':left,'right':right,'n_sources':len(diffs),'mean_difference':float(np.mean(diffs)),
            'sd_difference':float(np.std(diffs,ddof=1)),'ci95_low':lo,'ci95_high':hi,
            'exact_signflip_p':exact_signflip(diffs),'per_source_differences':json.dumps(per,sort_keys=True)}


def stable_seed(*parts):
    import hashlib
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode()).digest()[:4],'little')


def condition_mean(source_metrics, condition, split):
    vals=[float(r['accuracy']) for r in source_metrics if r['condition']==condition and r['split']==split]
    if len(vals)!=10: raise RuntimeError((condition,split,len(vals)))
    return float(np.mean(vals))


def baseline_mean(source_baselines, split):
    vals=[float(r['accuracy']) for r in source_baselines if r['split']==split and r['baseline']=='start_plus_length']
    if len(vals)!=10: raise RuntimeError(('baseline',split,len(vals)))
    return float(np.mean(vals))


def intervention_mean(source_interventions, condition, split, field):
    vals=[float(r[field]) for r in source_interventions if r['condition']==condition and r['split']==split and r.get(field,'') not in ('',None)]
    if len(vals)!=10: raise RuntimeError((condition,split,field,len(vals)))
    return float(np.mean(vals))


def main():
    AGG.mkdir(parents=True,exist_ok=True)
    missing=[]; complete=[]
    all_metrics=[]; all_per=[]; all_int=[]; all_up=[]; all_diag=[]; baseline_map={}; audit_map={}; ckpts=[]
    for s in PROTOCOL['all_sources']:
        for t in PROTOCOL['target_split_seeds']:
            for g in GROUPS:
                d=OUTPUT_ROOT/f'source_{s}'/f'target_{t}'/f'group_{g}'
                if not (d/'COMPLETE.json').is_file(): missing.append(str(d)); continue
                complete.append(str(d))
                all_metrics += read_csv(d/'metrics.csv')
                all_per += read_csv(d/'per_length.csv')
                all_int += read_csv(d/'relation_interventions.csv')
                all_up += read_csv(d/'update_curves.csv')
                all_diag += read_csv(d/'condition_diagnostics.csv')
                ckpts += read_csv(d/'checkpoint_manifest.csv')
                for r in read_csv(d/'baselines.csv'):
                    baseline_map[(r['source_seed'],r['target_seed'],r['split'],r['baseline'])]=r
                for r in read_csv(d/'dataset_audit.csv'):
                    audit_map[(s,t,r['split'])]=r
    readiness={'expected_group_jobs':len(PROTOCOL['all_sources'])*len(PROTOCOL['target_split_seeds'])*len(GROUPS),
               'complete_group_jobs':len(complete),'missing':missing,'ready':not missing}
    (AGG/'READINESS.json').write_text(json.dumps(readiness,indent=2)+'\n')
    (AGG/'READINESS.md').write_text(f"# Phase F readiness\n\n- complete group jobs: {len(complete)}/{readiness['expected_group_jobs']}\n- missing: {len(missing)}\n- ready: {not missing}\n")
    if missing: raise SystemExit(3)

    baselines=list(baseline_map.values()); audits=list(audit_map.values())
    write_csv(AGG/'all_metrics.csv',all_metrics); write_csv(AGG/'all_per_length.csv',all_per)
    write_csv(AGG/'all_relation_interventions.csv',all_int); write_csv(AGG/'all_update_curves.csv',all_up)
    write_csv(AGG/'all_condition_diagnostics.csv',all_diag); write_csv(AGG/'all_baselines.csv',baselines)
    write_csv(AGG/'all_dataset_audit.csv',audits); write_csv(AGG/'checkpoint_manifest.csv',ckpts)

    source_metrics=mean_rows(all_metrics,['source_seed','condition','split'],['accuracy','nll','ece','mean_confidence','mean_readout_margin','best_update','stopped_update'])
    source_baselines=mean_rows(baselines,['source_seed','split','baseline'],['accuracy'])
    source_int=mean_rows(all_int,['source_seed','condition','split'],['real_accuracy','zero_accuracy','zero_true_logp_drop','zero_prediction_change','counterfactual_accuracy','counterfactual_target_logp_gain','counterfactual_prediction_change'])
    source_diag=mean_rows(all_diag,['source_seed','condition','origin','kind','core_trainability'],['best_update','gain','log_gain','adapter_singular_1','adapter_singular_2','adapter_singular_ratio','adapter_polar_angle_deg'])
    write_csv(AGG/'source_level_metrics.csv',source_metrics); write_csv(AGG/'source_level_baselines.csv',source_baselines)
    write_csv(AGG/'source_level_relation_interventions.csv',source_int); write_csv(AGG/'source_level_condition_diagnostics.csv',source_diag)

    left=PROTOCOL['primary_condition_left']; right=PROTOCOL['primary_condition_right']
    general=[]
    for split in PROTOCOL['generalization_family']:
        general.append(contrast(source_metrics,left,right,split))
    adj=holm([r['exact_signflip_p'] for r in general])
    for r,a in zip(general,adj): r['holm_p_across_four_generalization_endpoints']=a
    write_csv(AGG/'primary_generalization_contrasts.csv',general)
    primary=next(r for r in general if r['split']=='id_seen_short')

    secondary=[]
    pairs=[
      ('fixed_gain10_nav_minus_matched','navigation_fixed_gain10_frozen','matched_random_fixed_gain10_frozen'),
      ('full2x2_nav_minus_matched','navigation_trainable_input_adapter','matched_random_trainable_input_adapter'),
      ('scalar_nav_minus_fresh_random','navigation_trainable_scalar_gain','fresh_random_trainable_scalar_gain'),
      ('scalar_matched_minus_fresh_random','matched_random_trainable_scalar_gain','fresh_random_trainable_scalar_gain'),
      ('fullcore_nav_minus_matched','navigation_full_core_trainable','matched_random_full_core_trainable'),
      ('fullcore_nav_minus_fresh_random','navigation_full_core_trainable','fresh_random_full_core_trainable'),
    ]
    for name,l,r in pairs:
        row=contrast(source_metrics,l,r,'id_seen_short'); row['contrast_name']=name; secondary.append(row)
    sec_adj=holm([r['exact_signflip_p'] for r in secondary])
    for r,a in zip(secondary,sec_adj): r['holm_p_secondary_family']=a
    write_csv(AGG/'secondary_id_contrasts.csv',secondary)

    baseline=baseline_mean(source_baselines,'id_seen_short')
    nav_id=condition_mean(source_metrics,left,'id_seen_short'); matched_id=condition_mean(source_metrics,right,'id_seen_short')
    nav_zero=intervention_mean(source_int,left,'id_seen_short','zero_true_logp_drop')
    matched_zero=intervention_mean(source_int,right,'id_seen_short','zero_true_logp_drop')
    nav_cf=intervention_mean(source_int,left,'id_seen_short','counterfactual_target_logp_gain')
    matched_cf=intervention_mean(source_int,right,'id_seen_short','counterfactual_target_logp_gain')
    gate=bool(nav_id>baseline and matched_id>baseline and nav_zero>0 and matched_zero>0 and nav_cf>0 and matched_cf>0)

    # Condition performance matrix.
    conds=[]
    for cond in sorted({r['condition'] for r in source_metrics}):
        row={'condition':cond}
        for split in PROTOCOL['generalization_family']:
            row[split+'_accuracy']=condition_mean(source_metrics,cond,split)
        vals=[float(r['best_update']) for r in source_metrics if r['condition']==cond]
        row['mean_best_update']=float(np.mean(vals)); row['fraction_source_mean_best_update_ge_7600']=float(np.mean(np.asarray(vals)>=7600))
        conds.append(row)
    write_csv(AGG/'condition_performance_matrix.csv',conds)

    result={
        'schema_version':'hard8x8_phaseF_fullscale_v1_result',
        'competence_gate_passed':gate,'start_plus_length_id_baseline':baseline,
        'nav_scalar_id':nav_id,'matched_scalar_id':matched_id,
        'primary':primary,'generalization_family':general,'secondary_id_contrasts':secondary,
    }
    (AGG/'FINAL_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')

    lines=['# Phase F full-scale harder 8×8 result','',f"- complete group jobs: {len(complete)}/{readiness['expected_group_jobs']}",
           f"- source seeds: {len(PROTOCOL['all_sources'])}",f"- fresh target splits per source: {len(PROTOCOL['target_split_seeds'])}",
           f"- trained conditions: {len({r['condition'] for r in all_metrics})}",f"- competence gate passed: **{gate}**",'',
           '## Primary scalar-gain navigation contrast','',
           f"- start+length ID baseline: {baseline:.4f}",f"- Nav scalar ID: {nav_id:.4f}",f"- matched scalar ID: {matched_id:.4f}",
           f"- Nav - matched ID: {float(primary['mean_difference']):+.4f}",
           f"- 95% source bootstrap CI: [{float(primary['ci95_low']):+.4f}, {float(primary['ci95_high']):+.4f}]",
           f"- exact sign-flip p: {float(primary['exact_signflip_p']):.6f}",'','## Orthogonal composition × horizon generalization','']
    for r in general:
        lines.append(f"- {r['split']}: Δ={float(r['mean_difference']):+.4f}, CI [{float(r['ci95_low']):+.4f}, {float(r['ci95_high']):+.4f}], exact p={float(r['exact_signflip_p']):.6f}, Holm={float(r['holm_p_across_four_generalization_endpoints']):.6f}")
    lines += ['', '## Secondary ID contrasts','']
    for r in secondary:
        lines.append(f"- {r['contrast_name']}: Δ={float(r['mean_difference']):+.4f}, exact p={float(r['exact_signflip_p']):.6f}, Holm={float(r['holm_p_secondary_family']):.6f}")
    lines += ['', '## Interpretation guard','',
              '- The scalar Nav-vs-matched contrast is interpreted as navigation-specific only if the competence gate passes.',
              '- Fixed-gain arms separate input scale from trainable-gain parameterization.',
              '- Fresh-random arms test whether the effect is specific to navigation pretraining rather than generic recurrent dynamics.',
              '- Full-core finetuning tests whether preserving the pretrained recurrent computation is necessary for the source advantage.',
              '- heldout_short / seen_long / heldout_long form an orthogonal composition × horizon generalization grid and are never used for checkpoint selection.']
    (AGG/'FINAL_RESULT.md').write_text('\n'.join(lines)+'\n')

    # Package metrics/results only; checkpoints remain on cluster.
    archive=OUTPUT_ROOT/'hard8x8_phaseF_fullscale_v1_results.zip'
    if archive.exists(): archive.unlink()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(AGG.rglob('*')):
            if p.is_file(): zf.write(p,p.relative_to(OUTPUT_ROOT))
        sub=OUTPUT_ROOT/'SUBMISSION.txt'
        if sub.is_file(): zf.write(sub,sub.relative_to(OUTPUT_ROOT))
    print('wrote',archive)
    print((AGG/'FINAL_RESULT.md').read_text())

if __name__=='__main__': main()
