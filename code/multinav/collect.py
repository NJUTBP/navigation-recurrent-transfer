"""Read only committed artifacts under ONE frozen run. Never load legacy archives."""
from __future__ import annotations
from pathlib import Path
import math
import numpy as np
import pandas as pd
from .common import *
from .stats import mean_ci, paired_source_effect, normalized_log_auc, holm

COLUMNS=['source_seed','target_seed','experiment','metric','arm','budget','transform','split','x','value','artifact']


def expected_stages(pair):
    result=['navigation','sequence_control','torus_topology','source_splits','source_reduced']
    for t in pair['main_target_seeds']:
        result += [f'{s}_t{t}' for s in ['exp5_nav','exp5_ctl','mapping','test_transforms','adapter','symbolic']]
    result += [f'hard_t{t}' for t in pair['hard_target_seeds']]
    for t in pair['factorial_target_seeds']: result += [f'{s}_t{t}' for s in ['factorial','action','path']]
    return result


def collect(root, plan):
    run=root/plan['run_root']; rows=[]; stages=[]; sources=[]; gates=[]; files={}; errors=[]
    def load(p):
        files[rel(p,root)]=sha(p)
        return read_json(p) if p.suffix=='.json' else pd.read_csv(p).to_dict('records')
    def add(s,t,experiment,metric,arm,value,artifact,**kwargs):
        if value is None: return
        try: value=float(value)
        except (ValueError,TypeError): return
        if not np.isfinite(value): return
        row=dict(source_seed=s,target_seed=t,experiment=experiment,metric=metric,arm=arm,
                 budget=-1,transform='',split='',x=-1,value=value,artifact=rel(artifact,root))
        row.update(kwargs); rows.append(row)
    for pair in plan['source_pairs']:
        s=pair['source_seed']; base=run/'sources'/f'nav_seed_{s}'
        identity=load(base/'source_identity.json') if (base/'source_identity.json').exists() else {}
        if identity and identity.get('plan_sha256') != plan['plan_sha256']:
            raise RuntimeError(f'Source {s}: identity belongs to another frozen plan')
        if identity and sha(root/identity['navigation_checkpoint']) != identity['navigation_checkpoint_sha256']:
            raise RuntimeError(f'Source {s}: locked navigation checkpoint changed')
        ctlq=load(base/'control_qualification.json') if (base/'control_qualification.json').exists() else {}
        topq=load(base/'topology_qualification.json') if (base/'topology_qualification.json').exists() else {}
        source=dict(identity,source_seed=s,control_seed=pair['control_seed'],
                    ctl_qualified=(ctlq.get('qualification') or {}).get('passed'),
                    ctl_training_completed=ctlq.get('training_completed',False),
                    topology_analysis_completed=topq.get('analysis_completed',False),
                    is_torus_like=topq.get('is_torus_like'),source_v3_evaluable=False,
                    source_classification=None,source_exact_gate=None,source_fibre_gate=None)
        status=load(base/'source_status.json') if (base/'source_status.json').exists() else {}
        source['worker_status']=status.get('status','missing')
        if source['worker_status']!='finished': errors.append(f'Source {s}: worker status {source["worker_status"]}')
        for name in expected_stages(pair):
            sp=base/'state'/(name+'.json'); st=load(sp) if sp.exists() else dict(status='missing')
            stages.append(dict(source_seed=s,stage=name,status=st['status'],reason=st.get('reason',st.get('error',''))))
            if st['status'] not in {'done','not_evaluable'}: errors.append(f'Source {s}: {name} = {st["status"]}')
        def committed(name, filename):
            st=base/'state'/(name+'.json'); p=base/'artifacts'/name/filename
            if not st.exists() or read_json(st).get('status')!='done': return None
            if not p.is_file(): errors.append(f'Missing committed artifact: {p}'); return None
            state=read_json(st); actual_path=rel(p.resolve(),root)
            expected=state.get('outputs',{}).get(actual_path)
            if expected and sha(p)!=expected: raise RuntimeError(f'Committed artifact modified: {p}')
            return p
        p=committed('navigation','history.json')
        if p:
            for row in load(p):
                for metric,key in [('validation_loss','validation_task_loss'),('grid_fraction','frac_grid_above_03')]:
                    add(s,-1,'navigation',metric,'navigation',row.get(key),p,x=int(row['epoch']))
        p=committed('sequence_control','training_history.json')
        if p:
            data=load(p); source['ctl_selected_epoch']=data.get('best_epoch')
            gate=data.get('training_gate') or {}; source['ctl_error_reduction']=gate.get('relative_nmse_reduction')
            for row in data['history']:
                val=row['validation']
                add(s,-1,'control','nmse','primary_delay',val['primary']['normalized_mse'],p,x=row['epoch'])
                add(s,-1,'control','nmse','auxiliary_mean',val.get('auxiliary_normalized_mse_mean'),p,x=row['epoch'])
            for lag,z in (data.get('best_validation',{}).get('by_lag') or {}).items():
                add(s,-1,'control','selected_delay_nmse','control',z.get('normalized_mse'),p,x=int(lag))
        for target in pair['main_target_seeds']:
            p=committed(f'exp5_nav_t{target}','raw_per_seed.json')
            if p:
                for arm,entries in load(p)['main'].items():
                    for z in entries:
                        add(s,target,'main','accuracy',arm,z['test_acc'],p)
                        if arm=='transfer_finetune': add(s,target,'mapping','accuracy','aligned',z['test_acc'],p)
                for hp in sorted(p.parent.glob(f'seed_{target}/*/history.json')):
                    arm=hp.parent.name
                    for z in load(hp):
                        add(s,target,'main','validation_accuracy',arm,z.get('validation_acc'),hp,x=z['epoch'])
            p=committed(f'exp5_ctl_t{target}','raw_per_seed.json')
            if p:
                for z in load(p)['main']['transfer_freeze']:
                    add(s,target,'main','accuracy','ctl01_frozen',z['test_acc'],p)
            p=committed(f'mapping_t{target}','raw_per_seed.json')
            if p:
                for arm,entries in load(p)['mapping_control'].items():
                    for z in entries: add(s,target,'mapping','accuracy',arm,z['test_acc'],p)
            p=committed(f'test_transforms_t{target}','test_transform_rows.json')
            if p:
                for z in load(p): add(s,target,'test_transforms','accuracy',z['arm'],z['test_accuracy'],p,transform=z['transform'])
            p=committed(f'adapter_t{target}','adapter_rescue_rows.json')
            if p:
                for z in load(p): add(s,target,'adapter','accuracy',z['arm'],z['test_accuracy'],p,transform=z['transform'])
            p=committed(f'symbolic_t{target}','symbolic_adapter_rows.json')
            if p:
                for z in load(p):
                    add(s,target,'symbolic','accuracy',z['arm'],z['test_accuracy'],p,budget=int(z['budget']))
                    add(s,target,'symbolic','no_recurrence_accuracy',z['arm'],z.get('test_accuracy_no_recurrence'),p,budget=int(z['budget']))
        for label,targets in [('hard',pair['hard_target_seeds']),('factorial',pair['factorial_target_seeds'])]:
            for target in targets:
                p=committed(f'{label}_t{target}',f'{label}_learning_curves.csv')
                if p:
                    for z in load(p):
                        for split,key in [('id','id_test_accuracy'),('long','long_test_accuracy'),('composition','composition_test_accuracy')]:
                            arm=z.get('condition',z.get('cell'))
                            add(s,target,label,'accuracy',arm,z[key],p,budget=int(z['unique_training_episodes']),split=split)
                if label=='factorial':
                    p=committed(f'factorial_t{target}','factorial_per_seed.csv')
                    if p:
                        for z in load(p):
                            for split,key in [('id','auc_id'),('long','auc_long'),('composition','auc_composition')]:
                                add(s,target,'factorial','auc',z['cell'],z[key],p,split=split)
        p=committed('torus_topology','torus_analysis_summary.json')
        if p:
            z=load(p); ph=z.get('persistent_cohomology') or {}; sel=z.get('selection') or {}
            source['topology_selected_clusters']=sel.get('selected_clusters')
            source['topology_selected_units']=sel.get('n_selected_units')
            source['topology_robust_min_gap']=sel.get('robust_min_gap_score')
            metrics=ph.get('metrics') or {}
            for metric_name,metric_payload in metrics.items():
                for homology,key in [('H1','H1_top10_lifetimes'),('H2','H2_top10_lifetimes')]:
                    for rank,v in enumerate(metric_payload.get(key) or [],1):
                        add(s,-1,'topology','lifetime',metric_name+'_'+homology,v,p,x=rank)
                add(s,-1,'topology','gap',metric_name+'_H1_2_to_3',metric_payload.get('H1_second_to_third'),p)
                add(s,-1,'topology','gap',metric_name+'_H2_1_to_2',metric_payload.get('H2_first_to_second'),p)
            add(s,-1,'topology','selection_score','robust_min_gap',sel.get('robust_min_gap_score'),p)
            add(s,-1,'topology','selected_units','auto_ensemble',sel.get('n_selected_units'),p)
            cands=p.parent/'candidate_topology.csv'
            if cands.exists():
                for row in load(cands):
                    add(s,-1,'topology','candidate_score',str(row.get('clusters')),row.get('robust_min_gap_score'),cands,
                        x=int(row.get('n_units') or -1),split='pass' if str(row.get('passes_robust_121_signature')).lower()=='true' else 'nonpass')
            source['topology_p_values']=None; source['topology_split']='automatic_morphology_plus_dual_metric_PH'
        p=committed('source_reduced','source_reduced_model_v3_summary.json')
        if p:
            z=load(p); quality=z['quality_gates']; source.update(source_v3_evaluable=True,
                source_classification=quality.get('classification'),source_exact_gate=quality['exact_2d_gate']['passed'],
                source_fibre_gate=quality['toroidal_base_plus_fibre_gate']['passed'])
            for gate_name in ['exact_2d_gate','toroidal_base_plus_fibre_gate']:
                for name,check in quality[gate_name]['checks'].items():
                    v=check.get('value'); threshold=check.get('threshold'); op=check.get('operator')
                    row=dict(source_seed=s,gate=gate_name,criterion=name,**check,artifact=rel(p,root)); gates.append(row)
                    if op in {'<=','>='} and v is not None and threshold and v>0:
                        margin=math.log2(v/threshold) if op=='>=' else math.log2(threshold/v)
                        add(s,-1,'source_gate','log2_margin',name,margin,p,split=gate_name)
        for target in pair['factorial_target_seeds']:
            p=committed(f'action_t{target}','action_causality_per_seed.csv')
            if p:
                for z in load(p):
                    for metric in ['minimum_confirmatory_null_ratio','normalized_phase_law_rmse','controlled_recurrent_reuse_gate_passed']:
                        val=z.get(metric)
                        if isinstance(val,str) and val.lower() in {'true','false'}: val=val.lower()=='true'
                        add(s,target,'action',metric,z['condition'],val,p,budget=int(z['unique_training_episodes']))
            p=committed(f'path_t{target}','path_invariance_per_seed.csv')
            if p:
                limits=original_config(root/plan['code_root'],'path_invariance')['analysis']
                for z in load(p):
                    specs=[('loop_holonomy',limits.get('max_loop_holonomy',.35)),
                           ('zero_padding_phase_difference',limits.get('max_padding_phase_difference',.25)),
                           ('path_identity_decoder_accuracy',float(z['path_identity_decoder_chance'])+limits.get('path_decoder_chance_tolerance',.10)),
                           ('residual_time_decoder_r2',limits.get('max_residual_time_r2',.20))]
                    passed=z.get('path_independence_gate_passed')
                    if isinstance(passed,str): passed=passed.lower()=='true'
                    add(s,target,'path','path_independence_gate_passed',z['condition'],passed,p,budget=int(z['unique_training_episodes']))
                    for metric,limit in specs:
                        add(s,target,'path','threshold_ratio',z['condition'],float(z[metric])/limit,p,transform=metric,budget=int(z['unique_training_episodes']))
        ci=base/'control_identity.json'
        if ci.exists():
            control_identity=load(ci)
            if control_identity.get('plan_sha256') != plan['plan_sha256']: raise RuntimeError('CTL identity has wrong plan')
            if sha(root/control_identity['checkpoint']) != control_identity['checkpoint_sha256']: raise RuntimeError('CTL checkpoint changed')
            source['ctl_core_sha256']=control_identity['core_sha256']
        p=committed('sequence_control','checkpoint_compatibility.json')
        if p: source['ctl_tensor_compatibility']=load(p)
        sources.append(source)
    seen={}
    for source in sources:
        h=source.get('core_sha256')
        if not h: errors.append(f'Missing independent source identity: {source["source_seed"]}')
        elif h in seen: errors.append(f'Duplicate NAVIGATION CORE weights: {seen[h]} and {source["source_seed"]}')
        else: seen[h]=source['source_seed']
    controls_seen={}
    for source in sources:
        h=source.get('ctl_core_sha256')
        if not h: errors.append(f'Missing independent CTL identity: {source["source_seed"]}')
        elif h in controls_seen: errors.append(f'Duplicate CTL core: {controls_seen[h]} and {source["source_seed"]}')
        else: controls_seen[h]=source['source_seed']
    frame=pd.DataFrame(rows,columns=COLUMNS)
    keys=[c for c in COLUMNS if c not in {'value','artifact'}]
    if frame.duplicated(keys).any(): raise RuntimeError('Duplicate source-target-metric cells; refusing ambiguous aggregation')
    return frame,sources,stages,gates,files,list(dict.fromkeys(errors))


def source_summary(frame,plan):
    keys=['experiment','metric','arm','budget','transform','split','x']; records=[]; source_rows=[]
    for key,group in frame.groupby(keys,dropna=False):
        info=dict(zip(keys,key)); exp=info['experiment']
        repeats=1 if exp in {'navigation','control','topology','source_gate'} else int(plan['config']['hard_target_repeats' if exp=='hard' else 'factorial_target_repeats' if exp in {'factorial','action','path'} else 'target_repeats'])
        means=[]
        for s,z in group.groupby('source_seed'):
            complete=z.target_seed.nunique()==repeats
            value=float(z.value.mean()) if complete else np.nan
            source_rows.append(dict(**info,source_seed=s,n_targets=z.target_seed.nunique(),complete=complete,value=value))
            if complete: means.append(value)
        records.append(dict(**info,**mean_ci(means),n_sources_planned=len(plan['source_pairs']),
                            scope='conditional_on_evaluable_sources' if exp in {'topology','source_gate','action','path'} else 'all_planned_sources',
                            coverage_complete=len(means)==len(plan['source_pairs'])))
    return pd.DataFrame(records),pd.DataFrame(source_rows)


def infer(frame, plan, sources):
    cfg=plan['config']; seeds=[p['source_seed'] for p in plan['source_pairs']]
    results=[]; details=[]; pairs=[]
    def subset(exp,metric='accuracy',**filters):
        x=frame[(frame.experiment==exp)&(frame.metric==metric)].copy()
        for key,value in filters.items(): x=x[x[key]==value]
        return x
    def compare(x,left,right,family,name,repeats=None,scope='all_planned_sources',selected=None):
        expected=seeds if selected is None else selected
        out,ds,ps=paired_source_effect(x,left,right,expected_sources=expected,
            expected_repeats=repeats or int(cfg['target_repeats']),bootstrap_samples=int(cfg['bootstrap_samples']),seed=int(cfg['statistics_seed'])+len(results))
        out.update(family=family,contrast=name,scope=scope,profile=cfg['profile'])
        if scope!='all_planned_sources': out['inference_status']='conditional_secondary_'+out['inference_status']
        if cfg['profile']=='smoke': out['inference_status']='SMOKE_NOT_SCIENTIFIC_INFERENCE'
        results.append(out)
        details.extend(dict(family=family,contrast=name,**z) for z in ds)
        pairs.extend(dict(family=family,contrast=name,**z) for z in ps.to_dict('records'))
    for left,right in [('transfer_freeze','ctl01_frozen'),('transfer_freeze','random_frozen'),('ctl01_frozen','random_frozen')]:
        compare(subset('main'),left,right,'F2_frozen_sources',left+'__minus__'+right)
    qualified=[z['source_seed'] for z in sources if z.get('ctl_qualified') is True]
    if len(qualified)>=2:
        compare(subset('main'),'transfer_freeze','ctl01_frozen','F2_qualified_CTL_sensitivity','nav_minus_ctl_qualified',
                scope='conditional_on_CTL_qualification',selected=qualified)
    for right in ['shuffled','mismatched_target']:
        compare(subset('mapping'),'aligned',right,'F3_mapping','aligned__minus__'+right)
    for arm in ['transfer_freeze','random_frozen']:
        x=subset('test_transforms'); x=x[x.arm==arm].copy(); x['arm']=x['transform']
        for right in ['rotate90','scale2x','signflip']:
            compare(x,'none',right,'F3_test_time_transforms',arm+'_none__minus__'+right)
    from gpm_cmap.models.adapter_wrap import transform_label
    transforms=original_config(Path.cwd()/plan['code_root'],'adapter_rescue')['transforms']
    if cfg['profile']=='smoke': transforms=['none','signflip',{'name':'scale','factor':.5}]
    for tf in [transform_label(z) for z in transforms if z!='none']:
        compare(subset('adapter',transform=tf),'nav_adapter','nav_noadapter','F3_adapter_gain','adapter_gain_'+tf)
    symbolic=subset('symbolic'); maxbudget=8 if cfg['profile']=='smoke' else 1000
    low=[4,8] if cfg['profile']=='smoke' else [10,20,50,100,200]
    auc=normalized_log_auc(symbolic,budgets=low)
    for right in ['ctl01_frozen','random_frozen','baseline_action_count_linear','baseline_linear_accumulator']:
        compare(symbolic[symbolic.budget==maxbudget],'nav_frozen',right,'F4_accuracy_1000','nav__minus__'+right)
        compare(auc,'nav_frozen',right,'F4_logAUC_10_200','nav__minus__'+right)
    for arm in ['nav_frozen','ctl01_frozen','random_frozen']:
        x=frame[(frame.experiment=='symbolic')&(frame.arm==arm)&(frame.budget==maxbudget)].copy(); x['arm']=x['metric']
        compare(x,'accuracy','no_recurrence_accuracy','F4_recurrence','intact_minus_zeroed_'+arm)
    hard_pairs=[('navigation_frozen','matched_random_frozen'),('navigation_trainable','matched_random_trainable'),
                ('navigation_frozen','navigation_hh_only_frozen'),('navigation_frozen','navigation_ih_only_frozen')]
    for split in ['id','long','composition']:
        for left,right in hard_pairs:
            compare(subset('hard',budget=maxbudget,split=split),left,right,'A2_max_budget',split+'_'+left+'__minus__'+right,
                    repeats=int(cfg['hard_target_repeats']))
    for spec in plan['factorial_contrasts']:
        compare(subset('factorial',metric='auc',split='long'),spec['left'],spec['right'],'A3_factorial_long_AUC',spec['name'],
                repeats=int(cfg['factorial_target_repeats']))
    result=pd.DataFrame(results)
    for family,idx in result.groupby('family').groups.items():
        for field in ['p_t','p_sign_flip','p_sign']:
            result.loc[idx,field+'_holm']=holm(result.loc[idx,field].values)
        result.loc[idx,'family_n_planned']=len(idx)
        result.loc[idx,'family_complete']=bool(result.loc[idx,'complete'].all())
    return result,pd.DataFrame(details),pd.DataFrame(pairs)


def mechanism_replication(frame,plan,sources):
    """All-source denominator; require every prespecified target repeat to pass.

    An unevaluable source is missing, not a measured failure. The fraction of
    confirmed passes over all planned sources is explicitly a lower-coverage
    summary. A diagnostic gate alone never overrides the upstream source gate.
    """
    cfg=plan['config']; source_map={s['source_seed']:s for s in sources}; rows=[]; counts=[]
    expected=int(cfg['factorial_target_repeats']); K=len(sources)
    for experiment,metric,template in [('action','controlled_recurrent_reuse_gate_passed','action_causality'),
                                        ('path','path_independence_gate_passed','path_invariance')]:
        arms=original_config(Path.cwd()/plan['code_root'],template)['target']['conditions']
        if experiment=='path':arms=[a for a in arms if 'matched_random' not in a]
        for arm in arms:
            local=[]
            for pair in plan['source_pairs']:
                source=source_map[pair['source_seed']]
                z=frame[(frame.experiment==experiment)&(frame.metric==metric)&(frame.arm==arm)&(frame.source_seed==pair['source_seed'])]
                complete=len(z)==expected and set(z.target_seed)==set(pair['factorial_target_seeds'])
                passed=bool((z.value==1.).all()) if complete else None
                upstream=source.get('source_exact_gate') is True or source.get('source_fibre_gate') is True
                row=dict(experiment=experiment,condition=arm,source_seed=pair['source_seed'],
                         n_target_repeats_planned=expected,n_target_repeats_observed=len(z),
                         n_target_gates_passed=int((z.value==1.).sum()),evaluable=complete,
                         all_target_gates_passed=passed,upstream_source_gate_passed=upstream,
                         combined_source_and_target_support=(upstream and passed) if complete else None,
                         source_classification=source.get('source_classification'))
                rows.append(row);local.append(row)
            counts.append(dict(experiment=experiment,condition=arm,n_sources_planned=K,
                n_sources_evaluable=sum(r['evaluable'] for r in local),
                n_sources_all_target_gates_passed=sum(r['all_target_gates_passed'] is True for r in local),
                n_sources_combined_support=sum(r['combined_source_and_target_support'] is True for r in local),
                rule='All prespecified target repeats must pass their native joint diagnostic gate; combined support additionally requires an upstream Source-v3 gate.'))
    return pd.DataFrame(rows),pd.DataFrame(counts)
