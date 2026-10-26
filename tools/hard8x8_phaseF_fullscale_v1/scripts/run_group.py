#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from fullscale_common import (
    PROJECT_ROOT,
    atomic_json,
    build_fullscale_split,
    deterministic_stratified_subsample,
    frequency_baseline,
    import_historical,
    intervention_summary,
    make_loader,
    split_audit,
    stable_seed32,
    to_historical_episodes,
    write_csv,
)

PACKAGE_ROOT = PROJECT_ROOT / 'tools' / 'hard8x8_phaseF_fullscale_v1'
PROTOCOL_PATH = PACKAGE_ROOT / 'PROTOCOL.json'
OUTPUT_ROOT = PROJECT_ROOT / 'rerun_best_validation' / 'hard8x8_diagnostics' / 'audit_v1' / 'phaseF_fullscale_v1'
SCHEMA_VERSION = 'hard8x8_phaseF_fullscale_v1'


class FixedGainPathIntModel(nn.Module):
    supports_prevalidated_inputs = True
    def __init__(self, base, gain: float):
        super().__init__(); self.base=base
        self.register_buffer('_fixed_gain', torch.tensor(float(gain), dtype=torch.float32))
    @property
    def n_nodes(self): return self.base.n_nodes
    @property
    def start_init(self): return self.base.start_init
    @property
    def grid_rnn(self): return self.base.grid_rnn
    @property
    def node_head(self): return self.base.node_head
    def recurrent_l2(self): return self.base.recurrent_l2()
    def gain(self): return self._fixed_gain
    def forward(self,start_node,velocities,lengths=None,start_coords=None,*,validate_inputs=True):
        return self.base(
            start_node=start_node,
            velocities=velocities*self._fixed_gain,
            lengths=lengths,
            start_coords=start_coords,
            validate_inputs=validate_inputs,
        )


class ScalarGainPathIntModel(nn.Module):
    supports_prevalidated_inputs = True
    def __init__(self, base):
        super().__init__(); self.base=base
        self.log_gain=nn.Parameter(torch.zeros((),dtype=torch.float32))
    @property
    def n_nodes(self): return self.base.n_nodes
    @property
    def start_init(self): return self.base.start_init
    @property
    def grid_rnn(self): return self.base.grid_rnn
    @property
    def node_head(self): return self.base.node_head
    def recurrent_l2(self): return self.base.recurrent_l2()
    def gain(self): return torch.exp(self.log_gain)
    def configure_trainable(self):
        self.base.freeze_grid_rnn_all()
        if getattr(self.base,'position_head',None) is not None:
            for p in self.base.position_head.parameters(): p.requires_grad=False
        if getattr(self.base,'periodic_encoder',None) is not None:
            for p in self.base.periodic_encoder.parameters(): p.requires_grad=False
        if getattr(self.base,'target_rhc_encoder',None) is not None:
            for p in self.base.target_rhc_encoder.parameters(): p.requires_grad=False
        self.log_gain.requires_grad=True
    def forward(self,start_node,velocities,lengths=None,start_coords=None,*,validate_inputs=True):
        return self.base(
            start_node=start_node,
            velocities=velocities*self.gain(),
            lengths=lengths,
            start_coords=start_coords,
            validate_inputs=validate_inputs,
        )


def adapter_summary(model) -> dict[str,Any]:
    report=model.alignment_report()
    A=np.asarray(report['adapter'],dtype=np.float64)
    U,S,Vt=np.linalg.svd(A); R=U@Vt
    angle=float(np.degrees(np.arctan2(R[1,0],R[0,0])))
    return {
        'adapter_a00':float(A[0,0]),'adapter_a01':float(A[0,1]),
        'adapter_a10':float(A[1,0]),'adapter_a11':float(A[1,1]),
        'adapter_fro_delta_identity':float(np.linalg.norm(A-np.eye(2),ord='fro')),
        'adapter_det':float(np.linalg.det(A)),
        'adapter_singular_1':float(S[0]),'adapter_singular_2':float(S[1]),
        'adapter_singular_ratio':float(S[0]/max(S[1],1e-12)),
        'adapter_polar_angle_deg':angle,
        'adapter_matrix':A.tolist(),
    }


def condition_specs() -> dict[str,dict[str,Any]]:
    return {
        'navigation_identity_frozen': {'origin':'navigation','kind':'fixed_gain','gain':1.0,'core':'frozen'},
        'matched_random_identity_frozen': {'origin':'matched_random','kind':'fixed_gain','gain':1.0,'core':'frozen'},
        'navigation_fixed_gain3_frozen': {'origin':'navigation','kind':'fixed_gain','gain':3.0,'core':'frozen'},
        'matched_random_fixed_gain3_frozen': {'origin':'matched_random','kind':'fixed_gain','gain':3.0,'core':'frozen'},
        'navigation_fixed_gain10_frozen': {'origin':'navigation','kind':'fixed_gain','gain':10.0,'core':'frozen'},
        'matched_random_fixed_gain10_frozen': {'origin':'matched_random','kind':'fixed_gain','gain':10.0,'core':'frozen'},
        'navigation_trainable_scalar_gain': {'origin':'navigation','kind':'scalar','core':'frozen'},
        'matched_random_trainable_scalar_gain': {'origin':'matched_random','kind':'scalar','core':'frozen'},
        'fresh_random_trainable_scalar_gain': {'origin':'fresh_random','kind':'scalar','core':'frozen'},
        'navigation_trainable_input_adapter': {'origin':'navigation','kind':'adapter2x2','core':'frozen'},
        'matched_random_trainable_input_adapter': {'origin':'matched_random','kind':'adapter2x2','core':'frozen'},
        'fresh_random_trainable_input_adapter': {'origin':'fresh_random','kind':'adapter2x2','core':'frozen'},
        'navigation_full_core_trainable': {'origin':'navigation','kind':'identity','core':'trainable'},
        'matched_random_full_core_trainable': {'origin':'matched_random','kind':'identity','core':'trainable'},
        'fresh_random_full_core_trainable': {'origin':'fresh_random','kind':'identity','core':'trainable'},
    }


def copy_fresh_random_grid(base, hist, model_cfg, device, *, seed:int):
    hist['set_seed'](int(seed))
    donor=hist['build_hard_model'](model_cfg,n_nodes=64,device=device)
    state={k:v.detach().clone() for k,v in donor.grid_rnn.state_dict().items()}
    base.grid_rnn.load_state_dict(state,strict=True)
    del donor


def configure_origin(base, spec, hist, *, nav_hh, nav_ih, matched_hh, matched_ih, cell_cls, model_cfg, device, random_core_seed):
    origin=spec['origin']; core=spec['core']
    if origin in {'navigation','matched_random'}:
        cell=cell_cls(
            origin,origin,
            'trainable' if core=='trainable' else 'frozen',
            'trainable' if core=='trainable' else 'frozen',
        )
        hist['install_factorial_weights'](
            base,cell=cell,nav_hh=nav_hh,nav_ih=nav_ih,
            random_hh=matched_hh,random_ih=matched_ih,
        )
    elif origin=='fresh_random':
        copy_fresh_random_grid(base,hist,model_cfg,device,seed=random_core_seed)
        if core=='trainable':
            base.set_grid_trainability(input_trainable=True,recurrent_trainable=True)
        else:
            base.freeze_grid_rnn_all()
    else:
        raise ValueError(origin)


def wrap_model(base,spec,hist,device):
    kind=spec['kind']
    if kind=='identity':
        model=base
    elif kind=='fixed_gain':
        model=FixedGainPathIntModel(base,float(spec['gain']))
    elif kind=='scalar':
        model=ScalarGainPathIntModel(base); model.configure_trainable()
    elif kind=='adapter2x2':
        Adapter=hist['AdapterPathIntModel']
        model=Adapter(base,in_dim=2,transform=torch.eye(2),adapter_init='identity')
        model.configure_trainable()
    else:
        raise ValueError(kind)
    return model.to(device)


def diagnostic_row(model,spec,cond):
    row={'condition':cond,'origin':spec['origin'],'kind':spec['kind'],'core_trainability':spec['core']}
    if spec['kind']=='fixed_gain': row['gain']=float(spec['gain'])
    if spec['kind']=='scalar':
        row['log_gain']=float(model.log_gain.detach().cpu()); row['gain']=float(model.gain().detach().cpu())
    if spec['kind']=='adapter2x2': row.update(adapter_summary(model))
    row['grid_any_trainable']=bool(any(p.requires_grad for p in model.grid_rnn.parameters()))
    return row


def compact_state_dict(model, core_trainable:bool):
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    if core_trainable:
        return state,'full'
    compact={k:v for k,v in state.items() if '.grid_rnn.' not in k and not k.startswith('grid_rnn.')}
    return compact,'frozen_core_omitted_reconstruct_from_origin_and_seed'


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source-seed',type=int,required=True)
    ap.add_argument('--target-seed',type=int,required=True)
    ap.add_argument('--group',choices=['fixed_scale','learned_interface','full_finetune'],required=True)
    args=ap.parse_args(); source=int(args.source_seed); target=int(args.target_seed); group=str(args.group)
    protocol=json.loads(PROTOCOL_PATH.read_text())
    if source not in protocol['all_sources'] or target not in protocol['target_split_seeds']:
        raise ValueError('source/target outside frozen Phase F protocol')
    conditions=list(protocol['conditions_by_group'][group])
    specs=condition_specs()
    if any(c not in specs for c in conditions): raise RuntimeError('unknown condition in protocol')

    out=OUTPUT_ROOT/f'source_{source}'/f'target_{target}'/f'group_{group}'
    if (out/'COMPLETE.json').exists():
        print('already complete',out,flush=True); return
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f'partial output exists: {out}; inspect before rerun')
    out.mkdir(parents=True,exist_ok=True)

    hist=import_historical(); device=torch.device('cuda')
    pools_nodes,manifest=build_fullscale_split(protocol,target)
    expected=protocol['expected_manifest_hashes'][str(target)]
    if manifest['manifest_hash'] != expected:
        raise RuntimeError(f'manifest hash mismatch {manifest["manifest_hash"]} != {expected}')
    pools={k:to_historical_episodes(v,hist) for k,v in pools_nodes.items()}
    for name,eps in pools.items():
        hfps=[hist['episode_fingerprint'](e) for e in eps]
        if hfps != manifest['fingerprints'][name]:
            raise RuntimeError(f'historical fingerprint mismatch: {name}')
    atomic_json(out/'dataset_manifest.json',manifest)
    write_csv(out/'dataset_audit.csv',split_audit(pools,manifest))

    train_eps=pools['train_8000']; val_eps=pools['validation_seen_short']
    test_names=['id_seen_short','heldout_short','seen_long','heldout_long']
    baseline_rows=[]
    for split in ['validation_seen_short']+test_names:
        baseline_rows.append({
            'source_seed':source,'target_seed':target,'split':split,
            'baseline':'start_plus_length','accuracy':frequency_baseline(train_eps,pools[split]),
            'dataset_manifest_hash':manifest['manifest_hash'],
        })
    write_csv(out/'baselines.csv',baseline_rows)

    nav_path=PROJECT_ROOT/'sources'/f'nav_seed_{source}'/'artifacts'/'navigation'/'best.pt'
    if not nav_path.is_file(): raise FileNotFoundError(nav_path)
    nav_hh,nav_ih,nav_meta=hist['load_navigation_grid_weights'](nav_path)
    matched_seed=stable_seed32('matched_random',target)
    matched_hh,matched_ih,matched_diag=hist['make_matched_random_weights'](
        nav_hh,nav_ih,seed=matched_seed,iterations=8,singular_rank=16,probe_count=64,strict=True,
    )
    atomic_json(out/'matched_random_diagnostics.json',matched_diag)

    model_cfg=protocol['model']; train_cfg=protocol['train']; batch_size=int(protocol['batch_size'])
    run_seed=stable_seed32('phaseF_run',target,protocol['training_examples'])
    loader_seed=stable_seed32('phaseF_loader',target,protocol['training_examples'])
    random_core_seed=stable_seed32('phaseF_fresh_random_core',source,target)
    for x in (run_seed,loader_seed,random_core_seed,matched_seed):
        if not (0 <= int(x) <= 0xFFFFFFFF): raise RuntimeError('seed outside uint32 contract')

    metrics=[]; per_length=[]; interventions=[]; updates=[]; diagnostics=[]; ckpts=[]
    cell_cls=hist['FactorialCell']
    intervention_conditions={
        'navigation_trainable_scalar_gain','matched_random_trainable_scalar_gain','fresh_random_trainable_scalar_gain',
        'navigation_trainable_input_adapter','matched_random_trainable_input_adapter',
    }

    for cond in conditions:
        spec=specs[cond]
        hist['set_seed'](run_seed)
        base=hist['build_hard_model'](model_cfg,n_nodes=64,device=device)
        configure_origin(
            base,spec,hist,nav_hh=nav_hh,nav_ih=nav_ih,matched_hh=matched_hh,matched_ih=matched_ih,
            cell_cls=cell_cls,model_cfg=model_cfg,device=device,random_core_seed=random_core_seed,
        )
        model=wrap_model(base,spec,hist,device)
        # Reset global RNG after any source-specific fresh-random core construction so
        # paired conditions enter optimization under the same non-loader RNG state.
        hist['set_seed'](run_seed)
        grid_any=any(p.requires_grad for p in model.grid_rnn.parameters())
        if spec['core']=='frozen' and grid_any: raise RuntimeError(f'grid unexpectedly trainable: {cond}')
        if spec['core']=='trainable' and not grid_any: raise RuntimeError(f'grid unexpectedly frozen: {cond}')
        if spec['kind']=='scalar' and abs(float(model.gain().detach().cpu())-1.0)>1e-8:
            raise RuntimeError('scalar did not start at unit gain')

        tr_loader=make_loader(train_eps,hist,batch_size,True,loader_seed)
        va_loader=make_loader(val_eps,hist,batch_size,False,loader_seed+1)
        training=hist['train_fixed_updates'](model,tr_loader,va_loader,device,train_cfg,seed=run_seed)
        if training.stopped_update != int(train_cfg['max_gradient_updates']):
            raise RuntimeError(f'fixed update contract violated in {cond}: {training.stopped_update}')
        model.load_state_dict(training.state_dict,strict=True)

        eval_splits=[('train_8000',train_eps),('validation_seen_short',val_eps)] + [(x,pools[x]) for x in test_names]
        evals={}
        for split,eps in eval_splits:
            loader=make_loader(eps,hist,batch_size,False,loader_seed+2)
            sm,_=hist['evaluate_loader_detailed'](model,loader,device,split_name=split,calibration_bins=15)
            evals[split]=sm
            metrics.append({
                'schema_version':SCHEMA_VERSION,'source_seed':source,'target_seed':target,'group':group,
                'condition':cond,'origin':spec['origin'],'kind':spec['kind'],'core_trainability':spec['core'],
                'split':split,'accuracy':sm['accuracy'],'nll':sm['nll'],'ece':sm['ece'],
                'mean_confidence':sm['mean_confidence'],'mean_readout_margin':sm['mean_readout_margin'],
                'dataset_manifest_hash':manifest['manifest_hash'],'best_update':training.best_update,
                'stopped_update':training.stopped_update,'run_seed':run_seed,'loader_seed':loader_seed,
                'random_core_seed':random_core_seed if spec['origin']=='fresh_random' else '',
            })
            for L,acc in sm['accuracy_by_path_length'].items():
                per_length.append({
                    'source_seed':source,'target_seed':target,'group':group,'condition':cond,
                    'split':split,'path_length':int(L),'accuracy':float(acc),
                })

        if cond in intervention_conditions:
            n=int(protocol['intervention_sample_per_test_split'])
            for split in test_names:
                sample=deterministic_stratified_subsample(
                    pools[split],n,seed_parts=(source,target,cond,split),
                )
                interventions.append({
                    'source_seed':source,'target_seed':target,'group':group,'condition':cond,
                    'split':split,'n_sample':len(sample),
                    **intervention_summary(model,sample,hist,device),
                })

        for h in training.history:
            updates.append({'source_seed':source,'target_seed':target,'group':group,'condition':cond,**h})
        drow={'source_seed':source,'target_seed':target,'group':group,'best_update':training.best_update,**diagnostic_row(model,spec,cond)}
        diagnostics.append(drow)

        state,storage_mode=compact_state_dict(model,spec['core']=='trainable')
        ckpt_dir=out/'checkpoints'/cond; ckpt_dir.mkdir(parents=True,exist_ok=True)
        ckpt=ckpt_dir/'best_validation.pt'
        torch.save({
            'schema_version':SCHEMA_VERSION,'model_state':state,'checkpoint_storage_mode':storage_mode,
            'model_config':model_cfg,'source_seed':source,'target_seed':target,'group':group,
            'condition':cond,'condition_spec':spec,'run_seed':run_seed,'loader_seed':loader_seed,
            'random_core_seed':random_core_seed if spec['origin']=='fresh_random' else None,
            'matched_random_seed':matched_seed,'best_update':training.best_update,
            'validation_metrics':{k:v for k,v in evals['validation_seen_short'].items() if not isinstance(v,dict)},
            'dataset_manifest_hash':manifest['manifest_hash'],'navigation_checkpoint':str(nav_path),
            'navigation_checkpoint_sha256':nav_meta['checkpoint_sha256'],
        },ckpt)
        ckpts.append({'source_seed':source,'target_seed':target,'group':group,'condition':cond,'path':str(ckpt),'storage_mode':storage_mode,'best_update':training.best_update})
        print(source,target,group,cond,'val',evals['validation_seen_short']['accuracy'],'ID',evals['id_seen_short']['accuracy'],'best',training.best_update,flush=True)
        del model,base
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    write_csv(out/'metrics.csv',metrics); write_csv(out/'per_length.csv',per_length)
    write_csv(out/'relation_interventions.csv',interventions); write_csv(out/'update_curves.csv',updates)
    write_csv(out/'condition_diagnostics.csv',diagnostics); write_csv(out/'checkpoint_manifest.csv',ckpts)
    atomic_json(out/'COMPLETE.json',{
        'schema_version':SCHEMA_VERSION,'status':'complete','source_seed':source,'target_seed':target,'group':group,
        'dataset_manifest_hash':manifest['manifest_hash'],'n_conditions':len(conditions),'conditions':conditions,
    })


if __name__=='__main__':
    main()
