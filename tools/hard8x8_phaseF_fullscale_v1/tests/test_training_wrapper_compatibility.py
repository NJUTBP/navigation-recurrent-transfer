#!/usr/bin/env python3
import sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import fullscale_common as common
from run_group import condition_specs,configure_origin,wrap_model
local=Path('/mnt/data/a1_capture_extract/historical_code')
if local.is_dir(): common.CODE_ROOT=local
h=common.import_historical(); device=torch.device('cpu')
cfg={'hidden_dim':32,'content_dim':16,'nonnegative':True,'cell_type':'rnn','use_bias':False,'dropout':0.0,'periodic_scales':0,'use_position_head':False,'start_init':'embedding','coord_dim':2}
# small deterministic episode set
nodes=[]
for s in range(16):
    r,c=divmod(s,8)
    if c<7: nodes.append((s,s+1))
    if r<7: nodes.append((s,s+8))
eps=common.to_historical_episodes(nodes,h); train=eps[:20]; val=eps[20:]
h['set_seed'](11); donor=h['build_hard_model'](cfg,n_nodes=64,device=device)
nav_hh=donor.grid_rnn.cell.h_proj.weight.detach().cpu().clone(); nav_ih=donor.grid_rnn.cell.in_proj.weight.detach().cpu().clone()
matched_hh,matched_ih,_=h['make_matched_random_weights'](nav_hh,nav_ih,seed=22,iterations=4,singular_rank=4,probe_count=8,strict=False)
cell=h['FactorialCell']; specs=condition_specs()
check=['navigation_fixed_gain3_frozen','navigation_trainable_scalar_gain','navigation_trainable_input_adapter','navigation_full_core_trainable','fresh_random_full_core_trainable']
for i,name in enumerate(check):
    spec=specs[name]; h['set_seed'](100); base=h['build_hard_model'](cfg,n_nodes=64,device=device)
    configure_origin(base,spec,h,nav_hh=nav_hh,nav_ih=nav_ih,matched_hh=matched_hh,matched_ih=matched_ih,cell_cls=cell,model_cfg=cfg,device=device,random_core_seed=500+i)
    model=wrap_model(base,spec,h,device); h['set_seed'](100)
    tr=common.make_loader(train,h,8,True,777); va=common.make_loader(val,h,8,False,778)
    res=h['train_fixed_updates'](model,tr,va,device,{'max_gradient_updates':2,'minimum_updates':2,'eval_every_updates':1,'early_stopping_patience_evals':99,'lr':1e-3,'weight_decay':0,'whh_weight_decay':1e-4,'grad_clip':1,'position_loss_weight':0,'calibration_bins':5},seed=100)
    assert res.stopped_update==2, (name,res.stopped_update)
print('PASS historical train_fixed_updates across representative Phase F condition kinds')
