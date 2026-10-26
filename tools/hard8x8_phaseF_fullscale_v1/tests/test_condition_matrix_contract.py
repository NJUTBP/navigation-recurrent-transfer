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
h['set_seed'](11); donor=h['build_hard_model'](cfg,n_nodes=64,device=device)
nav_hh=donor.grid_rnn.cell.h_proj.weight.detach().cpu().clone(); nav_ih=donor.grid_rnn.cell.in_proj.weight.detach().cpu().clone()
matched_hh,matched_ih,_=h['make_matched_random_weights'](nav_hh,nav_ih,seed=22,iterations=4,singular_rank=4,probe_count=8,strict=False)
cell=h['FactorialCell']
for i,(name,spec) in enumerate(condition_specs().items()):
    h['set_seed'](1234); base=h['build_hard_model'](cfg,n_nodes=64,device=device)
    configure_origin(base,spec,h,nav_hh=nav_hh,nav_ih=nav_ih,matched_hh=matched_hh,matched_ih=matched_ih,cell_cls=cell,model_cfg=cfg,device=device,random_core_seed=100+i)
    model=wrap_model(base,spec,h,device)
    grid=any(p.requires_grad for p in model.grid_rnn.parameters())
    assert grid == (spec['core']=='trainable'), (name,grid,spec)
    if spec['kind']=='scalar': assert abs(float(model.gain().detach())-1.0)<1e-8
print('PASS all 15 condition construction/trainability contracts')
