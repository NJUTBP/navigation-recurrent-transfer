#!/usr/bin/env python3
import sys,torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import fullscale_common as common
from run_group import FixedGainPathIntModel,ScalarGainPathIntModel
# Use captured historical source locally when available; use cluster code in production.
local=Path('/mnt/data/a1_capture_extract/historical_code')
if local.is_dir(): common.CODE_ROOT=local
h=common.import_historical(); device=torch.device('cpu')
cfg={'hidden_dim':32,'content_dim':16,'nonnegative':True,'cell_type':'rnn','use_bias':False,'dropout':0.0,'periodic_scales':0,'use_position_head':False,'start_init':'embedding','coord_dim':2}
h['set_seed'](123); base=h['build_hard_model'](cfg,n_nodes=64,device=device); base.freeze_grid_rnn_all()
x=torch.tensor([0,1]); v=torch.tensor([[[.02,0.],[0.,.02]],[[.02,0.],[-.02,0.]]]); L=torch.tensor([2,2])
base.eval();
with torch.no_grad(): y0=base(x,v,lengths=L,validate_inputs=False)['node_logits']
fixed=FixedGainPathIntModel(base,1.0); fixed.eval()
with torch.no_grad(): y1=fixed(x,v,lengths=L,validate_inputs=False)['node_logits']
assert float((y0-y1).abs().max())<1e-7
scalar=ScalarGainPathIntModel(base); scalar.configure_trainable(); scalar.eval()
with torch.no_grad(): y2=scalar(x,v,lengths=L,validate_inputs=False)['node_logits']
assert float((y0-y2).abs().max())<1e-7
assert not any(p.requires_grad for p in scalar.grid_rnn.parameters())
assert scalar.log_gain.requires_grad
print('PASS fixed/scalar wrapper identity and freeze contract')
