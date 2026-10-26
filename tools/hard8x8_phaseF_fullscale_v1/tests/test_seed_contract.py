#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from fullscale_common import stable_seed32
p=json.loads((ROOT/'PROTOCOL.json').read_text())
for s in p['all_sources']:
    for t in p['target_split_seeds']:
        vals=[stable_seed32('phaseF_run',t,8000),stable_seed32('phaseF_loader',t,8000),stable_seed32('phaseF_fresh_random_core',s,t),stable_seed32('matched_random',t)]
        assert all(0<=x<=2**32-1 for x in vals)
print('PASS Phase F uint32 seed contract')
