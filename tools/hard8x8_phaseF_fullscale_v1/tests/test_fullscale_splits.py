#!/usr/bin/env python3
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from fullscale_common import build_fullscale_split,_contains_heldout
p=json.loads((ROOT/'PROTOCOL.json').read_text())
for seed in p['target_split_seeds']:
    pools,m=build_fullscale_split(p,seed)
    assert m['manifest_hash']==p['expected_manifest_hashes'][str(seed)]
    assert {k:len(v) for k,v in pools.items()}=={
        'train_8000':8000,'validation_seen_short':1000,'id_seen_short':2000,
        'heldout_short':2000,'seen_long':2000,'heldout_long':2000}
    held={tuple(x) for x in p['held_out_action_bigrams']}
    assert all(not _contains_heldout(x,held) for x in pools['train_8000'])
    assert all(not _contains_heldout(x,held) for x in pools['validation_seen_short'])
    assert all(not _contains_heldout(x,held) for x in pools['id_seen_short'])
    assert all(_contains_heldout(x,held) for x in pools['heldout_short'])
    assert all(not _contains_heldout(x,held) for x in pools['seen_long'])
    assert all(_contains_heldout(x,held) for x in pools['heldout_long'])
print('PASS full-scale fresh orthogonal split hashes')
