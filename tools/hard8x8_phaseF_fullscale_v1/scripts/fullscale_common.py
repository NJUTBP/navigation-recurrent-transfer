#!/usr/bin/env python3
from __future__ import annotations
import os

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(os.environ.get('NRT_PROJECT_ROOT', str(Path(__file__).resolve().parents[3])))
CODE_ROOT = PROJECT_ROOT / 'code'
PYTHON_BIN = Path(sys.executable)

@dataclass(frozen=True)
class BridgeEpisode:
    nodes: tuple[int,...]
    n_levels: int = 8
    scale: float = 0.02
    @property
    def start_node(self): return int(self.nodes[0])
    @property
    def target_node(self): return int(self.nodes[-1])
    @property
    def length(self): return len(self.nodes)-1
    def coord(self,node:int):
        r,c=divmod(int(node),self.n_levels)
        return np.asarray([r,c],dtype=np.float32)*np.float32(self.scale)
    @property
    def velocities(self):
        out=[]
        for a,b in zip(self.nodes[:-1],self.nodes[1:]):
            ar,ac=divmod(a,self.n_levels); br,bc=divmod(b,self.n_levels)
            out.append(np.asarray([br-ar,bc-ac],dtype=np.float32)*np.float32(self.scale))
        return np.stack(out).astype(np.float32)
    @property
    def path_coords(self): return np.stack([self.coord(x) for x in self.nodes]).astype(np.float32)


def atomic_json(path:Path,obj:Any):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n')
    tmp.replace(path)


def write_csv(path:Path,rows:list[dict[str,Any]]):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('')
        return
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields: fields.append(k)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k:(json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v) for k,v in row.items()})


def hash_jsonable(x:Any)->str:
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def stable_seed32(*parts: Any) -> int:
    """Deterministically derive a NumPy legacy-compatible uint32 seed.

    Historical gpm_cmap.utils.set_seed ultimately calls np.random.seed(), which
    requires 0 <= seed <= 2**32 - 1.  Do not use arithmetic seed formulas that
    can overflow that contract for large target identifiers such as 82001.
    """
    payload = json.dumps(parts, separators=(',', ':'), default=str).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], byteorder='little', signed=False)


def neighbors(node:int,n:int=8):
    r,c=divmod(node,n); out=[]
    if r+1<n: out.append((r+1)*n+c)
    if r>0: out.append((r-1)*n+c)
    if c+1<n: out.append(r*n+c+1)
    if c>0: out.append(r*n+c-1)
    return out


def action_id(a:int,b:int,n:int=8)->int:
    ar,ac=divmod(a,n); br,bc=divmod(b,n)
    dr,dc=br-ar,bc-ac
    if dr==1:return 0
    if dr==-1:return 1
    if dc==1:return 2
    if dc==-1:return 3
    raise ValueError((a,b))


def enumerate_allowed_paths(n:int=8,max_len:int=5,held=frozenset({(0,2),(1,3),(2,1),(3,0)})):
    by_len={L:[] for L in range(1,max_len+1)}
    def rec(path:list[int],acts:list[int]):
        L=len(path)-1
        if L>=1:
            if not any((acts[i],acts[i+1]) in held for i in range(len(acts)-1)):
                by_len[L].append(tuple(path))
        if L>=max_len:return
        for nxt in neighbors(path[-1],n):
            rec(path+[nxt],acts+[action_id(path[-1],nxt,n)])
    for s in range(n*n): rec([s],[])
    return by_len


def _fingerprint_nodes(nodes:Sequence[int],n:int=8,scale:float=.02)->str:
    start,target=int(nodes[0]),int(nodes[-1])
    def coord(x):
        r,c=divmod(int(x),n); return np.asarray([r,c],dtype=np.float32)*np.float32(scale)
    path=np.stack([coord(x) for x in nodes]).astype(np.float32)
    vel=[]
    for a,b in zip(nodes[:-1],nodes[1:]):
        ar,ac=divmod(a,n); br,bc=divmod(b,n)
        vel.append(np.asarray([br-ar,bc-ac],dtype=np.float32)*np.float32(scale))
    vel=np.stack(vel).astype(np.float32)
    h=hashlib.sha256(); h.update(np.asarray([start,target],dtype=np.int64).tobytes())
    for arr in (vel,coord(start),coord(target),path):
        arr=np.ascontiguousarray(arr,dtype=np.float32)
        h.update(np.asarray(arr.shape,dtype=np.int64).tobytes()); h.update(arr.tobytes())
    return h.hexdigest()


def _largest_remainder_total(capacity_by_start:dict[int,int], total:int)->dict[int,int]:
    """Select `total` items proportional to available support across starts."""
    n=sum(capacity_by_start.values())
    if total>n: raise ValueError((total,n))
    ideal={s:capacity_by_start[s]*total/n for s in capacity_by_start}
    out={s:int(np.floor(v)) for s,v in ideal.items()}
    remaining=total-sum(out.values())
    order=sorted(capacity_by_start, key=lambda s:(-(ideal[s]-out[s]), s))
    for s in order[:remaining]: out[s]+=1
    if any(out[s]>capacity_by_start[s] for s in out): raise RuntimeError('allocation exceeds capacity')
    return out


def _partition_selected_by_start(selected_by_start:dict[int,list[tuple[int,...]]], quotas:dict[str,int]):
    """Partition selected support into splits while matching the same start distribution."""
    total=sum(len(v) for v in selected_by_start.values())
    if sum(quotas.values())!=total: raise ValueError((sum(quotas.values()),total))
    names=list(quotas)
    alloc={s:{name:0 for name in names} for s in selected_by_start}
    fractions=[]
    for s,rows in selected_by_start.items():
        m=len(rows)
        for name in names:
            ideal=m*quotas[name]/total
            base=int(np.floor(ideal)); alloc[s][name]=base
            fractions.append((ideal-base,s,name))
    deficits={name:quotas[name]-sum(alloc[s][name] for s in alloc) for name in names}
    leftovers={s:len(selected_by_start[s])-sum(alloc[s].values()) for s in alloc}
    # Greedy largest-remainder assignment with exact global quotas and exact per-start totals.
    while sum(deficits.values()):
        candidates=[]
        for frac,s,name in fractions:
            if deficits[name]>0 and leftovers[s]>0:
                candidates.append((frac,s,name))
        if not candidates: raise RuntimeError(f'cannot resolve split deficits {deficits} leftovers={leftovers}')
        frac,s,name=max(candidates,key=lambda x:(x[0],-x[1],x[2]))
        alloc[s][name]+=1; deficits[name]-=1; leftovers[s]-=1
        fractions=[x for x in fractions if not (x[1]==s and x[2]==name)]
    if any(leftovers.values()) or any(deficits.values()): raise RuntimeError('partition rounding failure')
    out={name:[] for name in names}
    for s in sorted(selected_by_start):
        rows=selected_by_start[s]; offset=0
        for name in names:
            n=alloc[s][name]; out[name].extend(rows[offset:offset+n]); offset+=n
        if offset!=len(rows): raise RuntimeError('start partition length mismatch')
    return out



def _contains_heldout(nodes: Sequence[int], held: set[tuple[int,int]]) -> bool:
    acts=[action_id(a,b) for a,b in zip(nodes[:-1],nodes[1:])]
    return any((acts[i],acts[i+1]) in held for i in range(len(acts)-1))


def enumerate_short_paths_by_composition(
    n:int=8,
    max_len:int=5,
    held=frozenset({(0,2),(1,3),(2,1),(3,0)}),
):
    """Enumerate complete short-path support separated by held-out composition."""
    seen={L:[] for L in range(1,max_len+1)}
    heldout={L:[] for L in range(1,max_len+1)}
    def rec(path:list[int],acts:list[int]):
        L=len(path)-1
        if L>=1:
            has=any((acts[i],acts[i+1]) in held for i in range(len(acts)-1))
            (heldout if has else seen)[L].append(tuple(path))
        if L>=max_len:
            return
        for nxt in neighbors(path[-1],n):
            rec(path+[nxt],acts+[action_id(path[-1],nxt,n)])
    for s in range(n*n):
        rec([s],[])
    return seen,heldout


def _select_stratified_paths(
    paths_by_len: Mapping[int, Sequence[tuple[int,...]]],
    quotas: Mapping[int,int],
    *,
    split_seed:int,
    seed_offset:int,
) -> list[tuple[int,...]]:
    out=[]
    for L in sorted(quotas):
        q=int(quotas[L])
        groups=defaultdict(list)
        for path in paths_by_len[int(L)]:
            groups[int(path[0])].append(tuple(path))
        for st in groups:
            rng=np.random.default_rng(stable_seed32('short_select',split_seed,seed_offset,L,st))
            order=rng.permutation(len(groups[st]))
            groups[st]=[groups[st][i] for i in order]
        take=_largest_remainder_total({st:len(rows) for st,rows in groups.items()}, q)
        for st in sorted(groups):
            out.extend(groups[st][:take[st]])
    rng=np.random.default_rng(stable_seed32('short_shuffle',split_seed,seed_offset))
    order=rng.permutation(len(out))
    return [out[i] for i in order]


def _allocate_even_starts(total:int,n_starts:int=64)->dict[int,int]:
    base=total//n_starts; rem=total%n_starts
    return {s:base+(1 if s<rem else 0) for s in range(n_starts)}


def _generate_long_pool(
    *,
    length_quotas: Mapping[int,int],
    split_seed:int,
    require_heldout:bool,
    held:set[tuple[int,int]],
    forbidden:set[str],
    seed_offset:int,
) -> list[tuple[int,...]]:
    """Generate deterministic unique long walks, balanced over start node within length."""
    result=[]
    seen=set()
    for L in sorted(int(x) for x in length_quotas):
        q=int(length_quotas[L])
        start_need=_allocate_even_starts(q,64)
        for st in range(64):
            need=start_need[st]
            if need<=0:
                continue
            rng=np.random.default_rng(stable_seed32('long',split_seed,seed_offset,L,st))
            accepted=0
            attempts=0
            while accepted<need:
                attempts += 1
                if attempts > 250000:
                    raise RuntimeError(
                        f'could not generate long pool L={L} start={st} '
                        f'require_heldout={require_heldout} accepted={accepted}/{need}'
                    )
                path=[st]
                acts=[]
                for _ in range(L):
                    nbr=neighbors(path[-1],8)
                    nxt=int(nbr[int(rng.integers(0,len(nbr)))])
                    acts.append(action_id(path[-1],nxt,8))
                    path.append(nxt)
                has=any((acts[i],acts[i+1]) in held for i in range(len(acts)-1))
                if bool(has) != bool(require_heldout):
                    continue
                tup=tuple(path)
                fp=_fingerprint_nodes(tup)
                if fp in forbidden or fp in seen:
                    continue
                seen.add(fp); result.append(tup); accepted += 1
    rng=np.random.default_rng(stable_seed32('long_shuffle',split_seed,seed_offset,require_heldout))
    order=rng.permutation(len(result))
    return [result[i] for i in order]


def build_fullscale_split(protocol:Mapping[str,Any], split_seed:int):
    held={tuple(int(v) for v in pair) for pair in protocol['held_out_action_bigrams']}
    seen_short, held_short = enumerate_short_paths_by_composition(held=frozenset(held))
    expected_seen={int(k):int(v) for k,v in protocol['support_counts_seen_short'].items()}
    expected_held={int(k):int(v) for k,v in protocol['support_counts_heldout_short'].items()}
    actual_seen={L:len(v) for L,v in seen_short.items()}
    actual_held={L:len(v) for L,v in held_short.items()}
    if actual_seen != expected_seen:
        raise RuntimeError(f'seen short support mismatch: {actual_seen} != {expected_seen}')
    if actual_held != expected_held:
        raise RuntimeError(f'heldout short support mismatch: {actual_held} != {expected_held}')

    q_train={int(k):int(v) for k,v in protocol['split_length_quotas']['train_8000'].items()}
    q_val={int(k):int(v) for k,v in protocol['split_length_quotas']['validation_seen_short_1000'].items()}
    q_id={int(k):int(v) for k,v in protocol['split_length_quotas']['id_seen_short_2000'].items()}

    train_by_len={}; val=[]; id_test=[]
    for L in range(1,6):
        groups=defaultdict(list)
        for path in seen_short[L]:
            groups[int(path[0])].append(tuple(path))
        for st in groups:
            rng=np.random.default_rng(stable_seed32('seen_short',split_seed,L,st))
            order=rng.permutation(len(groups[st]))
            groups[st]=[groups[st][i] for i in order]
        total_needed=q_train[L]+q_val[L]+q_id[L]
        select_counts=_largest_remainder_total({st:len(rows) for st,rows in groups.items()}, total_needed)
        selected={st:groups[st][:select_counts[st]] for st in groups}
        parts=_partition_selected_by_start(
            selected,
            {'train':q_train[L],'validation':q_val[L],'id_seen_short':q_id[L]},
        )
        train_by_len[L]=parts['train']
        val.extend(parts['validation'])
        id_test.extend(parts['id_seen_short'])

    train=[p for L in range(1,6) for p in train_by_len[L]]
    def shuffle(xs, label):
        rng=np.random.default_rng(stable_seed32('shuffle',split_seed,label))
        order=rng.permutation(len(xs))
        return [xs[i] for i in order]
    train=shuffle(train,'train')
    val=shuffle(val,'validation')
    id_test=shuffle(id_test,'id_seen_short')

    comp_q={int(k):int(v) for k,v in protocol['split_length_quotas']['heldout_short_2000'].items()}
    comp=_select_stratified_paths(
        held_short, comp_q, split_seed=split_seed, seed_offset=91001,
    )

    forbidden={_fingerprint_nodes(p) for p in train+val+id_test+comp}
    long_seen_q={int(k):int(v) for k,v in protocol['split_length_quotas']['seen_long_2000'].items()}
    long_held_q={int(k):int(v) for k,v in protocol['split_length_quotas']['heldout_long_2000'].items()}
    long_seen=_generate_long_pool(
        length_quotas=long_seen_q, split_seed=split_seed, require_heldout=False,
        held=held, forbidden=forbidden, seed_offset=92001,
    )
    forbidden.update(_fingerprint_nodes(p) for p in long_seen)
    long_held=_generate_long_pool(
        length_quotas=long_held_q, split_seed=split_seed, require_heldout=True,
        held=held, forbidden=forbidden, seed_offset=93001,
    )

    pools={
        'train_8000':train,
        'validation_seen_short':val,
        'id_seen_short':id_test,
        'heldout_short':comp,
        'seen_long':long_seen,
        'heldout_long':long_held,
    }
    fps={name:[_fingerprint_nodes(p) for p in rows] for name,rows in pools.items()}
    names=list(pools)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            overlap=set(fps[a]) & set(fps[b])
            if overlap:
                raise RuntimeError(f'fingerprint overlap {a} vs {b}: {len(overlap)}')
    for p in train+val+id_test+long_seen:
        if _contains_heldout(p,held):
            raise RuntimeError('seen-composition pool contains held-out bigram')
    for p in comp+long_held:
        if not _contains_heldout(p,held):
            raise RuntimeError('held-out-composition pool lacks held-out bigram')
    if any(len(p)-1 > 5 for p in train+val+id_test+comp):
        raise RuntimeError('short pool contains long trajectory')
    if any(len(p)-1 < 6 for p in long_seen+long_held):
        raise RuntimeError('long pool contains short trajectory')

    manifest={
        'schema_version':'hard8x8_fullscale_orthogonal_split_v1',
        'split_seed':int(split_seed),
        'n_levels':8,
        'velocity_scale':0.02,
        'coord_scale':0.02,
        'held_out_action_bigrams':[list(x) for x in sorted(held)],
        'support_counts_seen_short':actual_seen,
        'support_counts_heldout_short':actual_held,
        'counts':{k:len(v) for k,v in pools.items()},
        'length_counts':{k:dict(sorted(Counter(len(x)-1 for x in v).items())) for k,v in pools.items()},
        'start_counts':{k:dict(sorted(Counter(int(x[0]) for x in v).items())) for k,v in pools.items()},
        'target_counts':{k:dict(sorted(Counter(int(x[-1]) for x in v).items())) for k,v in pools.items()},
        'fingerprints':fps,
        'fingerprint_set_hashes':{k:hash_jsonable(sorted(v)) for k,v in fps.items()},
        'test_factorial':{
            'seen_short':{'composition':'seen','horizon':'short','pool':'id_seen_short'},
            'heldout_short':{'composition':'heldout','horizon':'short','pool':'heldout_short'},
            'seen_long':{'composition':'seen','horizon':'long','pool':'seen_long'},
            'heldout_long':{'composition':'heldout','horizon':'long','pool':'heldout_long'},
        },
        'selection_rule':(
            'seen short support fully enumerated; train/validation/ID disjoint and start-stratified; '
            'held-out short support fully enumerated and independently start-stratified; '
            'long pools generated deterministically with exact length/start quotas; '
            'composition and horizon crossed orthogonally; all pools fingerprint-disjoint'
        ),
    }
    manifest['manifest_hash']=hash_jsonable({k:v for k,v in manifest.items() if k!='manifest_hash'})
    return pools,manifest


def deterministic_stratified_subsample(episodes, n:int, *, seed_parts:Sequence[Any]):
    """Length-stratified deterministic subsample for interventions."""
    if n >= len(episodes):
        return tuple(episodes)
    by_len=defaultdict(list)
    for ep in episodes:
        by_len[int(ep.velocities.shape[0])].append(ep)
    lengths=sorted(by_len)
    per={L:n//len(lengths) for L in lengths}
    for L in lengths[: n-sum(per.values())]:
        per[L]+=1
    out=[]
    for L in lengths:
        rows=list(by_len[L])
        rng=np.random.default_rng(stable_seed32('intervention_sample',*seed_parts,L))
        order=rng.permutation(len(rows))
        out.extend(rows[i] for i in order[:min(per[L],len(rows))])
    if len(out)<n:
        used={id(x) for x in out}
        rest=[x for x in episodes if id(x) not in used]
        rng=np.random.default_rng(stable_seed32('intervention_fill',*seed_parts))
        order=rng.permutation(len(rest))
        out.extend(rest[i] for i in order[:n-len(out)])
    return tuple(out[:n])

def import_historical():
    sys.path.insert(0,str(CODE_ROOT))
    import gpm_cmap
    resolved=Path(gpm_cmap.__file__).resolve()
    if CODE_ROOT.resolve() not in resolved.parents: raise RuntimeError(f'wrong gpm_cmap import: {resolved}')
    from gpm_cmap.analysis.hard_pathint import build_hard_model, evaluate_loader_detailed, train_fixed_updates
    from gpm_cmap.analysis.recurrent_factorial import FactorialCell, install_factorial_weights, load_navigation_grid_weights, make_matched_random_weights
    from gpm_cmap.data.graph_tasks.pathint_episodes import PathIntEpisode, PathIntDataset, pad_collate, episode_fingerprint
    from gpm_cmap.utils.seed import make_torch_generator, make_worker_init_fn
    from gpm_cmap.utils import set_seed
    from gpm_cmap.models.adapter_wrap import AdapterPathIntModel
    return locals()


def to_historical_episodes(nodes_list, hist):
    PathIntEpisode=hist['PathIntEpisode']
    out=[]
    for nodes in nodes_list:
        ep=BridgeEpisode(tuple(nodes))
        pc=ep.path_coords; sc=pc[0]; tc=pc[-1]
        out.append(PathIntEpisode(
            start_node=ep.start_node, velocities=ep.velocities, target_node=ep.target_node,
            start_coords=sc.copy(), target_coords=tc.copy(), path_coords=pc,
            net_l1_distance=float(np.abs(tc-sc).sum()), path_unique_nodes=len(set(nodes)),
        ))
    return tuple(out)


def make_loader(episodes,hist,batch_size:int,shuffle:bool,seed:int):
    from torch.utils.data import DataLoader
    return DataLoader(hist['PathIntDataset'](list(episodes)),batch_size=batch_size,shuffle=shuffle,
        collate_fn=hist['pad_collate'],num_workers=0,
        worker_init_fn=hist['make_worker_init_fn'](int(seed)),
        generator=hist['make_torch_generator'](int(seed)) if shuffle else None)


def frequency_baseline(train_eps,test_eps):
    by_sl=defaultdict(Counter); by_s=defaultdict(Counter); global_c=Counter()
    for ep in train_eps:
        L=int(ep.velocities.shape[0]); by_sl[(int(ep.start_node),L)][int(ep.target_node)]+=1
        by_s[int(ep.start_node)][int(ep.target_node)]+=1; global_c[int(ep.target_node)]+=1
    global_pred=global_c.most_common(1)[0][0]
    correct=0
    for ep in test_eps:
        key=(int(ep.start_node),int(ep.velocities.shape[0]))
        c=by_sl.get(key)
        if c: pred=c.most_common(1)[0][0]
        elif by_s.get(int(ep.start_node)): pred=by_s[int(ep.start_node)].most_common(1)[0][0]
        else: pred=global_pred
        correct += int(pred==int(ep.target_node))
    return correct/len(test_eps)


def tv_distance(values_a,values_b,ncat:int):
    ca=np.bincount(np.asarray(values_a,dtype=int),minlength=ncat).astype(float); cb=np.bincount(np.asarray(values_b,dtype=int),minlength=ncat).astype(float)
    ca/=ca.sum(); cb/=cb.sum(); return .5*np.abs(ca-cb).sum()


def split_audit(pools_hist,manifest):
    rows=[]
    for name,eps in pools_hist.items():
        starts=[int(e.start_node) for e in eps]; targets=[int(e.target_node) for e in eps]; lens=[int(e.velocities.shape[0]) for e in eps]
        rows.append({'split':name,'n':len(eps),'mean_length':float(np.mean(lens)),
                     'start_unique':len(set(starts)),'target_unique':len(set(targets)),
                     'length_counts':dict(sorted(Counter(lens).items()))})
    ref=pools_hist['train_8000']
    for row in rows:
        eps=pools_hist[row['split']]
        row['start_tv_vs_train8000']=tv_distance([e.start_node for e in ref],[e.start_node for e in eps],64)
        row['target_tv_vs_train8000']=tv_distance([e.target_node for e in ref],[e.target_node for e in eps],64)
    return rows


@torch.no_grad()
def logits_for(model,episodes,hist,device,batch_size=128,zero=False):
    loader=make_loader(episodes,hist,batch_size,False,1)
    logits=[]; targets=[]
    model.eval()
    for batch in loader:
        start=batch['start_node'].to(device); vel=batch['velocities'].to(device); lengths=batch['lengths'].to(device)
        if zero: vel=torch.zeros_like(vel)
        out=model(start,vel,lengths=lengths,validate_inputs=False)
        logits.append(out['node_logits'].cpu()); targets.append(batch['target_node'].cpu())
    return torch.cat(logits),torch.cat(targets)


def counterfactual_swap(episodes):
    groups=defaultdict(list)
    for i,e in enumerate(episodes): groups[(int(e.start_node),int(e.velocities.shape[0]))].append(i)
    partner={}
    for key,inds in groups.items():
        if len(inds)<2: continue
        for i in inds:
            candidates=[j for j in inds if int(episodes[j].target_node)!=int(episodes[i].target_node)]
            if candidates:
                h=int(hashlib.sha256(f'{key}|{i}'.encode()).hexdigest()[:16],16)
                partner[i]=candidates[h%len(candidates)]
    originals=[]; swaps=[]
    for i,j in sorted(partner.items()):
        originals.append(episodes[i]); swaps.append(episodes[j])
    return originals,swaps


def intervention_summary(model,test_eps,hist,device):
    real_logits,targets=logits_for(model,test_eps,hist,device)
    zero_logits,_=logits_for(model,test_eps,hist,device,zero=True)
    real_lp=F.log_softmax(real_logits,dim=-1); zero_lp=F.log_softmax(zero_logits,dim=-1)
    idx=torch.arange(len(targets)); real_pred=real_logits.argmax(1); zero_pred=zero_logits.argmax(1)
    rows={
        'real_accuracy':float((real_pred==targets).float().mean()),
        'zero_accuracy':float((zero_pred==targets).float().mean()),
        'zero_true_logp_drop':float((real_lp[idx,targets]-zero_lp[idx,targets]).mean()),
        'zero_prediction_change':float((real_pred!=zero_pred).float().mean()),
    }
    orig,swap=counterfactual_swap(test_eps)
    if orig:
        olog,ot=logits_for(model,orig,hist,device); slog,st=logits_for(model,swap,hist,device)
        olp=F.log_softmax(olog,dim=-1); slp=F.log_softmax(slog,dim=-1); ii=torch.arange(len(st))
        rows.update({
            'n_counterfactual_pairs':len(st),
            'counterfactual_accuracy':float((slog.argmax(1)==st).float().mean()),
            'counterfactual_target_logp_gain':float((slp[ii,st]-olp[ii,st]).mean()),
            'counterfactual_prediction_change':float((olog.argmax(1)!=slog.argmax(1)).float().mean()),
        })
    return rows


