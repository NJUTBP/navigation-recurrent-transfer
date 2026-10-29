#!/usr/bin/env python3
"""Level-2 check: rebuild released frozen-core target models and recompute their test accuracies.

Families: phasef (Fig. 6, S5, S8; conditions navigation/matched_random/fresh_random with scalar gain),
e09a (S7), e6 (Fig. 7b-d, S12), e6i (Fig. 7e, S12). For every model the four test-set accuracies are
recomputed and compared with the stored metrics.csv of the original run (tolerance 0.002).

  NRT_PROJECT_ROOT=/path/to/project python tools/reevaluation/reevaluate_models.py --family e6i --out outputs/reeval_e6i.csv
"""
import argparse, csv, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reevaluation_common as nc  # noqa: E402


def specs(family):
    if family in ('phasef', 'e09a'):
        targets = nc.TARGETS if family == 'phasef' else nc.TARGETS_32K
        conds = ['navigation_trainable_scalar_gain', 'matched_random_trainable_scalar_gain'] + \
                (['fresh_random_trainable_scalar_gain'] if family == 'phasef' else [])
        return [{'kind': family, 'source': s, 'target': t, 'condition': c} for s in nc.SOURCES for t in targets for c in conds]
    if family == 'e6':
        return [{'kind': 'e6', 'source': s, 'target': t, 'rho': r} for s in nc.SOURCES for t in nc.TARGETS_32K
                for r in (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.9)]
    if family == 'e6i':
        return [{'kind': 'e6i', 'source': s, 'target': t, 'arm': a} for s in nc.SOURCES for t in nc.TARGETS_32K
                for a in ('nav_impair', 'rho01_rescue')]
    raise SystemExit(family)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--family', required=True, choices=['phasef', 'e09a', 'e6', 'e6i'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int, default=0, help='evaluate only the first N models (smoke test)')
    a = ap.parse_args()
    nc.require_released_packages()
    torch = nc.import_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fullscale, phasef = nc.load_phasef_modules()
    hist = fullscale.import_historical()
    protocol = nc.derive_protocol(updates=8000, targets=nc.TARGETS, schema_version='reevaluation_v1', status='reevaluation')
    resolver = nc.ModelResolver(hist=hist, fullscale=fullscale, phasef=phasef, model_cfg=protocol['model'])
    rows, pools_cache = [], {}
    todo = specs(a.family)[: a.limit or None]
    for spec in todo:
        t = spec['target']
        if t not in pools_cache:
            pools_cache = {t: nc.build_split(fullscale, hist, protocol, t)[0]}
        pools = pools_cache[t]
        core = resolver.resolve(spec)
        model, _, _ = nc.rebuild_frozen_target(core['checkpoint'], hist=hist, phasef=phasef, device=device,
                                               core_hh=core['hh'].to(torch.float32), core_ih=core['ih'].to(torch.float32),
                                               model_cfg=protocol['model'])
        seeds = nc.phasef_seeds(fullscale, protocol, spec['source'], t)
        summaries = {s: nc.evaluate_episodes(model, pools[s], hist=hist, fullscale=fullscale, device=device,
                                             loader_seed=seeds['loader_seed'], split_name=s)[0] for s in nc.TEST_SPLITS}
        gate, ok = nc.gate_r(summaries, nc.reference_rows(core))
        for g in gate:
            rows.append({**{k: v for k, v in spec.items()}, **g})
        print(core['label'], spec['source'], t, 'PASS' if ok else 'FAIL', flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        keys = sorted({k for r in rows for k in r})
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    n_fail = sum(1 for r in rows if not r['pass'])
    print(f'{len(todo)} models, {n_fail} split-level deviations above tolerance; written to {a.out}')
    return 1 if n_fail else 0


if __name__ == '__main__':
    raise SystemExit(main())
