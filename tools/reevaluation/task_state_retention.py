#!/usr/bin/env python3
"""One-step retention on task states (Supplementary Fig. S12c) for the E6 rho = 0, 0.10 cores and the two
rescaling arms. Rebuilds each frozen-core target model from the navigation checkpoint and recorded seeds,
checks its stored test accuracy (Gate R), rolls out 256 seen-composition test trials of length 12, samples
128 visited states and computes kappa(h) = ||ReLU(W_rec h)|| / ||h|| (Eq. 12). GPU recommended.

  NRT_PROJECT_ROOT=/path/to/project python tools/reevaluation/task_state_retention.py --out outputs/task_state_retention.csv
"""
import argparse, csv, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reevaluation_common as nc  # noqa: E402
import dyn_math as dm    # noqa: E402

SPECS = [('rho0', {'kind': 'e6', 'rho': 0.0}), ('rho0.1', {'kind': 'e6', 'rho': 0.1}),
         ('nav_impair', {'kind': 'e6i', 'arm': 'nav_impair'}), ('rho01_rescue', {'kind': 'e6i', 'arm': 'rho01_rescue'})]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--sources', type=int, nargs='*', default=list(nc.SOURCES))
    ap.add_argument('--targets', type=int, nargs='*', default=list(nc.TARGETS_32K))
    a = ap.parse_args()
    nc.require_released_packages()
    torch = nc.import_torch()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fullscale, phasef = nc.load_phasef_modules()
    hist = fullscale.import_historical()
    protocol = nc.derive_protocol(updates=8000, targets=nc.TARGETS, schema_version='reevaluation_v1', status='reevaluation')
    resolver = nc.ModelResolver(hist=hist, fullscale=fullscale, phasef=phasef, model_cfg=protocol['model'])
    rows = []
    for target in a.targets:
        pools, _, _ = nc.build_split(fullscale, hist, protocol, target)
        long12 = [e for e in pools['seen_long'] if int(e.velocities.shape[0]) == 12][:256]
        for source in a.sources:
            seed = nc.stable_seed32('nd2_b6_tangent', source, target)
            for arm, spec in SPECS:
                core = resolver.resolve({**spec, 'source': source, 'target': target})
                model, _, _ = nc.rebuild_frozen_target(core['checkpoint'], hist=hist, phasef=phasef, device=device,
                                                       core_hh=core['hh'].to(torch.float32), core_ih=core['ih'].to(torch.float32),
                                                       model_cfg=protocol['model'])
                seeds = nc.phasef_seeds(fullscale, protocol, source, target)
                summaries = {s: nc.evaluate_episodes(model, pools[s], hist=hist, fullscale=fullscale, device=device,
                                                     loader_seed=seeds['loader_seed'], split_name=s)[0] for s in nc.TEST_SPLITS}
                gate, ok = nc.gate_r(summaries, nc.reference_rows(core))
                if not ok:
                    raise RuntimeError(f'Reconstructed model does not reproduce stored accuracies: {core["label"]}: {gate}')
                w_hh = model.grid_rnn.cell.h_proj.weight.detach(); w_in = model.grid_rnn.cell.in_proj.weight.detach()
                gain = float(model.gain().detach().cpu())
                batch = nc.one_batch(long12, hist=hist, fullscale=fullscale)
                with torch.no_grad():
                    out = model(start_node=batch['start_node'].to(device), velocities=batch['velocities'].to(device),
                                lengths=batch['lengths'].to(device), start_coords=batch['start_coords'].to(device),
                                validate_inputs=False)
                    drives = (batch['velocities'].to(device) * gain) @ w_in.t()
                    states, _ = dm.rollout(out['h0'], w_hh, drives)
                    visited = states.reshape(-1, states.shape[-1])
                    g = torch.Generator().manual_seed(int(seed))
                    pick = torch.randperm(visited.shape[0], generator=g)[:128].to(device)
                    u = visited[pick]; u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-20)
                    kappa = torch.relu(u @ w_hh.t()).norm(dim=-1)
                rows.append({'source_seed': source, 'target_seed': target, 'arm': arm, 'kappa_first_mean': float(kappa.mean()),
                             'gate_r_pass': ok, 'learned_gain': gain})
                print(rows[-1], flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
