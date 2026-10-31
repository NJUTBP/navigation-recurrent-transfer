"""Build the source data of Supplementary Fig. S11 (rescaling interventions by path length and source).

Inputs are existing outputs of the formal experiments (no training):
  <mech>/E6_rho_scan/source_*/target_*/rho_0000|rho_0100/per_length.csv
  <mech>/E6_retention_intervention/source_*/target_*/arm_*/per_length.csv
  <mech>/E6_autonomous_retention/aggregate/all_retention.csv          (retention on reference states)
  <mech>/E6_retention_intervention/aggregate/all_condition_diagnostics.csv
  task-state retention table written by tools/reevaluation/task_state_retention.py
Output: <aggregate>/figure_source_data/SupplementaryFigureS12/
Statistical unit: source network (n = 10); target seeds are averaged within source first.

Usage:
  python export_figureS12_source_data.py --mech-root <PROJECT>/rerun_best_validation/followups/mechanism_v1 \
      --task-state-csv data/source_level_results/mechanism_v1/E6_task_state_retention.csv \
      --out data/aggregate_full/figure_source_data/SupplementaryFigureS12
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

C_FACTOR = 0.8032013898784912
SHORT_QUOTA = {1: 64, 2: 175, 3: 483, 4: 639, 5: 639}     # seen-short test quotas (Supplementary Methods S3)


def per_length(mech: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((mech / 'E6_rho_scan').glob('source_*/target_*/rho_*/per_length.csv')):
        rho = float(f.parent.name.split('_')[1]) / 1000
        if rho not in (0.0, 0.1):
            continue
        d = pd.read_csv(f); d['arm'] = 'rho0' if rho == 0 else 'rho0.1'; rows.append(d)
    for f in sorted((mech / 'E6_retention_intervention').glob('source_*/target_*/arm_*/per_length.csv')):
        rows.append(pd.read_csv(f))
    d = pd.concat(rows, ignore_index=True)[['source_seed', 'target_seed', 'arm', 'split', 'path_length', 'accuracy']]
    seen = d[((d.split == 'id_seen_short') & d.path_length.between(1, 5)) | ((d.split == 'seen_long') & d.path_length.between(6, 12))]
    n = seen.groupby('arm')[['source_seed', 'target_seed']].apply(lambda q: len(q.drop_duplicates()))
    assert (n == 30).all(), n.to_dict()
    return d, seen.groupby(['arm', 'source_seed', 'path_length'], as_index=False).accuracy.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--mech-root', type=Path, required=True)
    ap.add_argument('--task-state-csv', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    raw, prof = per_length(a.mech_root)
    c1 = pd.read_csv(a.mech_root / 'E6_autonomous_retention' / 'aggregate' / 'all_retention.csv')
    base = c1[c1.rho.isin([0.0, 0.1])].groupby(['rho', 'source_seed'], as_index=False).autonomous_retention_mean.mean()
    base['arm'] = np.where(np.isclose(base.rho, 0.0), 'rho0', 'rho0.1')
    diag = pd.read_csv(a.mech_root / 'E6_retention_intervention' / 'aggregate' / 'all_condition_diagnostics.csv')
    after = diag.groupby(['arm', 'source_seed'], as_index=False).source_retention_after_exact.mean()
    ref = pd.concat([base.rename(columns={'autonomous_retention_mean': 'retention'})[['arm', 'source_seed', 'retention']],
                     after.rename(columns={'source_retention_after_exact': 'retention'})[['arm', 'source_seed', 'retention']]])
    k0 = base[base.arm == 'rho0'].set_index('source_seed').autonomous_retention_mean
    k1 = base[base.arm == 'rho0.1'].set_index('source_seed').autonomous_retention_mean
    assert np.allclose(after[after.arm == 'nav_impair'].set_index('source_seed').source_retention_after_exact, C_FACTOR * k0, atol=2e-4)
    assert np.allclose(after[after.arm == 'rho01_rescue'].set_index('source_seed').source_retention_after_exact, k1 / C_FACTOR, atol=2e-4)
    ts = pd.read_csv(a.task_state_csv)
    task = ts.groupby(['arm', 'source_seed'], as_index=False).kappa_first_mean.mean().rename(columns={'kappa_first_mean': 'retention'})
    # summary statistics used in the text (source-level means of within-source differences)
    def pooled(g, split):
        if split == 'id_seen_short':
            w = g.path_length.map(SHORT_QUOTA).to_numpy(float); return float(np.sum(w * g.accuracy) / w.sum())
        return float(g.accuracy.mean())
    acc = (raw.groupby(['arm', 'split', 'source_seed', 'target_seed']).apply(lambda g: pooled(g, g.name[1]), include_groups=False)
           .groupby(['arm', 'split', 'source_seed']).mean())
    stats = {}
    for arm, base_arm in (('nav_impair', 'rho0'), ('rho01_rescue', 'rho0.1')):
        for split, key in (('id_seen_short', 'delta_alpha_SS'), ('seen_long', 'delta_alpha_SL')):
            d = acc.loc[arm, split] - acc.loc[base_arm, split]
            stats.setdefault(arm, {})[key] = round(float(d.mean()), 4)
    for name, df in (('retention_reference_states', ref), ('retention_task_states', task)):
        for arm, g in df.groupby('arm'):
            stats.setdefault(arm, {})[name + '_range'] = [round(float(g.retention.min()), 4), round(float(g.retention.max()), 4)]
    a.out.mkdir(parents=True, exist_ok=True)
    prof.to_csv(a.out / 'panelA_rescaling_profile.csv', index=False)
    ref.sort_values(['arm', 'source_seed']).to_csv(a.out / 'panelB_retention_reference_states.csv', index=False)
    task.sort_values(['arm', 'source_seed']).to_csv(a.out / 'panelC_retention_task_states.csv', index=False)
    (a.out / 's12_statistics.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')
    (a.out / 'PREPARE_COMPLETE.json').write_text(json.dumps({'status': 'complete', 'figure': 'SupplementaryFigureS12',
        'c': C_FACTOR, 'n_sources': int(prof.source_seed.nunique())}, indent=1), encoding='utf-8')
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
