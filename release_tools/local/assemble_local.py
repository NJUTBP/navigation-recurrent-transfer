#!/usr/bin/env python3
"""Copy the Level-1 figure source data from the local working figure folder into the repository.

Only the folders and files that the current plotting builders read are copied (see FIGURE_DATA_MAP.md).
The source folder is read only. Large files are reported so that they can be moved to the data archive.

  python release_tools/local/assemble_local.py --fig-all "E:\\FIG_ALL" --repo .
  python release_tools/local/assemble_local.py --fig-all "E:\\FIG_ALL" --repo . --render
"""
import argparse, shutil, subprocess, sys
from pathlib import Path

SOURCE_DATA_DIRS = ['Figure2', 'Figure3', 'Figure4', 'Figure5', 'Figure7', 'FigureA1', 'FigureA2', 'FigureA3', 'FigureA4',
                    'FigureA5', 'FigureA6', 'SupplementaryFigureS8', 'SupplementaryFigureS9', 'SupplementaryFigureS10',
                    'SupplementaryFigureS11']
SOURCE_DATA_FILES = ['SOURCE_DATA_MANIFEST.json', 'MECHANISM_FIGURE_SOURCE_FREEZE.json']
PHASEF_FILES = ['all_dataset_audit.csv', 'all_per_length.csv', 'all_update_curves.csv', 'condition_performance_matrix.csv',
                'primary_generalization_contrasts.csv', 'secondary_id_contrasts.csv', 'source_level_baselines.csv',
                'source_level_condition_diagnostics.csv', 'source_level_metrics.csv', 'source_level_relation_interventions.csv']
TOP_FILES = ['source_qualification.csv']
GITHUB_FILE_LIMIT_MB = 50


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fig-all', type=Path, required=True, help='folder that contains aggregate_full (e.g. E:\\FIG_ALL)')
    ap.add_argument('--repo', type=Path, default=Path('.'))
    ap.add_argument('--render', action='store_true', help='redraw all figures afterwards')
    a = ap.parse_args()
    src = a.fig_all / 'aggregate_full'
    dst = a.repo / 'data' / 'aggregate_full'
    missing, copied, large = [], 0, []
    for d in SOURCE_DATA_DIRS:
        s = src / 'figure_source_data' / d
        if not s.is_dir():
            missing.append(str(s)); continue
        for p in s.rglob('*'):
            if p.is_file():
                q = dst / 'figure_source_data' / d / p.relative_to(s)
                q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q); copied += 1
    for name in SOURCE_DATA_FILES:
        p = src / 'figure_source_data' / name
        if p.is_file():
            (dst / 'figure_source_data').mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst / 'figure_source_data' / name); copied += 1
    for name in PHASEF_FILES:
        p = src / 'hard8x8_phaseF_followup' / name
        if p.is_file():
            (dst / 'hard8x8_phaseF_followup').mkdir(parents=True, exist_ok=True); shutil.copy2(p, dst / 'hard8x8_phaseF_followup' / name); copied += 1
        else:
            missing.append(str(p))
    for name in TOP_FILES:
        p = src / name
        if p.is_file():
            shutil.copy2(p, dst / name); copied += 1
    total = 0
    for p in dst.rglob('*'):
        if p.is_file():
            total += p.stat().st_size
            if p.stat().st_size > GITHUB_FILE_LIMIT_MB * 1e6:
                large.append((p, p.stat().st_size))
    print(f'copied {copied} files; data/aggregate_full now {total / 1e6:.1f} MB')
    for p, n in large:
        print(f'  LARGE (> {GITHUB_FILE_LIMIT_MB} MB, keep in the data archive, not in git): {p} {n / 1e6:.1f} MB')
    if missing:
        print('MISSING (check names in your aggregate_full):', *missing, sep='\n  ')
    if a.render:
        cmd = [sys.executable, str(a.repo / 'figures' / 'redraw_all_figures.py'), '--aggregate', str(dst),
               '--out', str(a.repo / 'outputs' / 'figures')]
        print(' '.join(cmd)); raise SystemExit(subprocess.call(cmd))


if __name__ == '__main__':
    main()
