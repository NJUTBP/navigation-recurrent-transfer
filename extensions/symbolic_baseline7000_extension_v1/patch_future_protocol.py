#!/usr/bin/env python3
"""Patch a DEVELOPMENT source tree so future full runs train simple symbolic baselines at every episode budget, including 7000.

Do not run this against the frozen code directory of the already-completed current cohort.
"""
from pathlib import Path
import argparse, shutil, datetime, yaml

def main():
    ap=argparse.ArgumentParser();ap.add_argument('source_tree');args=ap.parse_args();root=Path(args.source_tree).resolve()
    files=[root/'gpm_cmap/configs/symbolic_adapter.yaml',root/'gpm_cmap/configs/symbolic_adapter_ctl01.yaml']
    for p in files:
        if not p.is_file():raise SystemExit('Missing '+str(p))
    stamp=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for p in files:
        shutil.copy2(p,p.with_name(p.name+'.pre_baseline7000_'+stamp))
        cfg=yaml.safe_load(p.read_text());cfg.setdefault('controls',{})['baseline_budgets']='all'
        p.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        print('[patched]',p,'controls.baseline_budgets=all')
    plan=root/'multinav/plan.py'
    if plan.is_file():
        text=plan.read_text(encoding='utf-8')
        text=text.replace('"Exp8_38_arm_budget_cells": 38*m','"Exp8_40_arm_budget_cells": 40*m')
        plan.write_text(text,encoding='utf-8');print('[patched workload label]',plan)
if __name__=='__main__':main()
