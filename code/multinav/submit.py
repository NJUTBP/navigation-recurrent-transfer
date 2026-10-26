"""Render reproducible LSF jobs; submit one source job per pair and an ended()-dependent report job."""
from __future__ import annotations
import argparse, fcntl, os, re, shlex, shutil, subprocess, sys
from pathlib import Path
from .common import *
from .plan import make_plan

q=shlex.quote

def script(root, plan, *, source_seed=None, dependency=None):
    cfg=plan['config']; lsf=cfg['lsf']; run=root/plan['run_root']; aggregate=source_seed is None
    title=('mn_report' if aggregate else f'mn_nav_{source_seed}')+'_'+cfg['run_id']
    logfile=run/'HPC_logs'/('report' if aggregate else f'nav_{source_seed}')
    queue=lsf.get('aggregate_queue') or lsf['queue'] if aggregate else lsf['queue']
    gpu=lsf.get('aggregate_gpu') if aggregate else lsf.get('gpu')
    for value in [queue,title,str(logfile),lsf.get('hosts'),lsf.get('walltime'),gpu]:
        if value is not None and ('\n' in str(value) or '\r' in str(value)): raise ValueError('Newline in LSF field')
    lines=['#!/bin/bash',f'#BSUB -q {queue}',f'#BSUB -n {int(lsf["cores"])}',f'#BSUB -J {q(title)}',
           f'#BSUB -o {q(str(logfile)+"_%J.out")}',f'#BSUB -e {q(str(logfile)+"_%J.err")}',
           '#BSUB -R "span[hosts=1]"']
    if gpu: lines.append(f'#BSUB -gpu "{gpu}"')
    if lsf.get('hosts'): lines.append(f'#BSUB -m "{lsf["hosts"]}"')
    if lsf.get('walltime'): lines.append(f'#BSUB -W {lsf["walltime"]}')
    if dependency: lines.append(f'#BSUB -w "{dependency}"')
    py=lsf['python']; boot=root/plan['code_root']/'multinav/bootstrap.py'; planfile=run/'plan.json'
    lines+=['set -euo pipefail',f'PROJECT_DIR={q(str(root))}', 'cd "$PROJECT_DIR"',
            'module load cuda || true',f'PYTHON={q(py)}','test -x "$PYTHON"',
            'export MPLBACKEND=Agg',f'export OMP_NUM_THREADS={int(cfg["threads"])}',
            f'export MKL_NUM_THREADS={int(cfg["threads"])}',f'export OPENBLAS_NUM_THREADS={int(cfg["threads"])}',
            'export PYTHONUNBUFFERED=1',f'export PYTHONHASHSEED={source_seed if source_seed is not None else cfg["statistics_seed"]}',
            'export CUBLAS_WORKSPACE_CONFIG=:4096:8']
    if not aggregate:
        envfile=run/'sources'/f'nav_seed_{source_seed}'/'environment.json'
        lines.append('"$PYTHON" '+q(str(boot))+' multinav.preflight --project "$PROJECT_DIR" --config '+q(str(run/'config_frozen.yaml'))+' --gpu --output '+q(str(envfile)))
        command=['multinav.run_source','--project',str(root),'--plan',str(planfile),'--source-seed',str(source_seed)]
    else: command=['multinav.aggregate','--project',str(root),'--plan',str(planfile),'--strict']
    lines.append('exec "$PYTHON" -u '+q(str(boot))+' '+' '.join(map(q,command)))
    return '\n'.join(lines)+'\n'


def submit_job(path):
    with path.open('r') as handle:
        result=subprocess.run(['bsub'],stdin=handle,text=True,capture_output=True,check=True)
    match=re.search(r'Job\s*<(\d+)>',result.stdout)
    if not match: raise RuntimeError('Could not parse bsub job ID. Do not blindly re-submit; inspect bjobs. '+result.stdout+result.stderr)
    return dict(job_id=int(match.group(1)),bsub_stdout=result.stdout,bsub_stderr=result.stderr,submitted_utc=now())


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project'); ap.add_argument('--config',default='multinav/config.yaml'); ap.add_argument('--run-id')
    ap.add_argument('--smoke',action='store_true'); ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--report-only',action='store_true'); ap.add_argument('--retry-sources',nargs='+',type=int)
    args=ap.parse_args(); root=project_root(args.project)
    planfile=make_plan(root,root/args.config,run_id=args.run_id,smoke=args.smoke); plan=read_json(planfile); run=planfile.parent
    directory=run/'lsf'; directory.mkdir(exist_ok=True); (run/'HPC_logs').mkdir(exist_ok=True)
    with (run/'.submission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledgerfile=run/'submission.json'; ledger=read_json(ledgerfile) if ledgerfile.exists() else dict(plan_sha256=plan['plan_sha256'],sources={},reports=[])
        seeds=[p['source_seed'] for p in plan['source_pairs']]
        selected=args.retry_sources or seeds
        if not set(selected)<=set(seeds): raise ValueError('Retry seed is outside the frozen plan')
        if args.report_only and args.retry_sources: raise ValueError('Choose report-only or retry-sources')
        if not args.dry_run and shutil.which('bsub') is None: raise RuntimeError('bsub not in PATH; use --dry-run outside the cluster')
        if not args.dry_run:
            executable=Path(plan['config']['lsf']['python'])
            if not executable.is_file() or not os.access(executable,os.X_OK):
                raise RuntimeError(f'Configured worker Python is not executable: {executable}')
            check=[str(executable),str(root/plan['code_root']/'multinav/bootstrap.py'),'multinav.preflight',
                   '--project',str(root),'--config',str(run/'config_frozen.yaml'),'--output',str(run/'submission_environment.json')]
            subprocess.run(check,cwd=root,check=True)
        ids=[]
        if not args.report_only:
            for seed in selected:
                path=directory/f'source_{seed}.lsf'; path.write_text(script(root,plan,source_seed=seed),encoding='utf-8')
                if args.dry_run: print('[render]',path); ids.append(f'SOURCE_JOB_{seed}'); continue
                prior=ledger['sources'].get(str(seed))
                if prior and not args.retry_sources:
                    print('[reuse submission]',seed,prior['job_id']); ids.append(str(prior['job_id'])); continue
                if prior:
                    # Avoid duplicate live workers. DONE/EXIT records may already have expired.
                    r=subprocess.run(['bjobs','-noheader','-o','stat',str(prior['job_id'])],text=True,capture_output=True)
                    states=r.stdout.split()
                    if any(s in {'PEND','RUN','PSUSP','USUSP','SSUSP','WAIT','PROV'} for s in states):
                        raise RuntimeError(f'Source {seed} still has a live LSF job {prior["job_id"]}')
                item=submit_job(path); ledger['sources'][str(seed)]=item; write_json(ledgerfile,ledger)
                ids.append(str(item['job_id'])); print('[submitted source]',seed,item['job_id'])
        # A retry may overlap other source workers. The report must also wait
        # for those live jobs, not only the explicitly retried sources.
        if args.retry_sources and not args.dry_run and not args.report_only:
            for key,prior in ledger['sources'].items():
                if int(key) in selected: continue
                status=subprocess.run(['bjobs','-noheader','-o','stat',str(prior['job_id'])],text=True,capture_output=True)
                if any(x in {'PEND','RUN','PSUSP','USUSP','SSUSP','WAIT','PROV'} for x in status.stdout.split()):
                    ids.append(str(prior['job_id']))
        dependency=' && '.join(f'ended({j})' for j in dict.fromkeys(ids)) or None
        report=directory/'aggregate_and_plot.lsf'; report.write_text(script(root,plan,dependency=dependency),encoding='utf-8')
        if args.dry_run:
            print('[render]',report); print('Dry-run dependency IDs are placeholders, not executable LSF job IDs.')
        else:
            item=submit_job(report); ledger['reports'].append(item); write_json(ledgerfile,ledger)
            print('[submitted aggregate + plots]',item['job_id'])
        print('Plan:',planfile); print('Figures:',run/'aggregate'/'figures'); print('Workload:',plan['workload'])

if __name__=='__main__': main()
