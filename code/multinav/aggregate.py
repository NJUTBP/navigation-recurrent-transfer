"""Audit, source-level inference, and eight current-manuscript figure sets."""
from __future__ import annotations
import argparse, fcntl, os, sys, traceback
from pathlib import Path
import pandas as pd
from scipy.stats import binomtest
from .common import *
from .collect import collect, source_summary, infer, mechanism_replication
from .plot import Figures


def run(root,planfile,strict=False):
    plan=read_json(planfile); directory=root/plan['run_root']; verify_tree(root/plan['code_root'],plan['code_manifest'])
    expected=plan['plan_sha256']; payload={k:v for k,v in plan.items() if k!='plan_sha256'}
    if digest(payload)!=expected: raise RuntimeError('Frozen plan hash mismatch')
    with (directory/'.aggregate.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempts=directory/'aggregate_attempts'; attempts.mkdir(exist_ok=True)
        n=1+max([int(p.name.split('_')[-1]) for p in attempts.glob('try_*')]+[0]); out=attempts/f'try_{n:03d}';out.mkdir()
        audit=dict(plan_sha256=expected,started_utc=now(),status='running',profile=plan['config']['profile'])
        write_json(out/'audit.json',audit)
        try:
            frame,sources,stages,gates,files,errors=collect(root,plan)
            summaries,source_rows=source_summary(frame,plan)
            contrasts,details,pairs=infer(frame,plan,sources)
            frame.to_csv(out/'target_level_metrics.csv',index=False)
            summaries.to_csv(out/'source_level_summary.csv',index=False);source_rows.to_csv(out/'source_means.csv',index=False)
            contrasts.to_csv(out/'source_level_contrasts.csv',index=False);details.to_csv(out/'source_paired_differences.csv',index=False)
            pairs.to_csv(out/'target_paired_differences.csv',index=False)
            pd.DataFrame(stages).to_csv(out/'stage_coverage.csv',index=False)
            pd.DataFrame(sources).to_csv(out/'source_qualification.csv',index=False);write_json(out/'source_qualification.json',sources)
            pd.DataFrame(gates).to_csv(out/'source_gate_components.csv',index=False)
            write_json(out/'input_artifact_checksums.json',files)
            denominator=len(plan['source_pairs']); pass_rows=[]
            for field in ['ctl_qualified','is_torus_like','source_exact_gate','source_fibre_gate']:
                count=sum(z.get(field) is True for z in sources)
                n_evaluable=sum(z.get(field) is not None for z in sources)
                interval=binomtest(count,denominator).proportion_ci(confidence_level=.95,method='exact')
                pass_rows.append(dict(criterion=field,n_planned=denominator,n_evaluable=n_evaluable,n_confirmed_pass=count,
                    confirmed_pass_fraction=count/denominator,binomial_low=interval.low,binomial_high=interval.high,
                    interpretation='Confirmed pass out of all planned sources; technical missingness is separately flagged, not called scientific failure.'))
            pd.DataFrame(pass_rows).to_csv(out/'source_gate_pass_counts.csv',index=False)
            mechanism_rows,mechanism_counts=mechanism_replication(frame,plan,sources)
            mechanism_rows.to_csv(out/'mechanism_replication_by_source.csv',index=False)
            mechanism_counts.to_csv(out/'mechanism_replication_counts.csv',index=False)
            primary=contrasts[contrasts.scope=='all_planned_sources']
            if not bool(primary.complete.all()): errors.append('One or more prespecified behavioural contrast families lack all planned source/target pairs')
            Figures(root,plan,out/'figures',source_rows,contrasts,details,sources).run()
            audit.update(status='complete' if not errors else 'incomplete',ended_utc=now(),errors=errors,
                         n_sources_planned=denominator,n_source_identities=sum(bool(s.get('core_sha256')) for s in sources),
                         n_target_metric_rows=len(frame),n_source_metric_rows=len(source_rows),
                         strict_passed=not errors,scientific_nonpasses=[s for s in stages if s['status']=='not_evaluable'])
            write_json(out/'audit.json',audit)
            report=['# 多导航源运行报告（不改动正文）','',f'- run_id: `{plan["config"]["run_id"]}`',f'- profile: `{plan["config"]["profile"]}`',
                    f'- 计划独立导航源：{denominator}',f'- 汇总完整性：**{audit["status"]}**','',
                    '## 统计口径','先在每个导航源内配对相同目标种子的实验臂，再对目标重复取平均；每个导航源仅贡献一个差值。',
                    '主区间为源均值的 95% t 区间（自由度 K-1）；另列整源区块 bootstrap、整源符号翻转、符号检验、逐源剔除结果。',
                    'Holm 按冻结的比较族校正；缺失比较不会缩小比较族。不完整队列不输出确认性 p 值。',
                    '相同源内的目标模型、轨迹、时间点、神经元、shuffle 和 PCA 维度均不增加独立源数。','',
                    '## 输出','`source_level_contrasts.csv` 为源级推断；`source_paired_differences.csv` 保留逐源配对差值。',
                    '`source_gate_pass_counts.csv` 同时保留计划数、可评估数、确认通过数；连续机制量只解释为可评估源条件下的结果。',
                    '`figures/Figure1`–`Figure4` 与 `FigureA1`–`FigureA4` 含各面板 SVG/PDF/PNG/CSV；同名总图为拼版预览。','',
                    '## 限制','这是一项新源队列，不复用旧图中的固定数值、固定神经元编号或旧 50 epoch 训练记录。',
                    '默认新源均完整训练 150 epoch；原代码的 8 个因子比较被显式冻结，正文所写 6 个的差异保留在计划中。',
                    '统计把源种子视为训练程序产生的独立重复，不代表跨任务、跨架构或跨数据分布的普遍性。','']
            if errors: report+=['## 需要处理的技术缺失',*['- '+e for e in errors]]
            if plan['config']['profile']=='smoke': report+=['','**SMOKE：所有数值只用于验证代码，不得作科学结果使用。**']
            (out/'REPORT_zh.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
            public=directory/'aggregate';tmp=directory/'aggregate.new'
            if tmp.is_symlink():tmp.unlink()
            tmp.symlink_to(os.path.relpath(out,directory));os.replace(tmp,public)
            print('Aggregate:',public);print('Status:',audit['status'])
            return 2 if strict and errors else 0
        except Exception:
            audit.update(status='failed',ended_utc=now(),traceback=traceback.format_exc());write_json(out/'audit.json',audit)
            raise


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project');ap.add_argument('--plan',required=True);ap.add_argument('--strict',action='store_true')
    args=ap.parse_args();root=project_root(args.project);os.chdir(root)
    raise SystemExit(run(root,root/args.plan,strict=args.strict))

if __name__=='__main__':main()
