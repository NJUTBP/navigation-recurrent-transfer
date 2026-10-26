#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest

SCRIPT_VERSION = 'best_validation_unified_aggregate_v2_schoyen_figureA3'
ACTIVE_BEHAVIOUR_PREFIXES = ['exp5_nav', 'exp5_ctl', 'mapping', 'test_transforms', 'adapter', 'symbolic']
RETIRED_COMPONENTS = [
    ('legacy_topology_export', 'Replaced by Schøyen torus_topology on validation best.pt.'),
    ('legacy_formal_topology', 'Replaced by selected-candidate Euclidean Schøyen persistent homology.'),
    ('legacy_cohomological_phase_anchor', 'Replaced by an independent TRUE-position Fourier chart on the Schøyen-selected ensemble.'),
]
COLUMNS = ['source_seed', 'target_seed', 'experiment', 'metric', 'arm', 'budget', 'transform', 'split', 'x', 'value', 'artifact']


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def relative(path: Path, base: Path) -> str:
    try:
        return str(path.resolve().relative_to(base.resolve()))
    except Exception:
        return str(path.resolve())


def load_yaml(path: Path) -> dict:
    obj = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if not isinstance(obj, dict):
        raise ValueError(f'Expected YAML mapping: {path}')
    return obj


def normalize_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == 'true':
            return True
        if value.lower() == 'false':
            return False
    return bool(value)


def build_analysis_plan(run_root: Path, code_root: Path, original_plan: dict) -> dict:
    plan = copy.deepcopy(original_plan)
    cfg = plan.setdefault('config', {})
    cfg['run_id'] = str(cfg.get('run_id', run_root.name)) + '_best_validation'
    cfg['checkpoint_policy'] = 'best_validation'
    cfg['profile'] = cfg.get('profile', 'formal')

    current_cfg_path = code_root / 'multinav' / 'config.yaml'
    if current_cfg_path.is_file():
        current_cfg = load_yaml(current_cfg_path)
        if isinstance(current_cfg.get('schoyen_topology'), dict):
            cfg['schoyen_topology'] = current_cfg['schoyen_topology']
        # Preserve frozen statistical design, but make the current source rule
        # explicit in the analysis provenance.
        cfg['checkpoint_policy'] = 'best_validation'

    # multinav.collect.infer uses this path to reopen the frozen experiment
    # configs. Absolute paths are intentional and safe with pathlib joining.
    plan['code_root'] = str(code_root.resolve())
    return plan


class HybridCollector:
    def __init__(self, run_root: Path, result_root: Path, code_root: Path, plan: dict):
        self.run_root = run_root.resolve()
        self.result_root = result_root.resolve()
        self.code_root = code_root.resolve()
        self.plan = plan
        self.rows: List[dict] = []
        self.stages: List[dict] = []
        self.sources: List[dict] = []
        self.files: Dict[str, str] = {}
        self.errors: List[str] = []

    def tracked_json(self, path: Path) -> Any:
        self.files[relative(path, self.run_root)] = sha256(path)
        return read_json(path)

    def tracked_csv(self, path: Path) -> List[dict]:
        self.files[relative(path, self.run_root)] = sha256(path)
        return pd.read_csv(path).to_dict('records')

    def add(self, s, t, experiment, metric, arm, value, artifact, **kwargs):
        if value is None:
            return
        try:
            value = float(value)
        except (ValueError, TypeError):
            return
        if not np.isfinite(value):
            return
        row = dict(
            source_seed=int(s), target_seed=int(t), experiment=experiment, metric=metric,
            arm=str(arm), budget=-1, transform='', split='', x=-1, value=value,
            artifact=relative(Path(artifact), self.run_root),
        )
        row.update(kwargs)
        self.rows.append(row)

    def _stage_status(self, base: Path, name: str, *, topology: bool = False) -> Tuple[str, str]:
        p = base / 'state' / f'{name}.json'
        if not p.is_file():
            return 'missing', ''
        z = self.tracked_json(p)
        raw = str(z.get('status', 'missing'))
        if topology and raw == 'success':
            return 'done', ''
        return raw, str(z.get('reason', z.get('error', '')) or '')

    def _new_committed(self, base: Path, name: str, filename: str, *, topology: bool = False) -> Optional[Path]:
        status, _ = self._stage_status(base, name, topology=topology)
        if status != 'done':
            return None
        p = base / 'artifacts' / name / filename
        if not p.is_file():
            self.errors.append(f'Missing committed rerun artifact: {p}')
            return None
        self.files[relative(p, self.run_root)] = sha256(p)
        return p

    def _old_artifact(self, base: Path, stage: str, filename: str) -> Optional[Path]:
        state = base / 'state' / f'{stage}.json'
        p = base / 'artifacts' / stage / filename
        if state.is_file():
            z = self.tracked_json(state)
            if z.get('status') != 'done':
                self.errors.append(f'Original reused stage is not done: source={base.name} stage={stage} status={z.get("status")}')
                return None
        if not p.is_file():
            self.errors.append(f'Missing original reused artifact: {p}')
            return None
        self.files[relative(p, self.run_root)] = sha256(p)
        return p

    def _record_stage_coverage(self, pair: dict, old_base: Path, new_base: Path):
        s = int(pair['source_seed'])
        for name in ['navigation', 'sequence_control']:
            status, reason = self._stage_status(old_base, name)
            normalized = 'reused_done' if status == 'done' else status
            self.stages.append(dict(source_seed=s, stage=name, status=normalized, provenance='original_frozen_run', reason=reason))
            if status != 'done':
                self.errors.append(f'Source {s}: reused {name} = {status}')

        for name in ['torus_topology', 'formal_topology_schoyen', 'source_splits_schoyen', 'source_reduced_schoyen']:
            status, reason = self._stage_status(new_base, name, topology=(name == 'torus_topology'))
            self.stages.append(dict(source_seed=s, stage=name, status=status, provenance='best_validation_rerun', reason=reason))
            if name in {'torus_topology', 'formal_topology_schoyen'} and status != 'done':
                self.errors.append(f'Source {s}: {name} = {status}')
            elif name in {'source_splits_schoyen', 'source_reduced_schoyen'} and status not in {'done', 'not_evaluable'}:
                self.errors.append(f'Source {s}: {name} = {status}')

        for target in pair['main_target_seeds']:
            for prefix in ACTIVE_BEHAVIOUR_PREFIXES:
                name = f'{prefix}_t{target}'
                status, reason = self._stage_status(new_base, name)
                self.stages.append(dict(source_seed=s, stage=name, status=status, provenance='best_validation_rerun', reason=reason))
                if status != 'done':
                    self.errors.append(f'Source {s}: {name} = {status}')

        for target in pair['hard_target_seeds']:
            name = f'hard_t{target}'
            status, reason = self._stage_status(new_base, name)
            self.stages.append(dict(source_seed=s, stage=name, status=status, provenance='best_validation_rerun', reason=reason))
            if status != 'done':
                self.errors.append(f'Source {s}: {name} = {status}')

        for target in pair['factorial_target_seeds']:
            fact = f'factorial_t{target}'
            status, reason = self._stage_status(new_base, fact)
            self.stages.append(dict(source_seed=s, stage=fact, status=status, provenance='best_validation_rerun', reason=reason))
            if status != 'done':
                self.errors.append(f'Source {s}: {fact} = {status}')
            for prefix in ['action_schoyen', 'path_schoyen']:
                name = f'{prefix}_t{target}'
                status, reason = self._stage_status(new_base, name)
                self.stages.append(dict(source_seed=s, stage=name, status=status, provenance='best_validation_rerun', reason=reason))
                if status not in {'done', 'not_evaluable'}:
                    self.errors.append(f'Source {s}: {name} = {status}')

    def collect(self):
        original_plan_sha = self.plan.get('plan_sha256')
        seen_nav: Dict[str, int] = {}
        seen_ctl: Dict[str, int] = {}

        for pair in self.plan['source_pairs']:
            s = int(pair['source_seed'])
            old_base = self.run_root / 'sources' / f'nav_seed_{s}'
            new_base = self.result_root / 'sources' / f'nav_seed_{s}'
            if not old_base.is_dir():
                self.errors.append(f'Source {s}: missing original source directory {old_base}')
            if not new_base.is_dir():
                self.errors.append(f'Source {s}: missing rerun source directory {new_base}')
                continue

            identity_path = new_base / 'source_identity.json'
            identity = self.tracked_json(identity_path) if identity_path.is_file() else {}
            if not identity:
                self.errors.append(f'Source {s}: missing rerun source_identity.json')
            else:
                if identity.get('checkpoint_policy') != 'best_validation':
                    self.errors.append(f'Source {s}: rerun identity checkpoint_policy={identity.get("checkpoint_policy")!r}')
                if identity.get('original_plan_sha256') and original_plan_sha and identity.get('original_plan_sha256') != original_plan_sha:
                    self.errors.append(f'Source {s}: rerun identity points to another frozen plan')
                cp = Path(identity.get('navigation_checkpoint', ''))
                if cp.is_file() and identity.get('navigation_checkpoint_sha256'):
                    if sha256(cp) != identity['navigation_checkpoint_sha256']:
                        raise RuntimeError(f'Source {s}: navigation best.pt checksum changed')

            ctlq_path = old_base / 'control_qualification.json'
            ctlq = self.tracked_json(ctlq_path) if ctlq_path.is_file() else {}
            topq_path = new_base / 'topology_qualification.json'
            topq = self.tracked_json(topq_path) if topq_path.is_file() else {}
            preflight_path = new_base / 'preflight.json'
            preflight = self.tracked_json(preflight_path) if preflight_path.is_file() else {}

            source = dict(identity)
            source.update(
                source_seed=s,
                control_seed=int(pair['control_seed']),
                ctl_qualified=(ctlq.get('qualification') or {}).get('passed'),
                ctl_training_completed=ctlq.get('training_completed', False),
                topology_analysis_completed=topq.get('analysis_completed', False),
                is_torus_like=topq.get('is_torus_like'),
                topology_selected_clusters=topq.get('selected_clusters'),
                topology_selected_units=topq.get('n_selected_units'),
                topology_robust_min_gap=topq.get('robust_min_gap_score'),
                topology_checkpoint_role=topq.get('checkpoint_role'),
                schoyen_formal_topology_passed=None,
                source_reduced_schoyen_evaluable=False,
                source_classification=None,
                source_exact_gate=None,
                source_fibre_gate=None,
                schoyen_phase_chart_test_rmse_rad=None,
                source_validation_task_loss=preflight.get('navigation_validation_task_loss'),
            )

            self._record_stage_coverage(pair, old_base, new_base)
            active_rows = [r for r in self.stages if r['source_seed'] == s and r['status'] not in {'retired_not_evaluable','not_evaluable'}]
            source['worker_status'] = 'finished' if active_rows and all(r['status'] in {'done', 'reused_done'} for r in active_rows) else 'incomplete'

            # Reused original navigation source diagnostics.
            p = self._old_artifact(old_base, 'navigation', 'history.json')
            if p:
                for row in self.tracked_json(p):
                    for metric, key in [('validation_loss', 'validation_task_loss'), ('grid_fraction', 'frac_grid_above_03')]:
                        self.add(s, -1, 'navigation', metric, 'navigation', row.get(key), p, x=int(row['epoch']))

            # Reused original CTL diagnostics.
            p = self._old_artifact(old_base, 'sequence_control', 'training_history.json')
            if p:
                data = self.tracked_json(p)
                source['ctl_selected_epoch'] = data.get('best_epoch')
                gate = data.get('training_gate') or {}
                source['ctl_error_reduction'] = gate.get('relative_nmse_reduction')
                for row in data.get('history', []):
                    val = row.get('validation') or {}
                    primary = val.get('primary') or {}
                    self.add(s, -1, 'control', 'nmse', 'primary_delay', primary.get('normalized_mse'), p, x=row.get('epoch', -1))
                    self.add(s, -1, 'control', 'nmse', 'auxiliary_mean', val.get('auxiliary_normalized_mse_mean'), p, x=row.get('epoch', -1))
                for lag, z in (data.get('best_validation', {}).get('by_lag') or {}).items():
                    self.add(s, -1, 'control', 'selected_delay_nmse', 'control', z.get('normalized_mse'), p, x=int(lag))

            compat = self._old_artifact(old_base, 'sequence_control', 'checkpoint_compatibility.json')
            if compat:
                source['ctl_tensor_compatibility'] = self.tracked_json(compat)
            else:
                source['ctl_tensor_compatibility'] = {}

            ci_path = old_base / 'control_identity.json'
            if ci_path.is_file():
                ci = self.tracked_json(ci_path)
                source['ctl_core_sha256'] = ci.get('core_sha256')
            else:
                source['ctl_core_sha256'] = None
                self.errors.append(f'Source {s}: missing original control_identity.json')

            # Rerun main cognitive experiments.
            for target in pair['main_target_seeds']:
                target = int(target)
                p = self._new_committed(new_base, f'exp5_nav_t{target}', 'raw_per_seed.json')
                if p:
                    data = self.tracked_json(p)
                    for arm, entries in (data.get('main') or {}).items():
                        for z in entries:
                            self.add(s, target, 'main', 'accuracy', arm, z.get('test_acc'), p)
                            if arm == 'transfer_finetune':
                                self.add(s, target, 'mapping', 'accuracy', 'aligned', z.get('test_acc'), p)
                    for hp in sorted(p.parent.glob(f'seed_{target}/*/history.json')):
                        arm = hp.parent.name
                        history = self.tracked_json(hp)
                        for z in history:
                            self.add(s, target, 'main', 'validation_accuracy', arm, z.get('validation_acc'), hp, x=z.get('epoch', -1))

                p = self._new_committed(new_base, f'exp5_ctl_t{target}', 'raw_per_seed.json')
                if p:
                    data = self.tracked_json(p)
                    for z in (data.get('main') or {}).get('transfer_freeze', []):
                        self.add(s, target, 'main', 'accuracy', 'ctl01_frozen', z.get('test_acc'), p)

                p = self._new_committed(new_base, f'mapping_t{target}', 'raw_per_seed.json')
                if p:
                    data = self.tracked_json(p)
                    for arm, entries in (data.get('mapping_control') or {}).items():
                        for z in entries:
                            self.add(s, target, 'mapping', 'accuracy', arm, z.get('test_acc'), p)

                p = self._new_committed(new_base, f'test_transforms_t{target}', 'test_transform_rows.json')
                if p:
                    for z in self.tracked_json(p):
                        self.add(s, target, 'test_transforms', 'accuracy', z.get('arm'), z.get('test_accuracy'), p, transform=z.get('transform', ''))

                p = self._new_committed(new_base, f'adapter_t{target}', 'adapter_rescue_rows.json')
                if p:
                    for z in self.tracked_json(p):
                        self.add(s, target, 'adapter', 'accuracy', z.get('arm'), z.get('test_accuracy'), p, transform=z.get('transform', ''))

                p = self._new_committed(new_base, f'symbolic_t{target}', 'symbolic_adapter_rows.json')
                if p:
                    for z in self.tracked_json(p):
                        self.add(s, target, 'symbolic', 'accuracy', z.get('arm'), z.get('test_accuracy'), p, budget=int(z.get('budget', -1)))
                        self.add(s, target, 'symbolic', 'no_recurrence_accuracy', z.get('arm'), z.get('test_accuracy_no_recurrence'), p, budget=int(z.get('budget', -1)))

            for label_name, targets in [('hard', pair['hard_target_seeds']), ('factorial', pair['factorial_target_seeds'])]:
                for target in targets:
                    target = int(target)
                    p = self._new_committed(new_base, f'{label_name}_t{target}', f'{label_name}_learning_curves.csv')
                    if p:
                        for z in self.tracked_csv(p):
                            for split, key in [('id', 'id_test_accuracy'), ('long', 'long_test_accuracy'), ('composition', 'composition_test_accuracy')]:
                                arm = z.get('condition', z.get('cell'))
                                self.add(s, target, label_name, 'accuracy', arm, z.get(key), p, budget=int(z.get('unique_training_episodes', -1)), split=split)
                    if label_name == 'factorial':
                        p = self._new_committed(new_base, f'factorial_t{target}', 'factorial_per_seed.csv')
                        if p:
                            for z in self.tracked_csv(p):
                                for split, key in [('id', 'auc_id'), ('long', 'auc_long'), ('composition', 'auc_composition')]:
                                    self.add(s, target, 'factorial', 'auc', z.get('cell'), z.get(key), p, split=split)

            # New Schøyen topology.
            p = self._new_committed(new_base, 'torus_topology', 'torus_analysis_summary.json', topology=True)
            if p:
                z = self.tracked_json(p)
                ph = z.get('persistent_cohomology') or {}
                sel = z.get('selection') or {}
                source['topology_selected_clusters'] = sel.get('selected_clusters')
                source['topology_selected_units'] = sel.get('n_selected_units')
                source['topology_robust_min_gap'] = sel.get('robust_min_gap_score')
                metrics = ph.get('metrics') or {}
                for metric_name, metric_payload in metrics.items():
                    for homology, key in [('H1', 'H1_top10_lifetimes'), ('H2', 'H2_top10_lifetimes')]:
                        for rank, v in enumerate(metric_payload.get(key) or [], 1):
                            self.add(s, -1, 'topology', 'lifetime', metric_name + '_' + homology, v, p, x=rank)
                    self.add(s, -1, 'topology', 'gap', metric_name + '_H1_2_to_3', metric_payload.get('H1_second_to_third'), p)
                    self.add(s, -1, 'topology', 'gap', metric_name + '_H2_1_to_2', metric_payload.get('H2_first_to_second'), p)
                self.add(s, -1, 'topology', 'selection_score', 'robust_min_gap', sel.get('robust_min_gap_score'), p)
                self.add(s, -1, 'topology', 'selected_units', 'auto_ensemble', sel.get('n_selected_units'), p)
                cands = p.parent / 'candidate_topology.csv'
                if cands.is_file():
                    for row in self.tracked_csv(cands):
                        passed = normalize_bool(row.get('passes_robust_121_signature'))
                        self.add(
                            s, -1, 'topology', 'candidate_score', str(row.get('clusters')),
                            row.get('robust_min_gap_score'), cands,
                            x=int(row.get('n_units') or -1), split='pass' if passed is True else 'nonpass',
                        )

            # Rebuilt FigureA3: Schoyen Euclidean formal topology.
            p = self._new_committed(new_base, 'formal_topology_schoyen', 'formal_topology_schoyen.json')
            if p:
                z = self.tracked_json(p)
                source['schoyen_formal_topology_passed'] = normalize_bool(z.get('formal_topology_passed'))
                for criterion, check in (z.get('checks') or {}).items():
                    margin = check.get('log2_margin')
                    if margin is not None:
                        self.add(s, -1, 'formal_topology', 'log2_margin', criterion, margin, p, split='euclidean')
                    value = check.get('value')
                    if value is not None:
                        self.add(s, -1, 'formal_topology', 'raw_value', criterion, value, p, split='euclidean')

            # Schoyen-anchored reduced source model: mature Source-v3 dynamics
            # machinery with a new independent Fourier phase chart.
            p = self._new_committed(new_base, 'source_reduced_schoyen', 'source_reduced_schoyen_summary.json')
            if p:
                z = self.tracked_json(p)
                quality = z.get('quality_gates') or {}
                source.update(
                    source_reduced_schoyen_evaluable=True,
                    source_classification=quality.get('classification'),
                    source_exact_gate=normalize_bool((quality.get('exact_2d_gate') or {}).get('passed')),
                    source_fibre_gate=normalize_bool((quality.get('toroidal_base_plus_fibre_gate') or {}).get('passed')),
                )
                phase_test = ((z.get('phase_chart') or {}).get('test') or {})
                phase_val = ((z.get('phase_chart') or {}).get('validation') or {})
                source['schoyen_phase_chart_test_rmse_rad'] = phase_test.get('phase_rmse_rad')
                self.add(s, -1, 'source_gate', 'phase_chart_rmse_rad', 'validation', phase_val.get('phase_rmse_rad'), p, split='schoyen_phase_chart')
                self.add(s, -1, 'source_gate', 'phase_chart_rmse_rad', 'test', phase_test.get('phase_rmse_rad'), p, split='schoyen_phase_chart')
                for gate_name in ['exact_2d_gate', 'toroidal_base_plus_fibre_gate']:
                    gate = quality.get(gate_name) or {}
                    for name, check in (gate.get('checks') or {}).items():
                        v = check.get('value'); threshold = check.get('threshold'); op = check.get('operator')
                        if op in {'<=','>='} and v is not None and threshold not in {None, 0} and float(v) > 0 and float(threshold) > 0:
                            margin = math.log2(float(v)/float(threshold)) if op == '>=' else math.log2(float(threshold)/float(v))
                            self.add(s, -1, 'source_gate', 'log2_margin', name, margin, p, split=gate_name)

            # Mechanism analyses use the completed best-validation factorial checkpoints
            # and the Schoyen-anchored reduced source artifact.
            for target in pair['factorial_target_seeds']:
                target = int(target)
                p = self._new_committed(new_base, f'action_schoyen_t{target}', 'action_causality_per_seed.csv')
                if p:
                    for z in self.tracked_csv(p):
                        for metric in ['minimum_confirmatory_null_ratio', 'normalized_phase_law_rmse', 'controlled_recurrent_reuse_gate_passed']:
                            val = z.get(metric)
                            if isinstance(val, str) and val.lower() in {'true','false'}:
                                val = val.lower() == 'true'
                            self.add(s, target, 'action', metric, z.get('condition'), val, p, budget=int(z.get('unique_training_episodes', -1)))
                p = self._new_committed(new_base, f'path_schoyen_t{target}', 'path_invariance_per_seed.csv')
                if p:
                    path_cfg = load_yaml(self.code_root / 'gpm_cmap' / 'configs' / 'path_invariance.yaml')['analysis']
                    for z in self.tracked_csv(p):
                        specs=[('loop_holonomy',path_cfg.get('max_loop_holonomy',.35)),
                               ('zero_padding_phase_difference',path_cfg.get('max_padding_phase_difference',.25)),
                               ('path_identity_decoder_accuracy',float(z['path_identity_decoder_chance'])+path_cfg.get('path_decoder_chance_tolerance',.10)),
                               ('residual_time_decoder_r2',path_cfg.get('max_residual_time_r2',.20))]
                        passed=normalize_bool(z.get('path_independence_gate_passed'))
                        self.add(s,target,'path','path_independence_gate_passed',z.get('condition'),passed,p,budget=int(z.get('unique_training_episodes',-1)))
                        for metric,limit in specs:
                            self.add(s,target,'path','threshold_ratio',z.get('condition'),float(z[metric])/float(limit),p,transform=metric,budget=int(z.get('unique_training_episodes',-1)))

            nav_core = source.get('core_sha256')
            if nav_core:
                if nav_core in seen_nav:
                    self.errors.append(f'Duplicate best-validation NAV core: {seen_nav[nav_core]} and {s}')
                else:
                    seen_nav[nav_core] = s
            else:
                self.errors.append(f'Source {s}: missing best-validation navigation core hash')

            ctl_core = source.get('ctl_core_sha256')
            if ctl_core:
                if ctl_core in seen_ctl:
                    self.errors.append(f'Duplicate CTL core: {seen_ctl[ctl_core]} and {s}')
                else:
                    seen_ctl[ctl_core] = s
            else:
                self.errors.append(f'Source {s}: missing CTL core hash')

            self.sources.append(source)

        frame = pd.DataFrame(self.rows, columns=COLUMNS)
        keys = [c for c in COLUMNS if c not in {'value', 'artifact'}]
        if not frame.empty and frame.duplicated(keys).any():
            dup = frame.loc[frame.duplicated(keys, keep=False), keys + ['artifact']].sort_values(keys)
            raise RuntimeError('Duplicate source-target-metric cells; refusing ambiguous aggregation\n' + dup.head(50).to_string(index=False))

        return frame, self.sources, self.stages, self.files, list(dict.fromkeys(self.errors))


def copy_torus_auto(result_root: Path, out: Path):
    src = result_root / 'aggregate' / 'torus_auto'
    dst = out / 'torus_auto'
    if not src.is_dir():
        return False
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return True


def publish_symlink(public: Path, target: Path):
    public.parent.mkdir(parents=True, exist_ok=True)
    tmp = public.with_name(public.name + '.new')
    if tmp.exists() or tmp.is_symlink():
        if tmp.is_dir() and not tmp.is_symlink():
            shutil.rmtree(tmp)
        else:
            tmp.unlink()
    tmp.symlink_to(os.path.relpath(target, public.parent), target_is_directory=True)
    if public.exists() or public.is_symlink():
        if public.is_symlink() or public.is_file():
            public.unlink()
        else:
            backup = public.with_name(public.name + '.backup_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
            public.rename(backup)
    os.replace(tmp, public)


def run(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root).resolve()
    result_root = Path(args.result_root).resolve()
    code_root = Path(args.code_root).resolve()
    plan_path = run_root / 'plan.json'

    required = [
        plan_path,
        code_root / 'multinav' / 'collect.py',
        code_root / 'multinav' / 'plot.py',
        result_root / 'sources',
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError('Required input(s) missing:\n  ' + '\n  '.join(missing))

    sys.path.insert(0, str(code_root))
    package_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_dir))
    from multinav.collect import source_summary, infer
    from plot_best_validation_full_v2 import BestValidationFigures

    original_plan = read_json(plan_path)
    plan = build_analysis_plan(run_root, code_root, original_plan)

    lock_path = result_root / '.aggregate_full.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        attempts = result_root / 'aggregate_full_attempts'
        attempts.mkdir(parents=True, exist_ok=True)
        nums = []
        for p in attempts.glob('try_*'):
            try:
                nums.append(int(p.name.split('_')[-1]))
            except ValueError:
                pass
        out = attempts / f'try_{(max(nums) + 1 if nums else 1):03d}'
        out.mkdir(parents=True, exist_ok=False)

        audit = {
            'schema': 'best_validation_unified_aggregate_audit_v2_schoyen_figureA3',
            'script_version': SCRIPT_VERSION,
            'started_utc': now(),
            'status': 'running',
            'original_run_root': str(run_root),
            'result_root': str(result_root),
            'code_root': str(code_root),
            'original_plan_sha256': original_plan.get('plan_sha256'),
            'analysis_run_id': plan['config']['run_id'],
            'checkpoint_policy': 'best_validation',
            'retired_components': [name for name, _ in RETIRED_COMPONENTS],
        }
        write_json(out / 'audit.json', audit)

        try:
            collector = HybridCollector(run_root, result_root, code_root, plan)
            frame, sources, stages, files, errors = collector.collect()

            # source_summary/infer use the original source/target blocking and
            # equal-source statistics, but operate only on the newly collected
            # best-validation behaviour frame.
            os.chdir(run_root)
            summaries, source_rows = source_summary(frame, plan)
            contrasts, details, pairs = infer(frame, plan, sources)

            frame.to_csv(out / 'target_level_metrics.csv', index=False)
            summaries.to_csv(out / 'source_level_summary.csv', index=False)
            source_rows.to_csv(out / 'source_means.csv', index=False)
            contrasts.to_csv(out / 'source_level_contrasts.csv', index=False)
            details.to_csv(out / 'source_paired_differences.csv', index=False)
            pairs.to_csv(out / 'target_paired_differences.csv', index=False)
            pd.DataFrame(stages).to_csv(out / 'stage_coverage.csv', index=False)
            pd.DataFrame(sources).to_csv(out / 'source_qualification.csv', index=False)
            write_json(out / 'source_qualification.json', sources)
            write_json(out / 'input_artifact_checksums.json', files)

            denominator = len(plan['source_pairs'])
            pass_rows = []
            for field in ['ctl_qualified', 'is_torus_like', 'schoyen_formal_topology_passed', 'source_exact_gate', 'source_fibre_gate']:
                count = sum(src.get(field) is True for src in sources)
                n_evaluable = sum(src.get(field) is not None for src in sources)
                interval = binomtest(count, denominator).proportion_ci(confidence_level=.95, method='exact')
                pass_rows.append(dict(
                    criterion=field, status='active', n_planned=denominator, n_evaluable=n_evaluable,
                    n_confirmed_pass=count, confirmed_pass_fraction=count / denominator,
                    binomial_low=interval.low, binomial_high=interval.high,
                    interpretation='Confirmed pass out of all planned navigation sources; scientific non-evaluability remains explicit.',
                ))
            pd.DataFrame(pass_rows).to_csv(out / 'source_gate_pass_counts.csv', index=False)

            gate_parts=[]
            for _, row in frame.iterrows():
                if row['experiment']=='formal_topology' and row['metric']=='log2_margin':
                    gate_parts.append(dict(source_seed=int(row.source_seed), gate='schoyen_euclidean_formal_topology', criterion=row.arm, log2_margin=float(row.value), passed=bool(row.value>=0), artifact=row.artifact))
                elif row['experiment']=='formal_topology' and row['metric']=='raw_value' and row.arm=='connected_H0':
                    gate_parts.append(dict(source_seed=int(row.source_seed), gate='schoyen_euclidean_formal_topology', criterion=row.arm, log2_margin=np.nan, passed=bool(float(row.value)==1.0), artifact=row.artifact))
                elif row['experiment']=='source_gate' and row['metric']=='log2_margin':
                    gate_parts.append(dict(source_seed=int(row.source_seed), gate=row.split, criterion=row.arm, log2_margin=float(row.value), passed=bool(row.value>=0), artifact=row.artifact))
            pd.DataFrame(gate_parts, columns=['source_seed','gate','criterion','log2_margin','passed','artifact']).to_csv(out / 'source_gate_components.csv', index=False)
            phase_rows=frame[(frame.experiment=='source_gate') & (frame.metric=='phase_chart_rmse_rad')].copy() if not frame.empty else pd.DataFrame()
            phase_rows.to_csv(out / 'schoyen_phase_chart_metrics.csv', index=False)

            retired_rows = [dict(component=name, status='retired_replaced', reason=reason) for name, reason in RETIRED_COMPONENTS]
            pd.DataFrame(retired_rows).to_csv(out / 'retired_analysis_components.csv', index=False)

            mechanism_rows=[]
            for src in sources:
                ss=int(src['source_seed'])
                for exp,metric in [('action','controlled_recurrent_reuse_gate_passed'),('path','path_independence_gate_passed')]:
                    z=frame[(frame.source_seed==ss)&(frame.experiment==exp)&(frame.metric==metric)] if not frame.empty else pd.DataFrame()
                    vals=z.value.astype(float).tolist() if not z.empty else []
                    mechanism_rows.append(dict(source_seed=ss,experiment=exp,n_target_rows=len(vals),all_target_gates_passed=bool(vals) and all(v>=0.5 for v in vals),formal_topology_passed=src.get('schoyen_formal_topology_passed'),source_exact_gate=src.get('source_exact_gate'),source_fibre_gate=src.get('source_fibre_gate')))
            mech_df=pd.DataFrame(mechanism_rows)
            mech_df.to_csv(out / 'mechanism_replication_by_source.csv', index=False)
            count_rows=[]
            for exp in ['action','path']:
                z=mech_df[mech_df.experiment==exp]
                count_rows.append(dict(experiment=exp,n_planned=denominator,n_evaluable=int((z.n_target_rows>0).sum()),n_all_target_gates_passed=int(z.all_target_gates_passed.sum())))
            pd.DataFrame(count_rows).to_csv(out / 'mechanism_replication_counts.csv', index=False)

            primary = contrasts[contrasts.scope == 'all_planned_sources'] if not contrasts.empty else contrasts
            if not primary.empty and not bool(primary.complete.all()):
                errors.append('One or more prespecified behavioural contrast families lack all planned source/target pairs')

            figures_dir = out / 'figures'
            BestValidationFigures(
                run_root,
                plan,
                figures_dir,
                source_rows,
                contrasts,
                details,
                sources,
                original_run_root=run_root,
                result_root=result_root,
            ).run()

            copied_torus = copy_torus_auto(result_root, out)

            audit.update(
                status='complete' if not errors else 'incomplete',
                ended_utc=now(),
                errors=errors,
                n_sources_planned=denominator,
                n_source_identities=sum(bool(s.get('core_sha256')) for s in sources),
                n_target_metric_rows=len(frame),
                n_source_metric_rows=len(source_rows),
                n_topology_complete=sum(s.get('topology_analysis_completed') is True for s in sources),
                n_torus_like=sum(s.get('is_torus_like') is True for s in sources),
                n_schoyen_formal_topology_pass=sum(s.get('schoyen_formal_topology_passed') is True for s in sources),
                n_source_reduced_schoyen=sum(s.get('source_reduced_schoyen_evaluable') is True for s in sources),
                n_source_exact_gate=sum(s.get('source_exact_gate') is True for s in sources),
                n_source_fibre_gate=sum(s.get('source_fibre_gate') is True for s in sources),
                torus_auto_summary_copied=copied_torus,
                strict_passed=not errors,
            )
            write_json(out / 'audit.json', audit)

            report = [
                '# Best-validation 多导航源统一汇总报告',
                '',
                f'- analysis run_id: `{plan["config"]["run_id"]}`',
                f'- 原 frozen run: `{run_root}`',
                f'- best-validation rerun: `{result_root}`',
                f'- 主迁移 checkpoint policy: `best_validation` (`navigation/best.pt`)',
                f'- 计划独立导航源：{denominator}',
                f'- Schøyen topology 完成：{audit["n_topology_complete"]}/{denominator}',
                f'- robust dual-metric (1,2,1) torus：{audit["n_torus_like"]}/{denominator}',
                f'- 汇总完整性：**{audit["status"]}**',
                '',
                '## 数据来源',
                '- Figure1 的 navigation 学习曲线与 FigureA4 的 CTL 训练/资格信息复用原 frozen run；这些 source-side 训练没有重新运行。',
                '- Figure2–Figure4、FigureA2 的认知/行为数据全部读取 `rerun_best_validation/sources/`。',
                '- FigureA1 topology 数据全部读取新的 `torus_topology` (Schøyen-style, validation best.pt)。',
                '- 原 balanced checkpoint 的 downstream 行为结果不会进入本次统计。',
                '',
                '## 统计口径',
                '先在每个导航源内配对相同目标种子的实验臂，再对目标重复取平均；每个导航源仅贡献一个差值。',
                '主区间为源均值的 95% t 区间；并保留源区块 bootstrap、符号翻转、符号检验和逐源剔除诊断。',
                'Holm 校正沿用冻结计划中的比较族。',
                '',
                '## FigureA3 新 formal-topology / mechanism 路线',
                f'- Schøyen Euclidean formal topology 通过：{audit["n_schoyen_formal_topology_pass"]}/{denominator}',
                f'- Schøyen-anchored reduced source 可评估：{audit["n_source_reduced_schoyen"]}/{denominator}',
                f'- exact-2D reduced gate：{audit["n_source_exact_gate"]}/{denominator}',
                f'- toroidal-base-plus-fibre gate：{audit["n_source_fibre_gate"]}/{denominator}',
                '- FigureA3a 使用 selected Schøyen candidate 的 Euclidean H1/H2 persistence-gap margin；H0=1 作为离散连通性 gate。',
                '- FigureA3b 使用独立 source-calibration trajectories 上建立的 Schøyen-selected Fourier phase chart，并复用原 reduced-dynamics/fibre held-out gates。',
                '- FigureA3c/d 使用新的 source_reduced_schoyen_v1 对已完成 best-validation factorial checkpoints 做 action-causality 与 path-invariance 分析。',
                '- 旧 topology_export / legacy formal_topology / cohomological phase anchor 明确 retired/replaced，不进入新机制统计。',
                '',
                '## 输出',
                '`source_level_contrasts.csv`：best-validation 行为主推断。',
                '`source_paired_differences.csv`：逐 navigation source 配对差值。',
                '`source_qualification.csv/json`：best.pt source 身份、CTL 与新 topology 资格。',
                '`stage_coverage.csv`：区分 original reused、best-validation rerun 和 retired stages。',
                '`figures/Figure1`–`Figure4`、`FigureA1`–`FigureA4`：逐面板 SVG/PDF/PNG/CSV；总图为 PNG/PDF contact sheet。',
                '`torus_auto/`：复制当前 10-source Schøyen cohort 汇总，便于同一目录归档。',
                '',
            ]
            if errors:
                report += ['## 需要处理的活动阶段缺失', *['- ' + e for e in errors], '']
            (out / 'REPORT_zh.md').write_text('\n'.join(report) + '\n', encoding='utf-8')

            public = result_root / 'aggregate_full'
            publish_symlink(public, out)
            print('Unified aggregate:', public)
            print('Figures:', public / 'figures')
            print('Status:', audit['status'])
            if args.strict and errors:
                return 2
            return 0

        except Exception:
            audit.update(status='failed', ended_utc=now(), traceback=traceback.format_exc())
            write_json(out / 'audit.json', audit)
            raise


def main() -> int:
    ap = argparse.ArgumentParser(description='Hybrid aggregate for best-validation behavioural rerun + Schøyen topology.')
    ap.add_argument('--run-root', required=True, help='Original multinav_paper_v1 run root containing plan.json/code/sources.')
    ap.add_argument('--result-root', required=True, help='Best-validation rerun root, normally <run-root>/rerun_best_validation.')
    ap.add_argument('--code-root', default=None, help='Current patched code root. Default: <run-root>/code.')
    ap.add_argument('--strict', action='store_true', help='Exit nonzero if any active stage/data cell is missing.')
    args = ap.parse_args()
    if args.code_root is None:
        args.code_root = str(Path(args.run_root) / 'code')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
