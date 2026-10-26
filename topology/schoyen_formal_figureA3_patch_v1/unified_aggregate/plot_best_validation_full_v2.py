from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from multinav.common import rel, read_json, write_json
from multinav.plot import Figures, label
from multinav.stats import mean_ci


class BestValidationFigures(Figures):
    """Paper figures for the best-validation rerun with hybrid source provenance.

    Navigation/CTL source diagnostics are read from the original frozen run.
    Downstream behaviour and Schøyen topology are read from rerun_best_validation.
    FigureA3 uses the Schøyen selected candidate's Euclidean persistent homology
    as formal topology and a new independent Fourier phase chart for mechanisms.
    """

    def __init__(
        self,
        root: Path,
        plan: dict,
        out: Path,
        source_frame: pd.DataFrame,
        contrasts: pd.DataFrame,
        details: pd.DataFrame,
        sources: list[dict],
        *,
        original_run_root: Path,
        result_root: Path,
    ):
        super().__init__(root, plan, out, source_frame, contrasts, details, sources)
        self.original_run_root = Path(original_run_root).resolve()
        self.result_root = Path(result_root).resolve()

    def _old_source(self, seed: int) -> Path:
        return self.original_run_root / 'sources' / f'nav_seed_{seed}'

    def _new_source(self, seed: int) -> Path:
        return self.result_root / 'sources' / f'nav_seed_{seed}'

    def representative_plots(self):
        s = self.representative
        old_base = self._old_source(s)
        new_base = self._new_source(s)
        identity = new_base / 'source_identity.json'
        if not identity.exists():
            self.note(
                'Figure1',
                'c_representative_unavailable',
                'c–d | Prespecified source unavailable',
                f'Source {s} was selected by seed order before outcomes. No replacement source is selected for illustration.',
            )
            return

        identity_payload = read_json(identity)
        if identity_payload.get('checkpoint_policy') != 'best_validation':
            self.note(
                'Figure1',
                'c_representative_unavailable',
                'c–d | Representative checkpoint policy mismatch',
                f'Source {s} rerun identity does not declare checkpoint_policy=best_validation.',
            )
            return

        # The navigation checkpoint analysis was already produced during the
        # original navigation training and is valid for best.pt independently
        # of the later downstream checkpoint-policy change.
        directory = old_base / 'artifacts' / 'navigation' / 'checkpoint_analysis' / 'best_validation'
        arrays = directory / 'nav_eval_arrays.npz'
        if arrays.exists():
            with np.load(arrays) as a:
                required = {'positions', 'decoded_positions', 'trajectory_ids', 'time_indices'}
                if required <= set(a.files):
                    pos = a['positions']
                    decoded = a['decoded_positions']
                    ids = a['trajectory_ids']
                    times = a['time_indices']
                    first = int(np.min(ids))
                    take = np.flatnonzero(ids == first)
                    take = take[np.argsort(times[take])]
                    fig, ax = self.create('Figure1', 'c_trajectory', f'c | Source {s}; best-validation checkpoint')
                    ax.plot(pos[take, 0], pos[take, 1], '-o', markersize=3, label='True')
                    ax.plot(decoded[take, 0], decoded[take, 1], '-o', markersize=3, label='Decoded')
                    ax.set_aspect('equal', adjustable='datalim')
                    ax.set_xlabel('x (simulator units)')
                    ax.set_ylabel('y (simulator units)')
                    ax.legend(frameon=False)
                    data = pd.DataFrame(
                        dict(
                            time=times[take],
                            true_x=pos[take, 0],
                            true_y=pos[take, 1],
                            decoded_x=decoded[take, 0],
                            decoded_y=decoded[take, 1],
                        )
                    )
                    self.save(
                        fig,
                        'Figure1',
                        'c_trajectory',
                        data,
                        scope='illustrative best-validation checkpoint; prespecified first source; not a replicate count',
                    )

        display = directory / 'multinav_grid_display.npz'
        if display.exists():
            with np.load(display) as g:
                scores = np.asarray(g['grid_scores'], dtype=float)
                maps = np.asarray(g['rate_maps'])
                units = np.asarray(g['unit_ids']).astype(int)
                order = np.argsort(-np.nan_to_num(scores, nan=-np.inf), kind='stable')
                for rank, j in enumerate(order[:4], 1):
                    rm = maps[j]
                    lo = np.nanmin(rm)
                    hi = np.nanmax(rm)
                    normalized = (rm - lo) / max(hi - lo, 1e-12)
                    fig, ax = self.create(
                        'Figure1',
                        f'd{rank}_map',
                        f'd{rank} | Source {s}, best-validation unit {units[j]}, G={scores[j]:.3f}',
                    )
                    im = ax.imshow(normalized.T, origin='lower', extent=(-1.1, 1.1, -1.1, 1.1))
                    fig.colorbar(im, ax=ax, label='Normalized activity')
                    ax.set_xlabel('x (simulator units)')
                    ax.set_ylabel('y (simulator units)')
                    self.save(
                        fig,
                        'Figure1',
                        f'd{rank}_map',
                        pd.DataFrame(normalized),
                        scope='best-validation checkpoint; top grid-score units in prespecified source',
                    )

                module = new_base / 'artifacts' / 'torus_topology' / 'torus_analysis_arrays.npz'
                selected = None
                if module.exists():
                    with np.load(module) as a:
                        if 'selected_unit_indices' in a.files:
                            selected = np.asarray(a['selected_unit_indices']).astype(int)

                fig, ax = self.create(
                    'Figure1',
                    'd5_grid_rank',
                    f'd5 | Source {s}: ' + ('auto-selected torus ensemble' if selected is not None else 'all best-validation units'),
                )
                if selected is not None:
                    # selected_unit_indices are unit IDs. Map them to the saved
                    # navigation display explicitly rather than assuming positional
                    # indexing into grid_scores.
                    score_by_unit = {int(u): float(v) for u, v in zip(units, scores)}
                    vals = np.asarray([score_by_unit[u] for u in selected if int(u) in score_by_unit], dtype=float)
                    vals = vals[np.isfinite(vals)]
                else:
                    vals = scores[np.isfinite(scores)]
                vals = np.sort(vals)[::-1]
                if vals.size:
                    ax.plot(np.arange(1, len(vals) + 1), vals)
                    ax.axhline(.3, linestyle=':')
                    ax.set_xlabel('Rank')
                    ax.set_ylabel('Grid score')
                    rank_df = pd.DataFrame({'rank': range(1, len(vals) + 1), 'grid_score': vals})
                else:
                    self.empty(ax)
                    rank_df = pd.DataFrame(columns=['rank', 'grid_score'])
                self.save(
                    fig,
                    'Figure1',
                    'd5_grid_rank',
                    rank_df,
                    scope='best-validation checkpoint; torus membership comes from new Schøyen analysis',
                )
        else:
            self.note(
                'Figure1',
                'd_missing_scores',
                'd | Best-validation unit displays unavailable',
                'The original navigation run did not save best_validation display arrays for the prespecified source; no archived balanced-checkpoint maps are substituted.',
            )

    def topology_coordinates(self):
        s = self.representative
        path = self._new_source(s) / 'artifacts' / 'torus_topology' / 'torus_analysis_arrays.npz'
        if path.exists():
            with np.load(path) as a:
                if {'umap3_embedding', 'spatial_bin_xy'} <= set(a.files):
                    u = np.asarray(a['umap3_embedding'])
                    pos = np.asarray(a['spatial_bin_xy'])
                    n = min(len(u), len(pos))
                    u = u[:n]
                    pos = pos[:n]
                    for dim, name in enumerate(['x', 'y']):
                        fig, ax = self.create(
                            'FigureA1',
                            f'c{dim+1}_coordinates',
                            f'c{dim+1} | Source {s}: auto-selected torus UMAP colored by spatial {name}',
                        )
                        im = ax.scatter(u[:, 0], u[:, 1], c=pos[:, dim], s=5)
                        fig.colorbar(im, ax=ax, label=f'spatial {name} bin')
                        ax.set_xlabel('UMAP 1')
                        ax.set_ylabel('UMAP 2')
                        self.save(
                            fig,
                            'FigureA1',
                            f'c{dim+1}_coordinates',
                            pd.DataFrame({'umap1': u[:, 0], 'umap2': u[:, 1], f'spatial_{name}': pos[:, dim]}),
                            scope='prespecified representative source; UMAP is visualization only',
                        )
                    return
        self.note(
            'FigureA1',
            'c_coordinates_unavailable',
            'c | Torus visualization unavailable',
            f'The prespecified illustrative source {s} has no evaluable auto-selected torus embedding.',
        )

    def _figure_a3(self):
        # A: formal topology is now the Euclidean PH result of the locked
        # Schoyen-selected candidate. H0==1 is discrete and reported in the note;
        # H1/H2 ratios have signed log2 margins around the 1.5 threshold.
        z = self.data('formal_topology', 'log2_margin')
        self.points(
            'FigureA3', 'a_formal_topology',
            'a | Schøyen Euclidean formal-topology margins',
            z, ylabel='Signed log2 criterion margin', reference=0,
        )

        # B: reuse the mature held-out reduced-model gates, but the phase anchor
        # is now the independent Fourier chart on the Schoyen-selected ensemble.
        z = self.data('source_gate', 'log2_margin')
        if not z.empty:
            z = z.copy()
            z['category'] = z['split'].map(label) + ' / ' + z['arm'].map(label)
        self.points(
            'FigureA3', 'b_reduced_model',
            'b | Schøyen-anchored reduced-model qualification',
            z, category='category', ylabel='Signed log2 criterion margin', reference=0,
        )

        formal = sum(src.get('schoyen_formal_topology_passed') is True for src in self.sources)
        evaluable = sum(src.get('source_reduced_schoyen_evaluable') is True for src in self.sources)
        exact = sum(src.get('source_exact_gate') is True for src in self.sources)
        fibre = sum(src.get('source_fibre_gate') is True for src in self.sources)
        rmses = [float(src['schoyen_phase_chart_test_rmse_rad']) for src in self.sources if src.get('schoyen_phase_chart_test_rmse_rad') is not None]
        rmse_text = 'missing' if not rmses else f'{np.mean(rmses):.3f} ± {np.std(rmses, ddof=1) if len(rmses)>1 else 0.0:.3f} rad'
        self.note(
            'FigureA3', 'b2_gate_denominators',
            'a–b | Formal/reduced-model denominators',
            f'Planned independent navigation sources: {self.K}.\n'
            f'Schøyen Euclidean formal topology passed: {formal}/{self.K}.\n'
            f'Schøyen-anchored reduced model evaluated: {evaluable}/{self.K}.\n'
            f'Exact-2D reduced gate passed: {exact}/{self.K}; toroidal-base-plus-fibre gate passed: {fibre}/{self.K}.\n'
            f'Mean held-out phase-chart RMSE: {rmse_text}.\n'
            'The coordinate chart is fitted only on an independent navigation calibration split; source test is not used for model selection.',
        )

        self.points(
            'FigureA3', 'c_action_null',
            'c | Minimum action-null / correct-action error ratio',
            self.data('action', 'minimum_confirmatory_null_ratio'),
            ylabel='Minimum confirmatory null ratio', reference=1.10,
        )
        z = self.data('path', 'threshold_ratio')
        if not z.empty:
            z = z.copy()
            z['category'] = z['arm'].map(label) + ' / ' + z['transform'].map(label)
        self.points(
            'FigureA3', 'd_path_diagnostics',
            'd | Path-diagnostic value / prespecified upper limit',
            z, category='category', ylabel='Ratio (<= 1 passes)', reference=1,
        )

    def run(self):
        cfg = self.plan['config']
        hidden = 32 if cfg['profile'] == 'smoke' else 4096
        self.note(
            'Figure1',
            'a_architecture',
            'a | Navigation pretraining and independent-source design',
            f'2D displacement -> {hidden}-unit rectified recurrent network -> place-code readout\n'
            'Transferred parameters: input projection and recurrent matrix.\n'
            f'Independent navigation sources: {self.K}; separate CTL source paired with each.\n'
            f'Source budget: {cfg["source_epochs"]} epochs x {cfg["source_steps_per_epoch"]} updates.\n'
            'Main source checkpoint rule: best navigation validation checkpoint (best.pt).\n'
            f'Representative displays: source {self.representative}, chosen by seed order, not outcomes.',
        )
        self.curve('Figure1', 'b1_loss', 'b1 | Navigation validation loss across sources', self.data('navigation', 'validation_loss'), ylabel='Validation task loss')
        self.curve('Figure1', 'b2_grid_fraction', 'b2 | Grid-score fraction across sources', self.data('navigation', 'grid_fraction'), ylabel='Fraction G > 0.3')
        self.representative_plots()

        self.note(
            'Figure2',
            'a_task',
            'a | Frozen-source transfer protocol',
            'Aligned 4 x 4 endpoint task. Start identity plus relation sequence predicts one of 16 endpoints.\n'
            'Target seeds are disjoint between source blocks and paired across arms within each block.\n'
            'Target repetitions are averaged before between-source uncertainty is calculated.\n'
            'All navigation transfer arms in this rerun use each source seed\'s best.pt checkpoint.',
        )
        self.curve(
            'Figure2',
            'b_training',
            'b | Trainable-condition validation accuracy',
            self.data('main', 'validation_accuracy', arm=['transfer_finetune', 'random_init', 'transfer_input_only', 'transfer_recurrent_only']),
            chance=.9,
        )
        arms = ['transfer_freeze', 'ctl01_frozen', 'random_frozen']
        self.points('Figure2', 'c_frozen_transfer', 'c | Frozen recurrent-source transfer', self.data('main', arm=arms), order=arms, reference=1/16, connect=True)
        self.effects('Figure2', 'd_paired_effects', 'd | Paired source differences', 'F2_frozen_sources')

        self.points('Figure3', 'a_mapping', 'a | Relation remapping controls', self.data('mapping'), order=['aligned', 'shuffled', 'mismatched_target'], reference=1/16, connect=True)
        z = self.data('test_transforms')
        z['category'] = z['arm'].map(label) + ' / ' + z['transform'] if not z.empty else []
        self.points('Figure3', 'b_fixed_interface', 'b | Test-only perturbations; no retraining', z, category='category', reference=1/16)
        z = self.data('adapter')
        if not z.empty:
            z['x'] = pd.Categorical(z['transform'], categories=list(dict.fromkeys(z['transform'])), ordered=True).codes
        self.adapter_plot(z)
        self.effects('Figure3', 'd_adapter_gain', 'd | Joint adapter-learning gain', 'F3_adapter_gain', scale=100)

        self.curve('Figure4', 'a_symbolic_learning', 'a | Learning arbitrary relation symbols', self.data('symbolic'), x='budget', logx=True, chance=1/16)
        self.effects('Figure4', 'b_budget1000', 'b | Navigation minus comparator at 1000 episodes', 'F4_accuracy_1000')
        self.effects('Figure4', 'c_low_budget_auc', 'c | Navigation minus comparator, log-AUC 10–200', 'F4_logAUC_10_200')
        maxbudget = 8 if cfg['profile'] == 'smoke' else 1000
        z = pd.concat(
            [self.data('symbolic', metric=m, budget=maxbudget, arm=['nav_frozen', 'ctl01_frozen', 'random_frozen']) for m in ['accuracy', 'no_recurrence_accuracy']],
            ignore_index=True,
        )
        if not z.empty:
            z['category'] = z.arm.map(label) + ' / ' + z.metric.map({'accuracy': 'intact', 'no_recurrence_accuracy': 'zeroed recurrence'})
        self.points('Figure4', 'd_recurrence', 'd | Post-training recurrence-use diagnostic', z, category='category', reference=1/16)

        complete = sum(src.get('topology_analysis_completed') is True for src in self.sources)
        passed = sum(src.get('is_torus_like') is True for src in self.sources)
        topo_cfg = cfg.get('schoyen_topology', {})
        self.note(
            'FigureA1',
            'a_topology_coverage',
            'a | Automatic hidden-layer torus discovery',
            f'Independent navigation sources planned: {self.K}.\n'
            f'Validation best.pt topology analyses completed: {complete}/{self.K}. Robust dual-metric (1,2,1) signatures: {passed}/{self.K}.\n'
            f'Each source: fresh TRUE-position clustering rollouts ({topo_cfg.get("cluster_trajectories", "?")} x {topo_cfg.get("cluster_seq_len", "?")}) and independent TDA rollouts ({topo_cfg.get("tda_trajectories", "?")} x {topo_cfg.get("tda_seq_len", "?")}).\n'
            'Band-like DBSCAN clusters are morphology-ranked automatically; a strict-to-broad candidate set is evaluated with Euclidean and cosine Ripser.\n'
            f'Representative displays use prespecified source {self.representative}.',
        )
        z = self.data('topology', 'lifetime', source_seed=self.representative)
        fig, ax = self.create('FigureA1', 'b_persistence', f'b | Ordered persistence lifetimes: source {self.representative}')
        if z.empty:
            self.empty(ax)
        else:
            for arm, g in z.groupby('arm'):
                ax.plot(g.x, g.value, '-o', markersize=3, label=arm)
            ax.legend(frameon=False)
            ax.set_xlabel('Rank')
            ax.set_ylabel('Persistence lifetime')
        self.save(fig, 'FigureA1', 'b_persistence', z, scope='prespecified representative source; Euclidean and cosine shown independently')
        self.topology_coordinates()
        z = self.data('topology', 'candidate_score', source_seed=self.representative)
        fig, ax = self.create('FigureA1', 'd_pca_sweep', f'd | Source {self.representative}: automatic cluster-ensemble candidates')
        if z.empty:
            self.empty(ax)
        else:
            order = np.arange(len(z))
            ax.bar(order, z.value)
            ax.axhline(max(float(topo_cfg.get('h1_gap_threshold', 1.5)), float(topo_cfg.get('h2_gap_threshold', 1.5))), linestyle=':')
            ax.set_xticks(order)
            ax.set_xticklabels(z.arm, rotation=45, ha='right')
            ax.set_ylabel('Robust minimum persistence-gap score')
        self.save(fig, 'FigureA1', 'd_pca_sweep', z, scope='automatic candidate ensembles; higher score means stronger weakest H1/H2 gap across both metrics')

        self.note(
            'FigureA2',
            'a_hard_task',
            'a | Harder endpoint-generalization protocol',
            '8 x 8 endpoints; training lengths 1–5; long test lengths 6–12; four held-out action bigrams.\n'
            'Six fixed conditions and seven fixed episode budgets. No new pilot selection.\n'
            'Training is not conditional on a successful topology or reduced-model gate.',
        )
        for panel, split in [('b', 'id'), ('c', 'composition')]:
            self.curve('FigureA2', panel + '_learning', panel + ' | ' + split + ' generalization', self.data('hard', split=split), x='budget', logx=True, chance=1/64)
        for split in ['id', 'long', 'composition']:
            self.points('FigureA2', 'd_' + split, 'd | 1000-episode ' + split + ' accuracy', self.data('hard', budget=maxbudget, split=split), reference=1/64)

        self._figure_a3()

        self.curve('FigureA4', 'a_control_training', 'a | Delayed-reconstruction source training', self.data('control', 'nmse'), ylabel='Validation normalized MSE', chance=[.5, 1.], xlabel='Epoch')
        self.curve('FigureA4', 'b_control_delays', 'b | Delay-wise error at selected checkpoint', self.data('control', 'selected_delay_nmse'), ylabel='Normalized MSE')
        qual = sum(src.get('ctl_qualified') is True for src in self.sources)
        text = f'CTL qualification passed: {qual}/{self.K}. The original frozen CTL sources are reused unchanged in the best-validation rerun.\n'
        for src in self.sources:
            reduction = 'missing' if src.get('ctl_error_reduction') is None else format(src['ctl_error_reduction'], '.4%')
            text += f'NAV {src["source_seed"]} / CTL {src["control_seed"]}: qualified={src.get("ctl_qualified")}; epoch={src.get("ctl_selected_epoch")}; NMSE reduction={reduction}\n'
        compatible = sum(src.get('ctl_tensor_compatibility', {}).get('production_full_transfer_shape_compatible') is True for src in self.sources)
        text += f'\nProduction tensor-shape compatibility: {compatible}/{self.K} independent CTL sources.'
        shapes = sorted({str((src.get('ctl_tensor_compatibility', {}).get('input_weight_shape'), src.get('ctl_tensor_compatibility', {}).get('recurrent_weight_shape'))) for src in self.sources})
        text += '\nObserved (input-weight shape, recurrent-weight shape): ' + ', '.join(shapes)
        text += '\nExpected formal shapes: (4096, 2) and (4096, 4096).'
        self.note('FigureA4', 'c_control_qualification', 'c | CTL source qualification and tensor compatibility', text)

        self.compose()
        write_json(self.out / 'figure_manifest.json', self.manifest)
