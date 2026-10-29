#!/usr/bin/env python3
"""Freeze figure-specific source data from the latest final MultiNav results.

This adapter is intentionally independent of historical plotting folders.  It
reads the current aggregate tables plus committed current-run artifacts and
writes one explicit source-data directory per manuscript figure.

It supports both:
  1. a canonical from-scratch final run; and
  2. the historical hybrid layout in which navigation lives under ``sources``
     while best-validation downstream/Schøyen artifacts live under
     ``rerun_best_validation``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import yaml

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # figures/
from paper_figures.common import FigureContext, load_csv, make_context, read_json, sha256, write_json  # noqa: E402


def _write(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def _copy_if_exists(src: Path, dst: Path) -> Path | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _subset(frame: pd.DataFrame, **filters) -> pd.DataFrame:
    z = frame.copy()
    if z.empty:
        return z
    for key, value in filters.items():
        if key not in z.columns:
            return z.iloc[0:0].copy()
        if isinstance(value, (list, tuple, set)):
            z = z[z[key].isin(value)]
        else:
            z = z[z[key] == value]
    return z.copy()


def _best_validation_epoch(history: list[dict]) -> int | None:
    valid=[]
    for row in history:
        try:
            value=float(row.get("validation_task_loss"))
            epoch=int(row.get("epoch"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            valid.append((value, epoch))
    return min(valid)[1] if valid else None


def _navigation_histories(ctx: FigureContext, inputs: list[Path]) -> pd.DataFrame:
    rows=[]
    for pair in ctx.source_pairs:
        seed=int(pair["source_seed"])
        nav=ctx.navigation_source_dir(seed)/"artifacts"/"navigation"
        hp=nav/"history.json"
        cp=nav/"config_used.yaml"
        if not hp.exists():
            continue
        inputs.append(hp)
        history=read_json(hp)
        selected=_best_validation_epoch(history)
        arena=2.2
        if cp.exists():
            inputs.append(cp)
            try:
                cfg=yaml.safe_load(cp.read_text(encoding="utf-8")) or {}
                arena=float((cfg.get("data") or {}).get("arena_size", arena))
            except Exception:
                pass
        for r in history:
            try: epoch=int(r["epoch"])
            except Exception: continue
            rel=r.get("path_integration_error_relative")
            try: rel=float(rel) if rel is not None else np.nan
            except Exception: rel=np.nan
            rows.append({
                "source_seed":seed,
                "epoch":epoch,
                "validation_loss":pd.to_numeric(r.get("validation_task_loss"),errors="coerce"),
                "pi_error_relative":rel,
                "pi_error_m":rel*arena if np.isfinite(rel) else np.nan,
                "arena_size_m":arena,
                "grid_fraction":pd.to_numeric(r.get("frac_grid_above_03"),errors="coerce"),
                "grid_fraction_conservative":pd.to_numeric(r.get("frac_grid_above_03_conservative"),errors="coerce"),
                "max_grid_score":pd.to_numeric(r.get("max_grid_score"),errors="coerce"),
                "selected_best_validation":bool(selected is not None and epoch==selected),
                "selected_epoch":selected,
            })
    return pd.DataFrame(rows)


def _representative_navigation(ctx: FigureContext, out: Path, inputs: list[Path]) -> None:
    out.mkdir(parents=True,exist_ok=True)
    seed=ctx.representative_source
    nav_source=ctx.navigation_source_dir(seed)
    base=nav_source/"artifacts"/"navigation"/"checkpoint_analysis"/"best_validation"
    meta={"source_seed":seed,"checkpoint_analysis":"best_validation"}
    arrays=base/"nav_eval_arrays.npz"
    if arrays.exists():
        inputs.append(arrays)
        with np.load(arrays,allow_pickle=False) as a:
            pos=np.asarray(a["positions"]); dec=np.asarray(a["decoded_positions"])
            ids=np.asarray(a["trajectory_ids"]); times=np.asarray(a["time_indices"])
            first=int(np.min(ids)); take=np.flatnonzero(ids==first); take=take[np.argsort(times[take])]
            _write(pd.DataFrame({"time":times[take],"true_x":pos[take,0],"true_y":pos[take,1],"decoded_x":dec[take,0],"decoded_y":dec[take,1]}),out/"representative_trajectory.csv")
    display=base/"multinav_grid_display.npz"
    torus=ctx.analysis_source_dir(seed)/"artifacts"/"torus_topology"/"torus_analysis_arrays.npz"
    selected_ids=None
    if torus.exists():
        inputs.append(torus)
        with np.load(torus,allow_pickle=False) as a:
            if "selected_unit_indices" in a.files:
                selected_ids=np.asarray(a["selected_unit_indices"],dtype=int)
    if display.exists():
        inputs.append(display)
        with np.load(display,allow_pickle=False) as g:
            unit_ids=np.asarray(g["unit_ids"],dtype=int)
            scores=np.asarray(g["grid_scores"],dtype=float)
            maps=np.asarray(g["rate_maps"],dtype=float)
        selected_mask=np.isin(unit_ids,selected_ids) if selected_ids is not None else np.zeros(len(unit_ids),dtype=bool)
        frame=pd.DataFrame({"unit":unit_ids,"grid_score":scores,"selected_by_schoyen":selected_mask})
        frame=frame.sort_values("grid_score",ascending=False,na_position="last",kind="stable").reset_index(drop=True)
        frame.insert(0,"rank",np.arange(1,len(frame)+1))
        _write(frame,out/"representative_grid_scores.csv")
        finite=np.nan_to_num(scores,nan=-np.inf); order=np.argsort(-finite,kind="stable")
        for rank,j in enumerate(order[:4],1):
            rm=np.asarray(maps[j],dtype=float); lo=np.nanmin(rm); hi=np.nanmax(rm)
            normalized=(rm-lo)/max(hi-lo,1e-12); yy,xx=np.indices(normalized.shape)
            _write(pd.DataFrame({"x_bin":xx.ravel(),"y_bin":yy.ravel(),"normalized_activation":normalized.ravel(),"unit":int(unit_ids[j]),"grid_score":float(scores[j]),"selected_by_schoyen":bool(selected_mask[j])}),out/f"rate_map_rank_{rank}.csv")
        meta["selected_n_units"]=None if selected_ids is None else int(len(selected_ids))
    write_json(out/"representative_metadata.json",meta)


def _representative_topology(ctx: FigureContext, source_means: pd.DataFrame, out: Path, inputs: list[Path]) -> None:
    out.mkdir(parents=True,exist_ok=True)
    seed=ctx.representative_source
    z=_subset(source_means,experiment="topology",source_seed=seed)
    _write(z,out/"representative_topology_metrics.csv")
    base=ctx.analysis_source_dir(seed)/"artifacts"/"torus_topology"
    arrays=base/"torus_analysis_arrays.npz"
    if arrays.exists():
        inputs.append(arrays)
        with np.load(arrays,allow_pickle=False) as a:
            if {"umap3_embedding","spatial_bin_xy"} <= set(a.files):
                u=np.asarray(a["umap3_embedding"],dtype=float); pos=np.asarray(a["spatial_bin_xy"],dtype=float)
                n=min(len(u),len(pos));u=u[:n];pos=pos[:n]
                _write(pd.DataFrame({"umap1":u[:,0],"umap2":u[:,1],"umap3":u[:,2],"spatial_x":pos[:,0],"spatial_y":pos[:,1]}),out/"representative_umap.csv")
    cand=base/"candidate_topology.csv"
    if cand.exists(): inputs.append(cand); _copy_if_exists(cand,out/"representative_candidates.csv")
    summary=base/"torus_analysis_summary.json"
    if summary.exists(): inputs.append(summary); _copy_if_exists(summary,out/"representative_topology_summary.json")
    coho=ctx.analysis_source_dir(seed)/"artifacts"/"figureA4_cohomology_torus"
    coho_csv=coho/"circular_coordinates.csv"
    coho_meta=coho/"analysis_summary.json"
    if coho_csv.exists(): inputs.append(coho_csv); _copy_if_exists(coho_csv,out/"representative_cohomology_torus.csv")
    if coho_meta.exists(): inputs.append(coho_meta); _copy_if_exists(coho_meta,out/"representative_cohomology_torus_meta.json")



def _formal_topology_from_gates(source_means: pd.DataFrame, gates: pd.DataFrame) -> pd.DataFrame:
    """Return finite Schøyen-Euclidean formal-topology margins.

    Historical hybrid aggregation can contain placeholder ``formal_topology``
    rows with NaN values even though ``source_gate_components.csv`` contains
    the complete formal-topology checks.  Prefer finite source-means rows, but
    reconstruct them from the gate table when needed.
    """
    current=_subset(source_means, experiment="formal_topology")
    finite=current[(current.get("metric", pd.Series(dtype=str)).astype(str)=="log2_margin") & pd.to_numeric(current.get("value"),errors="coerce").notna()] if not current.empty else current
    if not finite.empty:
        return current
    if gates.empty or "gate" not in gates.columns:
        return current
    g=gates[gates["gate"].astype(str)=="schoyen_euclidean_formal_topology"].copy()
    if g.empty:
        return current
    rows=[]
    for row in g.to_dict("records"):
        margin=pd.to_numeric(row.get("log2_margin"),errors="coerce")
        if not np.isfinite(margin):
            value=pd.to_numeric(row.get("value"),errors="coerce")
            threshold=pd.to_numeric(row.get("threshold"),errors="coerce")
            op=str(row.get("operator", ""))
            if np.isfinite(value) and np.isfinite(threshold) and value>0 and threshold>0 and op in {">=","<="}:
                margin=np.log2(value/threshold) if op==">=" else np.log2(threshold/value)
        if not np.isfinite(margin):
            continue
        rows.append(dict(
            source_seed=int(row["source_seed"]), target_seed=-1,
            experiment="formal_topology", metric="log2_margin",
            arm=str(row.get("criterion","")), budget=-1, transform="",
            split="euclidean", x=-1, value=float(margin),
            artifact=str(row.get("artifact","")),
        ))
    return pd.DataFrame(rows)


def _pick_latest_by_source(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    z=frame.copy()
    for key in ("budget","x"):
        if key in z.columns:
            q=pd.to_numeric(z[key],errors="coerce")
            if q.notna().any():
                z["__ord"]=q
                mx=z.groupby("source_seed")["__ord"].transform("max")
                z=z[z["__ord"]==mx].drop(columns="__ord")
                break
    if "source_seed" in z.columns and z.duplicated("source_seed").any():
        aggregations={c:"first" for c in z.columns if c not in {"source_seed","value"}}
        aggregations["value"]="mean"
        z=z.groupby("source_seed",as_index=False).agg(aggregations)
    return z


def _recurrence_table(source_means: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    """Stage one intact and one no-recurrence value per source/arm.

    Falls back to the already aggregated paired recurrence contrast if the
    absolute no-recurrence row is absent in a historical aggregate.
    """
    arms=["nav_frozen","ctl01_frozen","random_frozen"]
    sym=_subset(source_means,experiment="symbolic")
    out=[]
    for arm in arms:
        intact=_pick_latest_by_source(sym[(sym.get("arm")==arm)&(sym.get("metric")=="accuracy")].copy()) if not sym.empty else pd.DataFrame()
        zero=_pick_latest_by_source(sym[(sym.get("arm")==arm)&(sym.get("metric")=="no_recurrence_accuracy")].copy()) if not sym.empty else pd.DataFrame()
        intact_map={int(r.source_seed):float(r.value) for r in intact.itertuples() if np.isfinite(pd.to_numeric(r.value,errors="coerce"))}
        zero_map={int(r.source_seed):float(r.value) for r in zero.itertuples() if np.isfinite(pd.to_numeric(r.value,errors="coerce"))}
        contrast=f"intact_minus_zeroed_{arm}"
        pp=paired[(paired.get("family")=="F4_recurrence")&(paired.get("contrast")==contrast)] if not paired.empty else pd.DataFrame()
        diff_map={int(r.source_seed):float(r.difference) for r in pp.itertuples() if np.isfinite(pd.to_numeric(r.difference,errors="coerce"))}
        for seed,val in intact_map.items():
            out.append(dict(source_seed=seed,arm=arm,state="intact",value=val))
            z=zero_map.get(seed)
            if z is None and seed in diff_map:
                z=val-diff_map[seed]
            if z is not None and np.isfinite(z):
                out.append(dict(source_seed=seed,arm=arm,state="no_recurrence",value=float(z)))
    return pd.DataFrame(out).sort_values(["arm","source_seed","state"]) if out else pd.DataFrame(columns=["source_seed","arm","state","value"])


def _association_table_from_aggregate(source_means: pd.DataFrame, qualification: pd.DataFrame, paired: pd.DataFrame, navhist: pd.DataFrame) -> pd.DataFrame:
    """Build A6 entirely from aggregate tables when possible.

    Grid fraction is selected at each source's locked checkpoint epoch using
    ``source_qualification.csv`` + navigation rows in ``source_means.csv``.
    Raw navigation histories are only a fallback.
    """
    grid=[]
    nav=_subset(source_means,experiment="navigation",metric="grid_fraction")
    if not nav.empty and not qualification.empty and "checkpoint_epoch" in qualification.columns:
        qq=qualification[["source_seed","checkpoint_epoch"]].copy();qq["checkpoint_epoch"]=pd.to_numeric(qq["checkpoint_epoch"],errors="coerce")
        nn=nav[["source_seed","x","value"]].copy();nn["x"]=pd.to_numeric(nn["x"],errors="coerce")
        m=nn.merge(qq,on="source_seed",how="inner");m=m[m["x"]==m["checkpoint_epoch"]]
        grid=m[["source_seed","checkpoint_epoch","value"]].rename(columns={"checkpoint_epoch":"selected_epoch","value":"grid_fraction"}).copy()
    if isinstance(grid,list) or len(grid)==0:
        selected=navhist[navhist.selected_best_validation.astype(bool)].copy() if not navhist.empty else pd.DataFrame()
        grid=selected[["source_seed","selected_epoch","grid_fraction"]].copy() if not selected.empty else pd.DataFrame(columns=["source_seed","selected_epoch","grid_fraction"])
    out=grid.copy()
    specs=[
        ("F2_frozen_sources","transfer_freeze__minus__ctl01_frozen","aligned_nav_minus_dr"),
        ("F4_accuracy_1000","nav__minus__ctl01_frozen","symbolic1000_nav_minus_dr"),
        ("F4_logAUC_10_200","nav__minus__ctl01_frozen","symbolic_low_auc_nav_minus_dr"),
    ]
    for family,contrast,name in specs:
        z=paired[(paired.get("family")==family)&(paired.get("contrast")==contrast)][["source_seed","difference"]].rename(columns={"difference":name}) if not paired.empty else pd.DataFrame()
        out=out.merge(z,on="source_seed",how="left") if not z.empty else out.assign(**{name:np.nan})
    return out.sort_values("source_seed")

def _association_table(navhist: pd.DataFrame, paired: pd.DataFrame) -> pd.DataFrame:
    selected=navhist[navhist.selected_best_validation.astype(bool)].copy() if not navhist.empty else pd.DataFrame()
    if selected.empty:
        return pd.DataFrame()
    out=selected[["source_seed","selected_epoch","grid_fraction","pi_error_relative"]].copy()
    specs=[
        ("F2_frozen_sources","transfer_freeze__minus__ctl01_frozen","aligned_nav_minus_dr"),
        ("F4_accuracy_1000","nav__minus__ctl01_frozen","symbolic1000_nav_minus_dr"),
        ("F4_logAUC_10_200","nav__minus__ctl01_frozen","symbolic_low_auc_nav_minus_dr"),
    ]
    for family,contrast,name in specs:
        z=paired[(paired.family==family)&(paired.contrast==contrast)][["source_seed","difference"]].rename(columns={"difference":name}) if not paired.empty else pd.DataFrame()
        if z.empty:
            out[name]=np.nan
        else:
            out=out.merge(z,on="source_seed",how="left")
    return out.sort_values("source_seed")


def prepare(context: FigureContext) -> dict:
    agg=context.aggregate_dir; src=context.source_data_dir; src.mkdir(parents=True,exist_ok=True); inputs=[]
    tables={
        "target":agg/"target_level_metrics.csv", "source":agg/"source_means.csv",
        "summary":agg/"source_level_summary.csv", "contrasts":agg/"source_level_contrasts.csv",
        "paired":agg/"source_paired_differences.csv", "qualification":agg/"source_qualification.csv",
        "gates":agg/"source_gate_components.csv", "gate_counts":agg/"source_gate_pass_counts.csv",
        "mechanism_counts":agg/"mechanism_replication_counts.csv",
        "mechanism_by_source":agg/"mechanism_replication_by_source.csv",
        "torus_summary":agg/"torus_auto"/"torus_auto_summary.csv",
        "torus_candidates":agg/"torus_auto"/"torus_auto_candidates.csv",
    }
    for p in tables.values():
        if p.exists(): inputs.append(p)
    frames={k:load_csv(p) for k,p in tables.items()}
    sm=frames["source"]; contrasts=frames["contrasts"]; paired=frames["paired"]
    navhist=_navigation_histories(context,inputs)

    # Figure2 — navigation source properties (Figure1 is external/hand-drawn).
    f=src/"Figure2";f.mkdir(exist_ok=True)
    _write(navhist,f/"navigation_history.csv"); _representative_navigation(context,f,inputs)

    # Figure3 — aligned transfer.
    f=src/"Figure3";f.mkdir(exist_ok=True)
    _write(_subset(sm,experiment="main",metric="validation_accuracy"),f/"training_curves.csv")
    _write(_subset(sm,experiment="main",metric="accuracy",arm=["transfer_freeze","ctl01_frozen","random_frozen"]),f/"frozen_transfer.csv")
    _write(_subset(contrasts,family="F2_frozen_sources"),f/"contrasts.csv")
    _write(_subset(paired,family="F2_frozen_sources"),f/"paired_differences.csv")

    # Figure4 — fixed interface perturbations + learned adapter rescue.
    f=src/"Figure4";f.mkdir(exist_ok=True)
    _write(_subset(sm,experiment="test_transforms"),f/"test_transforms.csv")
    _write(_subset(sm,experiment="adapter"),f/"adapter.csv")
    _write(_subset(contrasts,family="F3_adapter_gain"),f/"adapter_contrasts.csv")
    _write(_subset(paired,family="F3_adapter_gain"),f/"adapter_paired_differences.csv")

    # Figure5 — arbitrary symbols / finite-budget reuse.
    f=src/"Figure5";f.mkdir(exist_ok=True)
    _write(_subset(sm,experiment="symbolic"),f/"symbolic.csv")
    _write(_subset(contrasts,family=["F4_accuracy_1000","F4_logAUC_10_200","F4_recurrence"]),f/"contrasts.csv")
    _write(_subset(paired,family=["F4_accuracy_1000","F4_logAUC_10_200","F4_recurrence"]),f/"paired_differences.csv")
    _write(_recurrence_table(sm,paired),f/"recurrence.csv")

    # FigureA1 — delayed reconstruction source control.
    f=src/"FigureA1";f.mkdir(exist_ok=True)
    _write(_subset(sm,experiment="control"),f/"control_source_means.csv")
    _write(frames["qualification"],f/"source_qualification.csv")

    # FigureA2 — relation-label / target-mapping controls.
    f=src/"FigureA2";f.mkdir(exist_ok=True)
    _write(_subset(sm,experiment="mapping"),f/"mapping.csv")
    _write(_subset(contrasts,family="F3_mapping"),f/"contrasts.csv")
    _write(_subset(paired,family="F3_mapping"),f/"paired_differences.csv")

    # FigureA3 — Schøyen-formal mechanism analysis.
    f=src/"FigureA3";f.mkdir(exist_ok=True)
    _write(_formal_topology_from_gates(sm,frames["gates"]),f/"formal_topology.csv")
    _write(_subset(sm,experiment="source_gate"),f/"source_reduced_gates.csv")
    _write(_subset(sm,experiment="action"),f/"action.csv")
    _write(_subset(sm,experiment="path"),f/"path.csv")
    _write(frames["qualification"],f/"source_qualification.csv")
    _write(frames["gate_counts"],f/"source_gate_pass_counts.csv")
    _write(frames["mechanism_by_source"],f/"mechanism_replication_by_source.csv")
    _write(frames["mechanism_counts"],f/"mechanism_replication_counts.csv")

    # FigureA4 — current Schøyen population topology (not legacy cohomology).
    f=src/"FigureA4";f.mkdir(exist_ok=True)
    _write(frames["torus_summary"],f/"torus_auto_summary.csv")
    _write(frames["torus_candidates"],f/"torus_auto_candidates.csv")
    _representative_topology(context,sm,f,inputs)

    # FigureA5 — harder 8x8 endpoint benchmark.
    f=src/"FigureA5";f.mkdir(exist_ok=True)
    _write(_subset(sm,experiment="hard"),f/"hard_source_means.csv")
    _write(_subset(contrasts,family="A2_max_budget"),f/"contrasts.csv")
    _write(_subset(paired,family="A2_max_budget"),f/"paired_differences.csv")

    # FigureA6 — exploratory source-level association between validation-best
    # grid fraction and transfer advantages.  This is descriptive, not causal.
    f=src/"FigureA6";f.mkdir(exist_ok=True)
    _write(_association_table_from_aggregate(sm,frames["qualification"],paired,navhist),f/"grid_transfer_associations.csv")

    manifest={
        "run_id":context.plan["config"]["run_id"], "plan":str(context.plan_path),
        "aggregate_dir":str(context.aggregate_dir), "n_navigation_sources":context.n_sources,
        "representative_source":context.representative_source,
        "checkpoint_policy":context.final_checkpoint_policy,
        "representative_navigation_source_dir":str(context.navigation_source_dir(context.representative_source)),
        "representative_analysis_source_dir":str(context.analysis_source_dir(context.representative_source)),
        "inputs":{str(p.resolve()):sha256(p) for p in sorted(set(inputs)) if p.exists() and p.is_file()},
        "note":"Round-9 visual design adapted to the current best-validation + Schøyen data contract. No legacy external_results or legacy formal-topology artifact is read.",
    }
    write_json(src/"SOURCE_DATA_MANIFEST.json",manifest); return manifest


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--project",required=True);ap.add_argument("--plan",required=True);ap.add_argument("--aggregate",required=True);args=ap.parse_args()
    ctx=make_context(Path(args.project),Path(args.plan),Path(args.aggregate));prepare(ctx);print(ctx.source_data_dir)

if __name__=="__main__": main()
