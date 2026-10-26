"""One LSF job = one independent source pair + its paper-only downstream stages."""
from __future__ import annotations
import argparse, copy, fcntl, os, subprocess, sys, traceback
from pathlib import Path
from .common import *


def file_inputs(value, root):
    result={}
    def visit(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if k not in {"output_dir","multinav_cell_cache"}: visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x: visit(v)
        elif isinstance(x,str) and ("/" in x or x.endswith((".pt",".npz",".json",".pkl"))):
            p=Path(x) if Path(x).is_absolute() else root/x
            if p.is_file(): result[str(p)]=sha(p)
    visit(value); return result


class Runner:
    def __init__(self, root, plan_path, source_seed):
        self.root=root; self.plan_path=plan_path; self.plan=read_json(plan_path); self.cfg=self.plan["config"]
        self.pair=next((p for p in self.plan["source_pairs"] if p["source_seed"]==source_seed),None)
        if self.pair is None: raise ValueError("Source seed is not in the frozen plan")
        self.base=root/self.plan["run_root"]/"sources"/f"nav_seed_{source_seed}"
        self.code=root/self.plan["code_root"]; self.base.mkdir(parents=True,exist_ok=True)
        self.errors=[]; self.lock_handle=None

    def lock(self):
        self.lock_handle=(self.base/".worker.lock").open("a")
        try: fcntl.flock(self.lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError("Another worker is already running this navigation seed")

    def artifact(self,name): return self.base/"artifacts"/name
    def relative(self,path): return rel(path,self.root)
    def state_path(self,name): return self.base/"state"/(safe_name(name)+".json")

    def stage(self,name,kind,cfg,required):
        safe_name(name); cfg=copy.deepcopy(cfg)
        inputs=file_inputs(cfg,self.root)
        fp=digest(dict(plan=self.plan["plan_sha256"],kind=kind,config=cfg,inputs=inputs))
        state_path=self.state_path(name)
        old=read_json(state_path) if state_path.exists() else {}
        if old.get("status")=="done":
            if old["fingerprint"] != fp: raise RuntimeError(f"Inputs changed for completed stage {name}; use a new run ID")
            for p,h in old["outputs"].items():
                if not (self.root/p).is_file() or sha(self.root/p)!=h:
                    raise RuntimeError(f"Completed artifact missing/changed: {p}")
            print(f"[reuse] {name}",flush=True); return True
        tries=self.base/"attempts"/name; tries.mkdir(parents=True,exist_ok=True)
        number=1+max([int(p.name.split("_")[-1]) for p in tries.glob("try_*") if p.is_dir()]+[0])
        out=tries/f"try_{number:03d}"; out.mkdir()
        cfg["output_dir"]=self.relative(out)
        cfg["multinav_cell_cache"]=self.relative(self.base/"cell_cache"/name/fp)
        cp=self.base/"configs"/f"{name}_{number:03d}.yaml"; write_yaml(cp,cfg)
        log=self.base/"logs"/f"{name}_{number:03d}.log"; log.parent.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,"-u",str(self.code/"multinav"/"bootstrap.py"),"multinav.stage",kind,"--config",self.relative(cp)]
        env=os.environ.copy(); env["PYTHONPATH"]=str(self.code)+os.pathsep+env.get("PYTHONPATH","")
        env["PYTHONHASHSEED"]=str(cfg.get("seed",0)); env["MPLBACKEND"]="Agg"
        state=dict(stage=name,kind=kind,status="running",fingerprint=fp,started_utc=now(),
                   config=self.relative(cp),log=self.relative(log),attempt=self.relative(out),inputs=inputs,command=cmd)
        write_json(state_path,state); print(f"[run] {name} -> {log}",flush=True)
        try:
            with log.open("w",encoding="utf-8") as stream:
                subprocess.run(cmd,cwd=self.root,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
            paths=[]
            for pattern in required:
                found=list(out.glob(pattern))
                if not found or any(not p.is_file() or p.stat().st_size==0 for p in found):
                    raise RuntimeError(f"Missing required artifact {name}/{pattern}")
                paths.extend(found)
            outputs={self.relative(p):sha(p) for p in paths}
            public=self.artifact(name); public.parent.mkdir(parents=True,exist_ok=True)
            tmp=public.with_name(public.name+".new")
            if tmp.is_symlink(): tmp.unlink()
            tmp.symlink_to(os.path.relpath(out,public.parent)); os.replace(tmp,public)
            state.update(status="done",ended_utc=now(),outputs=outputs)
            write_json(state_path,state); return True
        except Exception as exc:
            state.update(status="failed",ended_utc=now(),error=repr(exc)); write_json(state_path,state)
            self.errors.append(name); print(f"[FAILED] {name}: {exc}",flush=True); return False

    def blocked(self,name,reason,scientific=False):
        write_json(self.state_path(name),dict(stage=name,status="not_evaluable" if scientific else "blocked",
                   reason=reason,scientific_nonpass=scientific))

    def base_config(self,name,seed):
        c=original_config(self.code,name); c["seed"]=int(seed); c["n_seeds"]=1; c["device"]=self.cfg["device"]
        c.pop("output_dir",None)
        c.setdefault("runtime",{}).update(cpu_num_threads=int(self.cfg["threads"]),allow_device_fallback=False,
                                         deterministic_algorithms=bool(self.cfg["deterministic_algorithms"]))
        if "data" in c: c["data"]["num_workers"]=0
        if self.cfg["profile"]=="smoke": self.smoke(c,name)
        return c

    def smoke(self,c,name):
        if "model" in c:
            c["model"]["hidden_dim"]=32
            if "content_dim" in c["model"]: c["model"]["content_dim"]=8
        d=c.setdefault("data",{}); t=c.setdefault("train",{}); a=c.setdefault("analysis",{})
        d.update(batch_size=8,num_workers=0)
        if "num_episodes" in d: d["num_episodes"]=160
        if "epochs" in t: t.update(epochs=1,eval_every=1)
        if name.startswith("nav_pretrain"):
            d.update(num_trajectories=16,place_cells=32,steps_per_epoch=1)
            a.update(val_num_trajectories=16,every_n_epochs=1,max_units_to_plot=4,grid_bins=10)
        if name=="sequence_control_pretrain":
            a["val_num_sequences"]=16; t["log_every_steps"]=0
        if name in {"abstract_pathint_hard","recurrent_factorial","action_causality","path_invariance"}:
            d.update(validation_episodes=16,id_test_episodes=32,long_test_episodes=32,composition_test_episodes=32,
                     episode_budgets=[4,8])
            t.update(max_gradient_updates=2,minimum_updates=1,eval_every_updates=1,early_stopping_patience_evals=2)
            if "confirmatory" in c: c["confirmatory"]["episode_budgets"]=[4,8]
            if "matched_random" in c: c["matched_random"].update(strict_matching_gate=False,iterations=2,singular_rank=4,probe_count=8)
            if "statistics" in c: c["statistics"]["bootstrap_samples"]=20
        if name=="symbolic_adapter_ctl01":
            c["episode_budgets"]=[4,8]; c["low_budget_max"]=8
            c["controls"].update(baseline_budgets=[4,8],recurrence_ablation_budgets=[8])
            c["controls"]["baseline_train"].update(epochs=1,eval_every=1)
        if name=="adapter_rescue": c["transforms"]=["none","signflip",{"name":"scale","factor":0.5}]

    def lock_source(self):
        import hashlib, torch
        from gpm_cmap.utils.checkpointing import load_checkpoint
        out=self.artifact("navigation")
        policy=self.cfg["checkpoint_policy"]
        balanced=out/"best_grid_balanced.pt"
        ckpt=balanced if policy=="balanced_else_validation" and balanced.exists() else out/"best.pt"
        payload=load_checkpoint(ckpt,map_location="cpu")
        h=hashlib.sha256()
        for key in ("grid_rnn.cell.in_proj.weight","grid_rnn.cell.h_proj.weight"):
            tensor=payload["model_state"][key].detach().cpu().contiguous().numpy()
            h.update(key.encode()); h.update(str(tensor.shape).encode()); h.update(tensor.tobytes())
        history=read_json(out/"history.json")
        epochs=int(self.cfg["source_epochs"])
        if len(history)!=epochs: raise RuntimeError("Navigation history does not match locked source budget")
        identity=dict(schema="multinav_source_identity_v1",source_seed=self.pair["source_seed"],
                      navigation_checkpoint=self.relative(ckpt),navigation_checkpoint_sha256=sha(ckpt),
                      core_sha256=h.hexdigest(),checkpoint_epoch=payload.get("epoch"),checkpoint_policy=policy,
                      balanced_checkpoint_available=balanced.exists(),
                      fallback_used=policy=="balanced_else_validation" and not balanced.exists(),
                      completed_epochs=len(history),completed_training_updates=len(history)*int(self.cfg["source_steps_per_epoch"]),
                      source_reference_kind="navigation_checkpoint_only",plan_sha256=self.plan["plan_sha256"])
        write_json(self.base/"source_identity.json",identity)
        return identity

    def run(self,only=None):
        self.lock(); verify_tree(self.code,self.plan["code_manifest"])
        cfg=self.cfg; s=self.pair["source_seed"]; cseed=self.pair["control_seed"]
        nav=self.base_config("nav_pretrain_full",s)
        nav["train"]["epochs"]=int(cfg["source_epochs"]); nav["data"]["steps_per_epoch"]=int(cfg["source_steps_per_epoch"])
        nav["analysis"]["torus"]["enabled"]=False
        # Per-checkpoint short-rollout analyses are retained, integrated (non-held-out) topology is disabled.
        ok_nav=self.stage("navigation","navigation",nav,["history.json","summary.json","best.pt","checkpoint_analysis/*/summary.json","checkpoint_analysis/*/multinav_grid_display.npz"])
        ctl=self.base_config("sequence_control_pretrain",cseed)
        ctl["train"]["epochs"]=int(cfg["source_epochs"]); ctl["data"]["steps_per_epoch"]=int(cfg["source_steps_per_epoch"])
        ok_ctl=self.stage("sequence_control","sequence_control",ctl,["best_validation.pt","training_history.json","training_gate.json","checkpoint_compatibility.json"])
        if not ok_nav:
            self.blocked("downstream","Navigation training failed; no replacement seed is selected")
            return self.finish()
        identity=self.lock_source(); navckpt=identity["navigation_checkpoint"]
        ctlckpt=self.relative(self.artifact("sequence_control")/"best_validation.pt")
        qualification=read_json(self.artifact("sequence_control")/"training_gate.json") if ok_ctl else None
        if ok_ctl:
            import hashlib
            from gpm_cmap.utils.checkpointing import load_checkpoint
            control=load_checkpoint(ctlckpt,map_location="cpu"); h=hashlib.sha256()
            for key in ("grid_rnn.cell.in_proj.weight","grid_rnn.cell.h_proj.weight"):
                tensor=control["model_state"][key].detach().cpu().contiguous().numpy()
                h.update(key.encode());h.update(str(tensor.shape).encode());h.update(tensor.tobytes())
            write_json(self.base/"control_identity.json",dict(source_seed=s,control_seed=cseed,
                checkpoint=ctlckpt,checkpoint_sha256=sha(ctlckpt),core_sha256=h.hexdigest(),
                checkpoint_epoch=control.get("epoch"),plan_sha256=self.plan["plan_sha256"]))
        write_json(self.base/"control_qualification.json",dict(source_seed=s,control_seed=cseed,
                   policy="retain_all_and_flag",qualification=qualification,training_completed=ok_ctl))
        if only == "sources": return self.finish(partial=True)
        # Main 4x4 tasks share each source-specific target seed across all arms.
        for target in self.pair["main_target_seeds"]:
            common=self.base_config("abstract_pathint",target); common["transfer_checkpoint"]=navckpt
            common["transfer"]={"mode":"full","checkpoint_candidates":[navckpt]}
            for k in list(common["controls"]):
                if k.startswith("run_"): common["controls"][k]=False
            main_name=f"exp5_nav_t{target}"
            main_ok=self.stage(main_name,"exp5",common,["raw_per_seed.json","seed_*/*/history.json","seed_*/*/*_model.pt"])
            if ok_ctl:
                cc=copy.deepcopy(common); cc["transfer_checkpoint"]=ctlckpt; cc["transfer"]={"mode":"full","checkpoint_candidates":[ctlckpt]}
                self.stage(f"exp5_ctl_t{target}","exp5_ctl",cc,["raw_per_seed.json","transfer_source_manifest.json"])
            else: self.blocked(f"exp5_ctl_t{target}","CTL source training failed")
            self.stage(f"mapping_t{target}","mapping",common,["raw_per_seed.json"])
            if main_ok:
                vt=copy.deepcopy(common); vt["multinav_main_output"]=self.relative(self.artifact(main_name))
                self.stage(f"test_transforms_t{target}","test_transforms",vt,["test_transform_rows.json"])
            else: self.blocked(f"test_transforms_t{target}","Main frozen target checkpoint unavailable")
            ar=self.base_config("adapter_rescue",target); ar["nav_checkpoints"]=[navckpt]
            self.stage(f"adapter_t{target}","adapter",ar,["adapter_rescue_rows.json","adapter_rescue_rows.csv"])
            sy=self.base_config("symbolic_adapter_ctl01",target)
            sy["pretrained_sources"]=[dict(arm="nav_frozen",checkpoint=navckpt,source_type="navigation",require_gate=False)]
            if ok_ctl: sy["pretrained_sources"].append(dict(arm="ctl01_frozen",checkpoint=ctlckpt,source_type="sequence_control",require_gate=False))
            self.stage(f"symbolic_t{target}","symbolic",sy,["symbolic_adapter_rows.csv","symbolic_adapter_rows.json","protocol_manifest.json"])
        # Behavioural replication must not be gated on whether topology exists.
        for label,template,targets,kind,required in [
            ("hard","abstract_pathint_hard",self.pair["hard_target_seeds"],"hard",["hard_learning_curves.csv","checkpoint_manifest.json"]),
            ("factorial","recurrent_factorial",self.pair["factorial_target_seeds"],"factorial",["factorial_learning_curves.csv","factorial_per_seed.csv","checkpoint_manifest.json"]),
        ]:
            for target in targets:
                hc=self.base_config(template,target); hc["navigation_checkpoint"]=navckpt
                hc["source_reduced_artifact"]=self.relative(self.base/"source_identity.json")
                hc["confirmatory"]["target_seeds"]=[target]
                if kind=="factorial": hc.setdefault("statistics",{})["confirmatory_comparisons"]=self.plan["factorial_contrasts"]
                self.stage(f"{label}_t{target}",kind,hc,required)
        # New topology route: always analyse the VALIDATION checkpoint best.pt.
        # It replaces the legacy topology_export + grid-score-selected analyze_torus path.
        validation_ckpt=self.relative(self.artifact("navigation")/"best.pt")
        topo=copy.deepcopy(nav)
        topo.update(multinav_checkpoint=validation_ckpt)
        topo.setdefault("analysis",{})["schoyen_torus"]=copy.deepcopy(cfg["schoyen_topology"])
        topology_ok=self.stage("torus_topology","topology",topo,
            ["torus_analysis_summary.json","torus_analysis_arrays.npz","selected_ensemble.json","run_provenance.json"])
        summary=read_json(self.artifact("torus_topology")/"torus_analysis_summary.json") if topology_ok else {}
        persistent=summary.get("persistent_cohomology",{}) if summary else {}
        selected=summary.get("selection",{}) if summary else {}
        write_json(self.base/"topology_qualification.json",dict(
            source_seed=s,
            analysis_completed=topology_ok,
            checkpoint_role="best_validation",
            checkpoint=validation_ckpt,
            is_torus_like=summary.get("is_torus_like"),
            status=summary.get("status"),
            selected_clusters=selected.get("selected_clusters"),
            n_selected_units=selected.get("n_selected_units"),
            robust_min_gap_score=selected.get("robust_min_gap_score"),
            euclidean=(persistent.get("metrics") or {}).get("euclidean"),
            cosine=(persistent.get("metrics") or {}).get("cosine"),
            usable_for_source_v3=False,
            note="Automatic Schoyen-style hidden-layer torus discovery on validation best.pt. Legacy Source-v3 phase-coordinate contract is intentionally not produced."
        ))
        # The old Source-v3 fit required the legacy cohomological phase-coordinate
        # artifact. That is a different scientific object from the new automatic
        # torus discovery and is therefore retired with the old topology route.
        for st in ("source_splits","source_reduced"):
            self.blocked(st,"Retired with legacy topology route; new torus analysis is representational only",scientific=True)
        for target in self.pair["factorial_target_seeds"]:
            for kind in ("action","path"):
                self.blocked(f"{kind}_t{target}","Legacy Source-v3 phase reference retired with old topology route",scientific=True)
        return self.finish()

    def finish(self,partial=False):
        states=[read_json(p) for p in sorted((self.base/"state").glob("*.json"))]
        failures=[x["stage"] for x in states if x["status"]=="failed"]
        write_json(self.base/"source_status.json",dict(source_seed=self.pair["source_seed"],
            control_seed=self.pair["control_seed"],source_pair=self.pair,profile=self.cfg["profile"],
            status="partial_sources_only" if partial else "failed" if failures else "finished",
            failed_stages=failures,plan_sha256=self.plan["plan_sha256"],ended_utc=now(),
            scientific_nonpasses=[x for x in states if x["status"]=="not_evaluable"],stages=states))
        return 1 if failures else 0


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--plan",required=True)
    ap.add_argument("--source-seed",required=True,type=int); ap.add_argument("--project")
    ap.add_argument("--sources-only",action="store_true")
    args=ap.parse_args(); root=project_root(args.project); runner=Runner(root,root/args.plan,args.source_seed)
    try: code=runner.run(only="sources" if args.sources_only else None)
    except Exception:
        traceback.print_exc(); runner.errors.append("worker_exception")
        write_json(runner.base/"worker_exception.json",dict(time=now(),traceback=traceback.format_exc()))
        code=2
    raise SystemExit(code)

if __name__ == "__main__": main()
