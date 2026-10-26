#!/usr/bin/env python3
"""FigureA4C topology-derived circular coordinates from an existing Schøyen candidate.

This script is visualization-only. It reuses an already frozen PCA6/downsampled
candidate from the old best-validation rerun and does NOT alter formal topology
selection, FigureA3, or any behavioral result.

Pipeline
--------
existing candidate pca6_downsampled.npy
  -> deterministic greedy-permutation landmarks (default 500; mirrors n_perm=500)
  -> exact Euclidean Ripser H1 with cocycles on those landmarks
  -> take the two longest finite H1 classes
  -> smooth each cocycle into a circle-valued coordinate
  -> circular kNN interpolation from landmarks to all frozen 3000 points
  -> canonical torus:
       x=(R+r*cos(theta2))*cos(theta1)
       y=(R+r*cos(theta2))*sin(theta1)
       z=r*sin(theta2)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def candidate_key(labels) -> str:
    return "_".join(str(int(x)) for x in sorted(labels))


def locate_source_root(project: Path, seed: int) -> Path:
    candidates=[
        project/"rerun_best_validation"/"sources"/f"nav_seed_{seed}",
        project/"sources"/f"nav_seed_{seed}",
    ]
    for p in candidates:
        if p.exists():
            return p.resolve()
    raise FileNotFoundError("Could not find analysis source for seed %s:\n%s" %
                            (seed, "\n".join("  "+str(p) for p in candidates)))


def resolve_candidate(torus_dir: Path, requested: str):
    if requested == "selected":
        sel=torus_dir/"selected_ensemble.json"
        if not sel.exists():
            raise FileNotFoundError(sel)
        obj=json.loads(sel.read_text(encoding="utf-8"))
        labels=[int(x) for x in obj["selected_clusters"]]
    else:
        labels=[int(x) for x in requested.replace("_","+").split("+") if x.strip()]
    cdir=torus_dir/"candidates"/f"clusters_{candidate_key(labels)}"
    if not cdir.exists():
        raise FileNotFoundError(cdir)
    return labels,cdir


def greedy_farthest_indices(X: np.ndarray, n_landmarks: int) -> np.ndarray:
    """Deterministic Euclidean greedy permutation, starting at index 0."""
    X=np.asarray(X,dtype=np.float64)
    n=len(X)
    m=min(int(n_landmarks),n)
    out=np.empty(m,dtype=np.int64)
    out[0]=0
    d2=np.sum((X-X[0])**2,axis=1)
    d2[0]=-1.0
    for j in range(1,m):
        idx=int(np.argmax(d2))
        out[j]=idx
        newd2=np.sum((X-X[idx])**2,axis=1)
        d2=np.minimum(d2,newd2)
        d2[out[:j+1]]=-1.0
    return out


def finite_h1_order(h1: np.ndarray):
    h1=np.asarray(h1,dtype=float)
    finite=np.flatnonzero(np.isfinite(h1[:,1]))
    life=h1[finite,1]-h1[finite,0]
    good=np.isfinite(life)&(life>0)
    finite=finite[good]; life=life[good]
    order=np.argsort(life)[::-1]
    return finite[order],life[order]


def circular_coordinate_from_cocycle(cocycle, distance_matrix, threshold, coeff):
    """Same sparse least-squares cocycle smoothing used by the old codebase."""
    from scipy import sparse
    from scipy.sparse.linalg import lsmr

    dists=np.asarray(distance_matrix,dtype=np.float64)
    n=dists.shape[0]
    coc=np.asarray(cocycle,dtype=np.int64).copy()
    if coc.ndim!=2 or (coc.size and coc.shape[1]<3):
        raise ValueError("cocycle must have [i,j,value] columns")

    values_map={}
    half=int(coeff)/2.0
    for row in coc:
        i,j,value=int(row[0]),int(row[1]),int(row[2])
        if value>half:
            value-=int(coeff)
        if j>=i:
            key=(j,i); signed=float(value)
        else:
            key=(i,j); signed=float(-value)
        values_map[key]=signed

    rr,cc=np.tril_indices(n,k=-1)
    mask=np.isfinite(dists[rr,cc])&(dists[rr,cc]>0)&(dists[rr,cc]<=float(threshold))
    er=rr[mask]; ec=cc[mask]
    n_edges=len(er)
    if n_edges==0:
        raise RuntimeError("No VR edges at the circular-coordinate threshold")

    vertices=np.unique(np.concatenate([er,ec]))
    lookup=np.full(n,-1,dtype=np.int64)
    lookup[vertices]=np.arange(len(vertices),dtype=np.int64)

    rows=np.repeat(np.arange(n_edges,dtype=np.int64),2)
    cols=np.column_stack([lookup[er],lookup[ec]]).ravel()
    vals=np.tile(np.array([-1.0,1.0]),n_edges)
    incidence=sparse.coo_matrix((vals,(rows,cols)),shape=(n_edges,len(vertices))).tocsr()
    omega=np.fromiter((values_map.get((int(r),int(c)),0.0) for r,c in zip(er,ec)),
                      dtype=np.float64,count=n_edges)
    sol=lsmr(incidence,omega)
    coord=np.full(n,np.nan,dtype=np.float64)
    coord[vertices]=np.mod(sol[0],1.0)
    return coord, {"n_edges":int(n_edges),"n_vertices":int(len(vertices)),
                   "residual_norm":float(sol[3])}


def interpolate_circular(X_all, X_land, turns_land, landmark_indices, k=12):
    """Extend two landmark circle coordinates with weighted circular kNN means."""
    valid=np.isfinite(turns_land).all(axis=1)
    if int(valid.sum())<20:
        raise RuntimeError("Too few landmarks have both finite circular coordinates")
    XL=X_land[valid]
    TL=turns_land[valid]
    original_landmark_indices=np.asarray(landmark_indices,dtype=int)[valid]

    kk=min(int(k),len(XL))
    nn=NearestNeighbors(n_neighbors=kk,metric="euclidean").fit(XL)
    dist,ind=nn.kneighbors(X_all)
    scale=np.maximum(dist[:,-1],1e-9)
    w=np.exp(-((dist/scale[:,None])**2))
    denom=np.maximum(w.sum(axis=1),1e-12)

    out=np.empty((len(X_all),2),dtype=np.float64)
    confidence=np.empty_like(out)
    for d in range(2):
        unit=np.exp(2j*np.pi*TL[:,d])
        z=(w*unit[ind]).sum(axis=1)/denom
        mag=np.abs(z)
        phase=np.mod(np.angle(z)/(2*np.pi),1.0)
        # Near cancellation: use closest valid landmark rather than creating an
        # unstable angle.
        bad=mag<1e-5
        phase[bad]=TL[ind[bad,0],d]
        out[:,d]=phase
        confidence[:,d]=mag

    # Exact landmark values overwrite interpolation at the landmark locations.
    for local_idx,global_idx in enumerate(original_landmark_indices):
        out[int(global_idx)]=TL[local_idx]
        confidence[int(global_idx)]=1.0
    return out,confidence


def canonical_xyz(theta_rad,R=2.0,r=0.75):
    t1=theta_rad[:,0]; t2=theta_rad[:,1]
    x=(R+r*np.cos(t2))*np.cos(t1)
    y=(R+r*np.cos(t2))*np.sin(t1)
    z=r*np.sin(t2)
    return np.column_stack([x,y,z])


def plot_h1_bars(h1,path,top=30):
    order,life=finite_h1_order(h1)
    fig,ax=plt.subplots(figsize=(6.0,4.2))
    n=min(top,len(order))
    for rank,idx in enumerate(order[:n]):
        b,d=h1[int(idx),:2]
        ax.hlines(rank,b,d,linewidth=1.4)
    ax.invert_yaxis()
    ax.set_xlabel("Filtration radius")
    ax.set_ylabel("H1 lifetime rank")
    ax.set_title("Euclidean persistent cohomology on PCA6 landmarks")
    ax.grid(alpha=.18)
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


def plot_theta_square(df,path):
    fig,ax=plt.subplots(figsize=(5.2,4.8))
    sc=ax.scatter(df.theta1_turns,df.theta2_turns,c=df.pc1,cmap="viridis",
                  s=5,alpha=.65,linewidth=0,rasterized=True)
    ax.set_xlim(0,1);ax.set_ylim(0,1)
    ax.set_xlabel(r"$\theta_1/2\pi$")
    ax.set_ylabel(r"$\theta_2/2\pi$")
    ax.set_aspect("equal")
    ax.set_title("Topology-derived circular coordinates")
    fig.colorbar(sc,ax=ax,label="PC1")
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


def plot_torus(df,path,color_col,title):
    fig=plt.figure(figsize=(8.0,6.2))
    ax=fig.add_subplot(111,projection="3d")
    sc=ax.scatter(df.torus_x,df.torus_y,df.torus_z,c=df[color_col],cmap="viridis",
                  s=5,alpha=.70,linewidth=0,depthshade=False,rasterized=True)
    ax.view_init(elev=24,azim=-55)
    ax.set_axis_off()
    try: ax.set_box_aspect((1,1,.58))
    except Exception: pass
    fig.colorbar(sc,ax=ax,orientation="horizontal",fraction=.04,pad=.02,label=color_col)
    fig.suptitle(title,fontsize=11)
    fig.tight_layout();fig.savefig(path,dpi=240,bbox_inches="tight");plt.close(fig)


def plot_spatial_if_available(df,path):
    if not {"spatial_x","spatial_y"}.issubset(df.columns):
        return
    fig,axs=plt.subplots(1,2,figsize=(9.0,4.0))
    for j,ax in enumerate(axs,1):
        c=np.cos(2*np.pi*df[f"theta{j}_turns"].to_numpy(float))
        sc=ax.scatter(df.spatial_x,df.spatial_y,c=c,cmap="twilight",
                      s=5,alpha=.7,linewidth=0,rasterized=True)
        ax.set_aspect("equal"); ax.set_title(rf"$\cos(\theta_{j})$")
        ax.set_xlabel("spatial bin x")
        if j==1: ax.set_ylabel("spatial bin y")
        fig.colorbar(sc,ax=ax,fraction=.046,pad=.03)
    fig.suptitle("Decoded circular coordinates mapped back to physical space")
    fig.tight_layout();fig.savefig(path,dpi=220);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",required=True,type=Path)
    ap.add_argument("--source-seed",type=int,default=6101)
    ap.add_argument("--candidate",default="selected",
                    help="'selected' (default) or explicit cluster key such as 5 or 1+3+5")
    ap.add_argument("--landmarks",type=int,default=500)
    ap.add_argument("--interp-k",type=int,default=12)
    ap.add_argument("--coeff",type=int,default=47)
    ap.add_argument("--lifetime-fraction",type=float,default=.99)
    ap.add_argument("--R",type=float,default=2.0)
    ap.add_argument("--r",type=float,default=.75)
    ap.add_argument("--aggregate",type=Path,default=None)
    ap.add_argument("--overwrite",action="store_true")
    args=ap.parse_args()

    project=args.project.expanduser().resolve()
    source_root=locate_source_root(project,args.source_seed)
    torus=source_root/"artifacts"/"torus_topology"
    labels,cdir=resolve_candidate(torus,args.candidate)
    out=source_root/"artifacts"/"figureA4_cohomology_torus"
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} already exists; pass --overwrite to recompute")
    if args.overwrite and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)

    Xds=np.asarray(np.load(cdir/"pca6_downsampled.npy"),dtype=np.float64)
    if Xds.ndim!=2 or Xds.shape[1]!=6 or not np.isfinite(Xds).all():
        raise RuntimeError(f"Invalid PCA6 downsample cloud: {Xds.shape}")
    n_land=min(int(args.landmarks),len(Xds))
    if n_land<50: raise RuntimeError("Need at least 50 cohomology landmarks")

    print("="*80,flush=True)
    print("FigureA4C topology-derived circular coordinates",flush=True)
    print("host=",socket.gethostname(),flush=True)
    print("source=",args.source_seed,"candidate=",labels,flush=True)
    print("Xds=",Xds.shape,"landmarks=",n_land,flush=True)
    print("="*80,flush=True)

    t0=time.time()
    landmark_idx=greedy_farthest_indices(Xds,n_land)
    XL=Xds[landmark_idx]

    try:
        from ripser import ripser
    except Exception as exc:
        raise RuntimeError("ripser is required in the neuron environment") from exc

    # Exact H1 cocycles on the deterministic 500-point greedy-permutation cloud.
    rr=ripser(XL,maxdim=1,coeff=int(args.coeff),metric="euclidean",do_cocycles=True)
    h1=np.asarray(rr["dgms"][1],dtype=np.float64)
    order,lifetimes=finite_h1_order(h1)
    if len(order)<2:
        raise RuntimeError("Persistent cohomology returned fewer than two finite H1 classes")
    class_indices=order[:2].astype(int)
    second=int(class_indices[1])
    threshold=float(h1[second,0]+float(args.lifetime_fraction)*(h1[second,1]-h1[second,0]))

    D=pairwise_distances(XL,metric="euclidean")
    h1_cocycles=rr["cocycles"][1]
    coords=[];decode_diag=[]
    for class_idx in class_indices:
        c,diag=circular_coordinate_from_cocycle(
            h1_cocycles[int(class_idx)],D,threshold=threshold,coeff=int(args.coeff))
        coords.append(c); decode_diag.append(diag)
    turns_land=np.column_stack(coords)

    turns_all,confidence=interpolate_circular(
        Xds,XL,turns_land,landmark_idx,k=int(args.interp_k))
    theta=2*np.pi*turns_all
    xyz=canonical_xyz(theta,R=float(args.R),r=float(args.r))

    df=pd.DataFrame({
        "point_index":np.arange(len(Xds),dtype=int),
        "theta1_turns":turns_all[:,0],
        "theta2_turns":turns_all[:,1],
        "theta1_radians":theta[:,0],
        "theta2_radians":theta[:,1],
        "phase_confidence_1":confidence[:,0],
        "phase_confidence_2":confidence[:,1],
        "torus_x":xyz[:,0],"torus_y":xyz[:,1],"torus_z":xyz[:,2],
        "pc1":Xds[:,0],
        "is_cohomology_landmark":np.isin(np.arange(len(Xds)),landmark_idx),
    })

    # For the formal selected candidate, recover the physical spatial-bin
    # coordinates when the top-level selected arrays match this candidate.
    top=torus/"torus_analysis_arrays.npz"
    if top.exists():
        try:
            with np.load(top,allow_pickle=False) as a:
                top_labels=[int(x) for x in np.asarray(a["selected_cluster_labels"]).reshape(-1)]
                if sorted(top_labels)==sorted(labels):
                    full_pos=np.asarray(a["spatial_bin_xy"],dtype=float)
                    ds_idx=np.asarray(a["downsample_indices"],dtype=int)
                    if len(ds_idx)==len(Xds) and ds_idx.max()<len(full_pos):
                        pos=full_pos[ds_idx]
                        df["spatial_x"]=pos[:,0];df["spatial_y"]=pos[:,1]
        except Exception:
            pass

    df.to_csv(out/"circular_coordinates.csv",index=False)
    pd.DataFrame({
        "landmark_local_index":np.arange(len(XL),dtype=int),
        "point_index":landmark_idx,
        "theta1_turns":turns_land[:,0],
        "theta2_turns":turns_land[:,1],
        "pc1":XL[:,0],
    }).to_csv(out/"landmark_circular_coordinates.csv",index=False)
    np.savez_compressed(
        out/"cohomology_arrays.npz",
        pca6_downsampled=Xds.astype(np.float32),
        landmark_indices=landmark_idx.astype(np.int64),
        landmark_pca6=XL.astype(np.float32),
        H1_diagram=h1,
        H1_class_indices=class_indices,
        H1_cocycle_1=np.asarray(h1_cocycles[int(class_indices[0])]),
        H1_cocycle_2=np.asarray(h1_cocycles[int(class_indices[1])]),
        circular_turns_landmarks=turns_land,
        circular_turns_all=turns_all,
        canonical_torus_xyz=xyz,
    )

    plot_h1_bars(h1,out/"H1_persistent_cohomology.png")
    plot_theta_square(df,out/"theta1_theta2_square.png")
    plot_torus(df,out/"canonical_torus_pc1.png","pc1",
               "Canonical torus from two persistent-cohomology H1 coordinates")
    # Cyclic-coordinate diagnostics.
    dtheta=df.copy()
    dtheta["cos_theta1"]=np.cos(df.theta1_radians)
    dtheta["cos_theta2"]=np.cos(df.theta2_radians)
    plot_torus(dtheta,out/"canonical_torus_theta1.png","cos_theta1",
               "Canonical torus colored by cos(theta1)")
    plot_torus(dtheta,out/"canonical_torus_theta2.png","cos_theta2",
               "Canonical torus colored by cos(theta2)")
    plot_spatial_if_available(df,out/"circular_coordinates_in_physical_space.png")

    formal_diagrams=cdir/"diagrams.npz"
    formal_h1_top=None
    if formal_diagrams.exists():
        try:
            with np.load(formal_diagrams,allow_pickle=False) as a:
                fh1=np.asarray(a["euclidean_H1"],dtype=float)
            _,flife=finite_h1_order(fh1)
            formal_h1_top=[float(x) for x in flife[:5]]
        except Exception:
            formal_h1_top=None

    summary={
        "status":"success",
        "analysis":"figureA4_cohomology_torus_v1",
        "role":"visualization_only",
        "source_seed":int(args.source_seed),
        "selected_clusters":[int(x) for x in labels],
        "candidate_dir":str(cdir),
        "input_pca6_downsampled":str(cdir/"pca6_downsampled.npy"),
        "input_sha256":sha256(cdir/"pca6_downsampled.npy"),
        "n_points":int(len(Xds)),
        "n_cohomology_landmarks":int(n_land),
        "landmark_selection":"deterministic Euclidean greedy farthest-point permutation, start index 0",
        "metric":"euclidean",
        "coeff":int(args.coeff),
        "H1_top_lifetimes_landmark_exact":[float(x) for x in lifetimes[:10]],
        "H1_class_indices":[int(x) for x in class_indices],
        "common_coordinate_threshold":threshold,
        "lifetime_fraction":float(args.lifetime_fraction),
        "decode_diagnostics":decode_diag,
        "landmark_joint_finite_fraction":float(np.mean(np.isfinite(turns_land).all(axis=1))),
        "interpolation_k":int(args.interp_k),
        "phase_confidence_mean":[float(np.mean(confidence[:,0])),float(np.mean(confidence[:,1]))],
        "phase_confidence_p05":[float(np.quantile(confidence[:,0],.05)),float(np.quantile(confidence[:,1],.05))],
        "canonical_torus":{"R":float(args.R),"r":float(args.r)},
        "formal_candidate_euclidean_H1_top_lifetimes":formal_h1_top,
        "elapsed_seconds":float(time.time()-t0),
        "note":(
            "This artifact is a topology-derived visualization only. It does not change "
            "the frozen Schøyen candidate, formal Euclidean topology decision, or FigureA3. "
            "Circular coordinates are identifiable up to sign, phase offset, permutation, "
            "and integral basis changes."
        ),
    }
    write_json(out/"analysis_summary.json",summary)

    # Optional immediate staging; patched prepare_source_data will also do this
    # on every later redraw so make_all_figures cannot silently lose the files.
    if args.aggregate is not None:
        agg=args.aggregate.expanduser().resolve()
        stage=agg/"figure_source_data"/"FigureA4"
        stage.mkdir(parents=True,exist_ok=True)
        shutil.copy2(out/"circular_coordinates.csv",stage/"representative_cohomology_torus.csv")
        shutil.copy2(out/"analysis_summary.json",stage/"representative_cohomology_torus_meta.json")

    print("[done]",out,flush=True)
    print("[H1 top]",[float(x) for x in lifetimes[:5]],flush=True)
    print("[classes]",class_indices.tolist(),"threshold",threshold,flush=True)
    print("[elapsed min]",(time.time()-t0)/60.0,flush=True)


if __name__=="__main__":
    main()
