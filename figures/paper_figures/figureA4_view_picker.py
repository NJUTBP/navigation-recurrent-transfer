#!/usr/bin/env python3
"""Interactive camera picker for FigureA4C.

Rotate the frozen representative_schoyen_umap3 point cloud with the mouse.
Press S to save the current Matplotlib elev/azim into figureA4C_view.json.
The publication renderer reads that file automatically.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def equal_cube_limits(frame: pd.DataFrame, pad: float = 0.04):
    xyz = frame[["umap1", "umap2", "umap3"]].to_numpy(float)
    xyz = xyz[np.all(np.isfinite(xyz), axis=1)]
    lo = np.nanmin(xyz, axis=0)
    hi = np.nanmax(xyz, axis=0)
    center = (lo + hi) / 2.0
    radius = max(0.5 * float(np.max(hi - lo)) * (1.0 + pad), 1e-6)
    return tuple((float(c - radius), float(c + radius)) for c in center)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggregate", type=Path, required=True, help="Path to aggregate_full")
    ap.add_argument("--elev", type=float, default=None)
    ap.add_argument("--azim", type=float, default=None)
    ap.add_argument("--point-size", type=float, default=5.0)
    ap.add_argument("--alpha", type=float, default=0.72)
    ap.add_argument("--pad", type=float, default=0.04)
    ap.add_argument(
        "--layout", choices=["single", "quad"], default="single",
        help="Final FigureA4C layout to save. 'single' gives the chosen view maximum space.",
    )
    args = ap.parse_args()

    aggregate = args.aggregate.expanduser().resolve()
    data_dir = aggregate / "figure_source_data" / "FigureA4"
    csv_path = data_dir / "representative_schoyen_umap3.csv"
    view_path = data_dir / "figureA4C_view.json"

    if not csv_path.exists():
        raise SystemExit(f"Missing: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"umap1", "umap2", "umap3", "pc1"}
    if not required.issubset(frame.columns):
        raise SystemExit(
            f"{csv_path} lacks columns: {sorted(required - set(frame.columns))}"
        )

    elev, azim = 20.0, -60.0
    if view_path.exists():
        try:
            old = json.loads(view_path.read_text(encoding="utf-8"))
            views = old.get("views", [])
            if views:
                v = views[0]
                if isinstance(v, dict):
                    elev, azim = float(v["elev"]), float(v["azim"])
                else:
                    elev, azim = float(v[0]), float(v[1])
        except Exception:
            pass

    if args.elev is not None:
        elev = args.elev
    if args.azim is not None:
        azim = args.azim

    vals = frame["pc1"].to_numpy(float)
    xyz = frame[["umap1", "umap2", "umap3"]].to_numpy(float)
    limits = equal_cube_limits(frame, pad=args.pad)

    fig = plt.figure(figsize=(8.6, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=vals, cmap="viridis",
        s=args.point_size, alpha=args.alpha,
        linewidth=0, depthshade=False,
    )
    ax.set_xlim3d(*limits[0])
    ax.set_ylim3d(*limits[1])
    ax.set_zlim3d(*limits[2])
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()
    ax.view_init(elev=elev, azim=azim)

    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=.04, pad=.02)
    cb.set_label("PC1")

    status = fig.text(
        .5, .965, "", ha="center", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=.3", fc="white", ec="0.8", alpha=.9),
    )
    fig.text(
        .5, .02,
        "Mouse-drag to rotate.  S = save camera.  P = print camera.  Q/Esc = close.",
        ha="center", va="bottom", fontsize=9,
    )

    def camera_text():
        return f"elev={ax.elev:.2f}°, azim={ax.azim:.2f}°"

    def refresh_status():
        status.set_text(camera_text())
        try:
            fig.canvas.manager.set_window_title(
                "FigureA4C camera | " + camera_text()
            )
        except Exception:
            pass
        fig.canvas.draw_idle()

    def save_view():
        payload = {
            "layout": args.layout,
            "views": [{"elev": float(ax.elev), "azim": float(ax.azim)}],
            "source": "manual_matplotlib_rotation",
            "embedding": "representative_schoyen_umap3.csv",
            "note": (
                "Camera choice only; embedding and topology statistics are unchanged."
            ),
        }
        view_path.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[saved] {view_path}")
        print(f"[camera] {camera_text()}")
        refresh_status()

    def on_release(event):
        if event.inaxes is ax:
            print("[camera]", camera_text())
            refresh_status()

    def on_key(event):
        key = (event.key or "").lower()
        if key == "s":
            save_view()
        elif key == "p":
            print("[camera]", camera_text())
            refresh_status()
        elif key in {"q", "escape"}:
            plt.close(fig)

    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    refresh_status()

    backend = str(plt.get_backend())
    if "agg" in backend.lower():
        print(
            f"[warning] Matplotlib backend is {backend!r}, which is non-interactive."
        )
        print(
            "[warning] Re-run with MPLBACKEND=TkAgg (or QtAgg) if no window appears."
        )

    print(f"[data] {csv_path}")
    print(f"[points] {len(frame)}")
    print(
        "[controls] rotate with mouse; press S when the torus view is satisfactory"
    )
    print("[initial camera]", camera_text())
    plt.show()


if __name__ == "__main__":
    main()
