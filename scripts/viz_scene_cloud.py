"""Mac-side: visualize a dumped Gaussian cloud + run Test A (ground shape check).

Input: <scene>_cloud.npz from dump_scene_cloud.py (scp'd from Marlowe).

Produces, per scene:
  1. <scene>_cloud3d.html     — interactive 3D view (rotate/zoom in the browser):
                                 cloud colored by RGB or semantic class, the real
                                 trajectory as a white line, goal marker.
  2. <scene>_ground_testA.png — THE Test A plot: ground height along the real
                                 trajectory, three lines:
                                   gray dashed = flat assumption (z=0)
                                   yellow      = TRUE profile (camera z - 0.6; robot was there)
                                   blue        = Gaussian-derived local ground
                                                 (low percentile of splat z within 0.75 m)
                                 If blue tracks yellow, the Gaussians capture the
                                 ground shape -> we can use them for footprint
                                 height ANYWHERE, not just on the driven line.
  3. printed stats: |gaussian_ground - true_ground| median / p90.

Usage:
    python scripts/viz_scene_cloud.py [--color rgb|semantic|height|time]
        [--min_opacity 0.05] outputs/scene_clouds/<scene>_cloud.npz [...]

--color semantic paints each gaussian with its fused SAM3 class (see
outputs/legend.png); height/time use a viridis colormap. --min_opacity hides
near-transparent floater gaussians that clutter the point view.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.palette import CLASS_COLORS_255

COLOR_MODE = "rgb"
MIN_OPACITY = 0.05


def local_ground_from_cloud(points: np.ndarray, xy: np.ndarray,
                            radius: float = 0.75, pct: float = 5.0):
    """Ground height at each query (x, y): the `pct`th percentile of z among
    cloud points within `radius` meters horizontally. Low percentile ~= ground
    (splats above it are vegetation/obstacles); not min, which hits floaters."""
    from scipy.spatial import cKDTree
    tree = cKDTree(points[:, :2])
    out = np.full(len(xy), np.nan)
    for i, q in enumerate(xy):
        idx = tree.query_ball_point(q, radius)
        if len(idx) >= 20:
            out[i] = np.percentile(points[idx, 2], pct)
    return out


def run(npz_path: Path):
    d = np.load(npz_path)
    pts, labels, colors = d["points"], d["labels"], d["colors"]
    traj = d["traj_positions"]
    true_ground = d["traj_cam_z"] - float(d["camera_height_m"])
    scene = npz_path.stem.replace("_cloud", "")
    out_dir = npz_path.parent

    # ---- 1. interactive 3D + top-down panel (plotly) ----
    # Filter floaters by opacity when available (they clutter the point view but
    # are nearly invisible in actual renders), then subsample for the browser.
    opac = d["opacities"] if "opacities" in d else np.ones(len(pts))
    sizes = d["sizes"] if "sizes" in d else np.zeros(len(pts))
    times = d["times"] if "times" in d else np.full(len(pts), -1.0)
    keep = opac >= MIN_OPACITY
    print(f"  opacity filter >= {MIN_OPACITY}: kept {keep.mean():.0%} of points")
    kpts, kcol, klab = pts[keep], colors[keep], labels[keep]
    kop, ksz, ktm = opac[keep], sizes[keep], times[keep]
    sub = np.random.default_rng(0).choice(len(kpts), min(len(kpts), 120_000), replace=False)

    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        if COLOR_MODE == "semantic":
            pal = CLASS_COLORS_255
            cvals = [f"rgb({pal[l][0]},{pal[l][1]},{pal[l][2]})" if l >= 0
                     else "rgb(80,80,80)" for l in klab[sub]]
        elif COLOR_MODE == "height":
            cvals = kpts[sub, 2]
        elif COLOR_MODE == "time":
            cvals = ktm[sub]
        else:  # rgb
            cvals = [f"rgb({r},{g},{b})" for r, g, b in kcol[sub]]
        # marker size from gaussian scale: 1px (tiny) .. 4px (big blobs)
        msize = 1 + 3 * np.clip(ksz[sub] / max(np.percentile(ksz[sub], 95), 1e-6), 0, 1)
        marker = dict(size=msize, color=cvals, opacity=0.8)
        if COLOR_MODE in ("height", "time"):
            marker["colorscale"] = "Viridis"; marker["showscale"] = True

        fig = make_subplots(cols=2, column_widths=[0.7, 0.3],
                            specs=[[{"type": "scene"}, {"type": "xy"}]],
                            subplot_titles=(f"{scene} — 3D (color: {COLOR_MODE})",
                                            "top-down map"))
        fig.add_trace(go.Scatter3d(x=kpts[sub, 0], y=kpts[sub, 1], z=kpts[sub, 2],
                                   mode="markers", marker=marker, name="gaussians"),
                      row=1, col=1)
        fig.add_trace(go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=true_ground,
                                   mode="lines", line=dict(color="white", width=6),
                                   name="real trajectory"), row=1, col=1)
        # top-down: same subsample projected to (x, y)
        td_marker = dict(size=2, color=cvals)
        if COLOR_MODE in ("height", "time"):
            td_marker["colorscale"] = "Viridis"
        fig.add_trace(go.Scattergl(x=kpts[sub, 0], y=kpts[sub, 1], mode="markers",
                                   marker=td_marker, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scattergl(x=traj[:, 0], y=traj[:, 1], mode="lines",
                                   line=dict(color="white", width=2),
                                   showlegend=False), row=1, col=2)
        fig.add_trace(go.Scattergl(x=[traj[-1, 0]], y=[traj[-1, 1]], mode="markers",
                                   marker=dict(size=10, color="lime", symbol="star"),
                                   showlegend=False), row=1, col=2)
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=2)
        fig.update_layout(scene=dict(aspectmode="data", bgcolor="black"),
                          template="plotly_dark", title=scene)
        html = out_dir / f"{scene}_cloud3d.html"
        fig.write_html(str(html))
        print(f"  {html}  <- open in browser; left: rotate/zoom, right: floor plan")
    except ImportError:
        print("  (plotly not installed: pip install plotly for the interactive 3D; "
              "skipping to Test A)")

    # ---- 2. Test A ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gauss_ground = local_ground_from_cloud(pts, traj[:, :2])
    x = np.arange(len(traj))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axhline(0, color="gray", ls="--", lw=1, label="flat assumption (z=0)")
    ax.plot(x, true_ground, color="goldenrod", lw=2, label="TRUE ground (trajectory-derived)")
    ax.plot(x, gauss_ground, color="tab:blue", lw=1.5, label="Gaussian-derived local ground")
    ax.set_xlabel("frame along real trajectory"); ax.set_ylabel("ground height (m)")
    ax.set_title(f"Test A — can the Gaussians tell us the ground shape?  [{scene}]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out_dir / f"{scene}_ground_testA.png"
    fig.savefig(png, dpi=120); plt.close(fig)

    ok = ~np.isnan(gauss_ground)
    err = np.abs(gauss_ground[ok] - true_ground[ok])
    print(f"  {png}")
    print(f"  Test A [{scene}]: |gaussian - true| median {np.median(err)*100:.1f} cm, "
          f"p90 {np.percentile(err, 90)*100:.1f} cm  "
          f"(coverage {ok.mean():.0%} of trajectory)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--color" in argv:
        i = argv.index("--color"); COLOR_MODE = argv[i + 1]; del argv[i:i + 2]
    if "--min_opacity" in argv:
        i = argv.index("--min_opacity"); MIN_OPACITY = float(argv[i + 1]); del argv[i:i + 2]
    for p in argv:
        print(f"=== {p} ===")
        run(Path(p))
