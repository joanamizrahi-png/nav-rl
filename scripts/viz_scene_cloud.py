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

Usage: python scripts/viz_scene_cloud.py outputs/scene_clouds/rugd_trail_00_cloud.npz [...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.palette import CLASS_COLORS_255


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

    # ---- 1. interactive 3D (plotly; graceful fallback to matplotlib PNG) ----
    sub = np.random.default_rng(0).choice(len(pts), min(len(pts), 120_000), replace=False)
    try:
        import plotly.graph_objects as go
        rgb = [f"rgb({r},{g},{b})" for r, g, b in colors[sub]]
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=pts[sub, 0], y=pts[sub, 1], z=pts[sub, 2],
                                   mode="markers", marker=dict(size=1, color=rgb),
                                   name="gaussians"))
        fig.add_trace(go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=true_ground,
                                   mode="lines", line=dict(color="white", width=6),
                                   name="real trajectory (on true ground)"))
        fig.update_layout(scene=dict(aspectmode="data", bgcolor="black"),
                          template="plotly_dark", title=scene)
        html = out_dir / f"{scene}_cloud3d.html"
        fig.write_html(str(html))
        print(f"  {html}  <- open in browser, rotate with mouse")
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
    for p in sys.argv[1:]:
        print(f"=== {p} ===")
        run(Path(p))
