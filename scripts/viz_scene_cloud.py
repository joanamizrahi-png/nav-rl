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
HIDE_SKY = False
# Episode overlay (--episodes N): draw N sampled J-spec spawns/goals in the
# cloud. Defaults match the J-50 arms (cone 50, goals 5-10 m, lateral 0.4 m).
EPISODES = 0
CONE_DEG = 50.0
GOAL_DIST = (5.0, 10.0)
SPAWN_LAT = 0.4
SIZE_BY_SCALE = False
GROUND_PCT = 5.0


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
    # FRAME FIX (2026-09-01): dump_scene_cloud writes `points` through cal.A,
    # which carries the 08-13 y-mirror (F = diag(1,-1,1)), but stores
    # traj_positions RAW from the poses npz. Un-mirrored, every 3D view drew
    # the trajectory reflected across the corridor. Apply F here so path and
    # cloud share one frame (also what NavCalibration.positions returns).
    traj = np.asarray(d["traj_positions"], dtype=float) * np.array([1.0, -1.0, 1.0])
    true_ground = d["traj_cam_z"] - float(d["camera_height_m"])
    scene = npz_path.stem.replace("_cloud", "")
    # Folder layout (option A): scene_clouds/{clouds,splats,html,ground}/.
    # Accept npz paths inside clouds/ or flat (older layout).
    root = npz_path.parent.parent if npz_path.parent.name == "clouds" else npz_path.parent
    html_dir = root / "html"; ground_dir = root / "ground"
    html_dir.mkdir(exist_ok=True); ground_dir.mkdir(exist_ok=True)

    # ---- 1. interactive 3D + top-down panel (plotly) ----
    # Filter floaters by opacity when available (they clutter the point view but
    # are nearly invisible in actual renders), then subsample for the browser.
    opac = d["opacities"] if "opacities" in d else np.ones(len(pts))
    sizes = d["sizes"] if "sizes" in d else np.zeros(len(pts))
    times = d["times"] if "times" in d else np.full(len(pts), -1.0)
    keep = opac >= MIN_OPACITY
    if HIDE_SKY:                      # optional, off by default (--hide_sky)
        keep &= ~np.isin(labels, (0, 1))
    print(f"  filters kept {keep.mean():.0%} of points")
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
        # Constant small markers by default. The size-by-scale mapping
        # (--size_by_scale) inflates far/uncertain gaussians into huge blobs
        # that bury the scene ("white avalanche") — root cause of the earlier
        # all-white view, so it's opt-in only. Sizes as plain list, not numpy
        # (plotly's binary encoding can silently kill the WebGL trace).
        if SIZE_BY_SCALE:
            msize = (1.5 + 1.5 * np.clip(ksz[sub] / max(np.percentile(ksz[sub], 95), 1e-6), 0, 1))
            size_arg = [round(float(s), 2) for s in msize]
        else:
            size_arg = 1.5
        marker = dict(size=size_arg, color=cvals, opacity=0.8)
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
        # EPISODE OVERLAY (2026-09-01, her ask): the J-50 training distribution
        # drawn inside the world it samples from — spawns on the path (after
        # yaw/lateral jitter) and their goals in the +-CONE_DEG/2 tangent cone.
        if EPISODES > 0 and len(traj) > 12:
            r = np.random.default_rng(0)
            sp, gl = [], []
            for _ in range(EPISODES):
                f = int(r.integers(10, max(len(traj) - 6, 11)))
                p = traj[f, :2].copy()
                fw = traj[min(f + 1, len(traj) - 1), :2] - traj[max(f - 1, 0), :2]
                base = float(np.arctan2(fw[1], fw[0]))
                p = p + r.uniform(-1, 1) * SPAWN_LAT * np.array(
                    [-np.sin(base), np.cos(base)])
                th = base + np.deg2rad(r.uniform(-1, 1) * CONE_DEG / 2.0)
                dd = r.uniform(*GOAL_DIST)
                sp.append(p)
                gl.append(p + dd * np.array([np.cos(th), np.sin(th)]))
            sp, gl = np.array(sp), np.array(gl)
            zf = float(np.median(true_ground))
            for rc, kw in ((1, dict(row=1, col=1)), (2, dict(row=1, col=2))):
                S3 = go.Scatter3d if rc == 1 else go.Scattergl
                common = dict(x=sp[:, 0], y=sp[:, 1], mode="markers",
                              marker=dict(size=4 if rc == 1 else 7,
                                          color="orange"),
                              name="spawns", showlegend=(rc == 1))
                if rc == 1:
                    common["z"] = np.full(len(sp), zf)
                fig.add_trace(S3(**common), **kw)
                commong = dict(x=gl[:, 0], y=gl[:, 1], mode="markers",
                               marker=dict(size=4 if rc == 1 else 7,
                                           color="lime", symbol="x"),
                               name="goals", showlegend=(rc == 1))
                if rc == 1:
                    commong["z"] = np.full(len(gl), zf)
                fig.add_trace(S3(**commong), **kw)
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=2)
        fig.update_layout(scene=dict(aspectmode="data", bgcolor="black"),
                          template="plotly_dark", title=scene)
        html = html_dir / f"{scene}_cloud3d_{COLOR_MODE}.html"
        fig.write_html(str(html))
        print(f"  {html}  <- open in browser; left: rotate/zoom, right: floor plan")
    except ImportError:
        print("  (plotly not installed: pip install plotly for the interactive 3D; "
              "skipping to Test A)")

    # ---- 2. Test A ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gauss_ground = local_ground_from_cloud(pts, traj[:, :2], pct=GROUND_PCT)
    x = np.arange(len(traj))
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axhline(0, color="gray", ls="--", lw=1, label="flat assumption (z=0)")
    ax.plot(x, true_ground, color="goldenrod", lw=2, label="TRUE ground (trajectory-derived)")
    ax.plot(x, gauss_ground, color="tab:blue", lw=1.5,
            label=f"Gaussian-derived local ground (p{GROUND_PCT:g})")
    ax.set_xlabel("frame along real trajectory"); ax.set_ylabel("ground height (m)")
    ax.set_title(f"Test A — can the Gaussians tell us the ground shape?  [{scene}]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    png = ground_dir / f"{scene}_ground_testA_p{GROUND_PCT:g}.png"
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
    if "--hide_sky" in argv:
        HIDE_SKY = True; argv.remove("--hide_sky")
    if "--size_by_scale" in argv:
        SIZE_BY_SCALE = True; argv.remove("--size_by_scale")
    if "--pct" in argv:
        i = argv.index("--pct"); GROUND_PCT = float(argv[i + 1]); del argv[i:i + 2]
    if "--episodes" in argv:
        i = argv.index("--episodes"); EPISODES = int(argv[i + 1]); del argv[i:i + 2]
    if "--cone_deg" in argv:
        i = argv.index("--cone_deg"); CONE_DEG = float(argv[i + 1]); del argv[i:i + 2]
    for p in argv:
        print(f"=== {p} ===")
        run(Path(p))
