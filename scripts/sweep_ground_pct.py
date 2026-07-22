"""Sweep the ground-estimator percentile over many scenes (Mac-only, no GPU).

For each scene cloud: build the neighbor structure ONCE, then evaluate the
Gaussian-ground estimate at every percentile in the sweep against the
trajectory-derived true ground. Neighbor search dominates the cost, so all
percentiles together cost barely more than one.

Outputs:
  outputs/scene_clouds/ground_pct_sweep.csv    — scene x pct median/p90 errors
  outputs/scene_clouds/ground_pct_sweep.png    — median error vs pct, one thin
                                                 line per scene + bold mean.
                                                 The minimum of the bold line is
                                                 the percentile to adopt.

Usage: python scripts/sweep_ground_pct.py outputs/scene_clouds/*_cloud.npz
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

PCTS = [0, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]   # override with --pcts "15,20,30,40,50"
RADIUS = 0.75
MAX_TREE_POINTS = 150_000     # subsample for speed; negligible accuracy cost


def sweep_scene(npz_path: Path, rng):
    from scipy.spatial import cKDTree
    d = np.load(npz_path)
    pts, traj = d["points"], d["traj_positions"]
    true_ground = d["traj_cam_z"] - float(d["camera_height_m"])
    if len(pts) > MAX_TREE_POINTS:
        pts = pts[rng.choice(len(pts), MAX_TREE_POINTS, replace=False)]
    tree = cKDTree(pts[:, :2])
    neigh = tree.query_ball_point(traj[:, :2], RADIUS)      # once per scene
    rows = []
    for pct in PCTS:
        est = np.full(len(traj), np.nan)
        for i, idx in enumerate(neigh):
            if len(idx) >= 20:
                est[i] = np.percentile(pts[idx, 2], pct)
        ok = ~np.isnan(est)
        err = np.abs(est[ok] - true_ground[ok])
        rows.append((pct, float(np.median(err)), float(np.percentile(err, 90)),
                     float(ok.mean())))
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    global PCTS
    argv = sys.argv[1:]
    if "--pcts" in argv:
        i = argv.index("--pcts")
        PCTS = [float(v) for v in argv[i + 1].split(",")]
        del argv[i:i + 2]
    paths = [Path(p) for p in argv]
    rng = np.random.default_rng(0)
    all_rows = []
    per_scene = {}
    for p in paths:
        scene = p.stem.replace("_cloud", "")
        rows = sweep_scene(p, rng)
        per_scene[scene] = rows
        for pct, med, p90, cov in rows:
            all_rows.append({"scene": scene, "pct": pct, "median_cm": med * 100,
                             "p90_cm": p90 * 100, "coverage": cov})
        best = min(rows, key=lambda r: r[1])
        print(f"{scene:24s} best pct={best[0]:>4} -> median {best[1]*100:.1f} cm")

    base = paths[0].parent
    out_dir = (base.parent / "ground") if base.name == "clouds" else base
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "ground_pct_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    med_matrix = []
    for scene, rows in per_scene.items():
        meds = [r[1] * 100 for r in rows]
        med_matrix.append(meds)
        ax.plot(PCTS, meds, color="tab:blue", alpha=0.25, lw=1)
    mean_curve = np.mean(med_matrix, axis=0)
    ax.plot(PCTS, mean_curve, color="tab:red", lw=2.5, marker="o",
            label="mean over scenes")
    best_pct = PCTS[int(np.argmin(mean_curve))]
    ax.axvline(best_pct, color="tab:red", ls="--", lw=1,
               label=f"best mean: pct={best_pct}")
    ax.set_xlabel("ground percentile"); ax.set_ylabel("median |error| (cm)")
    ax.set_title(f"Ground-estimator percentile sweep ({len(per_scene)} scenes, "
                 f"radius {RADIUS} m)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "ground_pct_sweep.png", dpi=120)
    print(f"\nwrote {out_dir/'ground_pct_sweep.csv'} and ground_pct_sweep.png")
    print(f"mean curve (cm): " +
          ", ".join(f"p{p:g}={v:.1f}" for p, v in zip(PCTS, mean_curve)))


if __name__ == "__main__":
    main()
