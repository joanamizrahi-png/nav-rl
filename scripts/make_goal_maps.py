"""Goal-placement maps: verify and choose per-scene goals BEFORE training on them.

Why: goals so far were "trajectory frame 30" — an arbitrary moment, not a chosen
place. A good goal must be (a) on walkable terrain, (b) in well-reconstructed
space, (c) at a controlled distance from the spawn region. This script scores
every trajectory frame on those criteria, recommends a goal per scene, and
draws a top-down map so a human can approve/override each one.

Scoring, per candidate frame j (position p_j):
  density     = points within 0.75 m of p_j (normalized by the scene median)
                -> "is this spot well reconstructed?"
  walkability = among near-ground points (z < 0.4 m) within 0.75 m, the fraction
                whose semantic class has traversability score >= 0.5
                -> "is this spot terrain a Go2 can stand on?"
  distance    = path length from frame 0 to j (want TARGET_DIST_M, tolerance band)

Recommended goal = the frame inside the distance band maximizing
walkability (ties -> density). Falls back to nearest-band frame if none qualify.

Outputs:
  outputs/goal_maps/<scene>_goalmap.png   top-down: cloud (semantic colors),
      trajectory (white), OLD goal frame 30 (red X), recommended goal (green star),
      spawn range (white dots)
  outputs/goal_maps/goals.json            scene -> {goal_xy, goal_frame, spawn_hi,
      walkability, density} — consumed by training later
  printed table for quick review

Usage (Mac): python scripts/make_goal_maps.py [--target_dist 3.5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.eval.palette import CLASS_COLORS_255
from src.eval.traversability import load_traversability

RADIUS = 0.75
GROUND_Z_MAX = 0.4
OLD_GOAL_FRAME = 30


def score_scene(cloud_npz: Path, trav: np.ndarray, target: float, band: float):
    from scipy.spatial import cKDTree
    d = np.load(cloud_npz)
    pts, labels = d["points"], d["labels"]
    traj = d["traj_positions"]
    tree = cKDTree(pts[:, :2])

    path_len = np.concatenate([[0], np.cumsum(
        np.linalg.norm(np.diff(traj[:, :2], axis=0), axis=1))])

    rows = []
    counts = []
    for j in range(len(traj)):
        idx = tree.query_ball_point(traj[j, :2], RADIUS)
        counts.append(len(idx))
        near = [i for i in idx if pts[i, 2] < GROUND_Z_MAX]
        if len(near) >= 20:
            walk = float(np.mean(trav[np.clip(labels[near], 0, len(trav) - 1)] >= 0.5))
        else:
            walk = 0.0
        rows.append({"frame": j, "dist": float(path_len[j]),
                     "count": len(idx), "walk": walk})
    med = max(float(np.median(counts)), 1.0)
    for r in rows:
        r["density"] = r["count"] / med

    in_band = [r for r in rows
               if abs(r["dist"] - target) <= band and r["density"] >= 0.5]
    pool = in_band or sorted(rows, key=lambda r: abs(r["dist"] - target))[:5]
    best = max(pool, key=lambda r: (round(r["walk"], 2), r["density"]))
    return d, rows, best


def draw_map(d, rows, best, scene: str, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, labels, traj = d["points"], d["labels"], d["traj_positions"]
    sub = np.random.default_rng(0).choice(len(pts), min(len(pts), 60_000), replace=False)
    cols = CLASS_COLORS_255[np.clip(labels[sub], 0, len(CLASS_COLORS_255) - 1)] / 255.0

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(pts[sub, 0], pts[sub, 1], s=0.5, c=cols, alpha=0.5, linewidths=0)
    ax.plot(traj[:, 0], traj[:, 1], color="white", lw=2, label="real trajectory")
    og = traj[min(OLD_GOAL_FRAME, len(traj) - 1)]
    ax.scatter([og[0]], [og[1]], marker="x", s=120, color="red",
               label=f"old goal (frame {OLD_GOAL_FRAME})")
    bg = traj[best["frame"]]
    ax.scatter([bg[0]], [bg[1]], marker="*", s=250, color="lime",
               label=f"recommended (frame {best['frame']}, "
                     f"walk {best['walk']:.0%}, {best['dist']:.1f} m)")
    spawn_hi = max(1, best["frame"] - 5)
    ax.scatter(traj[:spawn_hi:4, 0], traj[:spawn_hi:4, 1], s=12, color="white",
               edgecolors="black", linewidths=0.3, label="spawn range")
    lo = traj[:, :2].min(0) - 4.0; hi = traj[:, :2].max(0) + 4.0
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal"); ax.set_facecolor("black")
    ax.set_title(f"{scene} — goal placement")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_dist", type=float, default=3.5)
    ap.add_argument("--band", type=float, default=1.0)
    args = ap.parse_args()

    trav = load_traversability()
    out_dir = REPO_ROOT / "outputs/goal_maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    goals = {}
    clouds = sorted((REPO_ROOT / "outputs/scene_clouds/clouds").glob("*_cloud.npz"))
    print(f"{'scene':24s} {'frame':>5s} {'dist':>6s} {'walk':>6s} {'density':>7s}")
    for c in clouds:
        scene = c.stem.replace("_cloud", "")
        d, rows, best = score_scene(c, trav, args.target_dist, args.band)
        draw_map(d, rows, best, scene, out_dir / f"{scene}_goalmap.png")
        goals[scene] = {"goal_frame": int(best["frame"]),
                        "goal_xy": [float(v) for v in d["traj_positions"][best["frame"], :2]],
                        "spawn_hi": max(1, int(best["frame"]) - 5),
                        "walkability": round(best["walk"], 3),
                        "density": round(best["density"], 3),
                        "dist_m": round(best["dist"], 2)}
        flag = "  <-- LOW WALKABILITY, review!" if best["walk"] < 0.6 else ""
        print(f"{scene:24s} {best['frame']:5d} {best['dist']:6.1f} "
              f"{best['walk']:6.0%} {best['density']:7.2f}{flag}")
    with open(out_dir / "goals.json", "w") as f:
        json.dump(goals, f, indent=2)
    print(f"\nwrote {out_dir}/goals.json and {len(goals)} maps")


if __name__ == "__main__":
    main()
