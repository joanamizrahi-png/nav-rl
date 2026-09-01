"""Find real goal coordinates of a chosen terrain class, off the recorded path.

Every terrain experiment we want to run — "put the goal on grass and see if the
policy stops at the sidewalk edge", "aim the footprint at a real obstacle and
check the gate keeps it" — has been blocked on somebody hand-reading an x,y off
a topdown plot. The scene cloud already knows: it carries SAM3 labels fused
onto the gaussians, so the grass IS locatable, in metres, in nav frame.

Height band matters. Obstacles live at body height (0.15-1.2 m, the same band
SceneEnv._load_obstacle_points uses); terrain lives on the ground plane
(below 0.15 m). Asking for grass in the obstacle band returns nothing.

Usage (login node, no GPU):
    python scripts/pick_goal.py --scene gnd_AUw360 --class_name grass
    python scripts/pick_goal.py --scene gnd_AUw360 --class_name obstacle \
        --z_min 0.15 --z_max 1.2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

V14_NAMES = ["void", "sky", "trail", "grass", "rough", "water", "sidewalk",
             "road", "pavement", "stairs", "obstacle", "vegetation", "person",
             "vehicle"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--class_name", default="grass")
    ap.add_argument("--clouds_dir",
                    default="/scratch/m000204-pm06b/joana/outputs/scene_clouds/clouds")
    ap.add_argument("--poses_dir", default="/scratch/m000204-pm06b/joana/outputs/poses")
    ap.add_argument("--z_min", type=float, default=-0.5,
                    help="ground plane by default; use 0.15 for body-height obstacles")
    ap.add_argument("--z_max", type=float, default=0.15)
    ap.add_argument("--min_off", type=float, default=1.5,
                    help="metres from the recorded path: near enough to reach")
    ap.add_argument("--max_off", type=float, default=6.0,
                    help="...far enough that reaching it means leaving the path")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--grid", type=float, default=1.0,
                    help="snap candidates to this grid so they are not all the "
                         "same patch of grass")
    ap.add_argument("--png", default="")
    args = ap.parse_args()

    if args.class_name not in V14_NAMES:
        raise SystemExit(f"class must be one of {V14_NAMES}")
    cid = V14_NAMES.index(args.class_name)

    cloud = np.load(Path(args.clouds_dir) / f"{args.scene}_cloud.npz")
    pts, labs = cloud["points"], cloud["labels"].astype(int)
    poses = np.load(Path(args.poses_dir) / f"{args.scene}_poses.npz")
    key = "positions" if "positions" in poses else list(poses.keys())[0]
    path = np.asarray(poses[key], dtype=float)
    path = path[:, :3, 3][:, :2] if path.ndim == 3 else path[:, :2]

    band = (pts[:, 2] > args.z_min) & (pts[:, 2] < args.z_max)
    sel = (labs == cid) & band
    print(f"{args.scene}: {len(pts)} points, {int((labs == cid).sum())} "
          f"labelled {args.class_name}, {int(sel.sum())} of those in "
          f"z=[{args.z_min},{args.z_max}]")
    if not sel.any():
        raise SystemExit(f"no {args.class_name} points in that height band — "
                         f"terrain lives below 0.15 m, obstacles above it")

    xy = pts[sel][:, :2]
    # distance from each candidate to the recorded walk
    d = np.sqrt(((xy[:, None, :] - path[None, :, :]) ** 2).sum(-1))
    dmin = d.min(1)
    near = (dmin >= args.min_off) & (dmin <= args.max_off)
    print(f"  {int(near.sum())} within {args.min_off}-{args.max_off} m of the path")
    # Always show where this class actually LIVES relative to the walk. On
    # gnd_AUw360 every grass point turned out to be under 1.5 m from the path —
    # the boundary is at the walk's edge, which the bare "none in band" error
    # hid. The distribution is the finding, not the failure.
    print(f"  offset from path (m):  min {dmin.min():.2f}  p10 "
          f"{np.percentile(dmin, 10):.2f}  p50 {np.percentile(dmin, 50):.2f}  "
          f"p90 {np.percentile(dmin, 90):.2f}  max {dmin.max():.2f}")
    if not near.any():
        lo, hi = float(np.percentile(dmin, 25)), float(np.percentile(dmin, 75))
        raise SystemExit(f"none in that offset band — this class sits at "
                         f"{lo:.2f}-{hi:.2f} m; try --min_off {lo:.1f} "
                         f"--max_off {hi:.1f}")

    xy, dmin = xy[near], dmin[near]
    nearest_frame = d[near].argmin(1)

    # snap to a grid so the shortlist spans different patches, not one blob
    keyed = {}
    for i in range(len(xy)):
        k = (round(xy[i, 0] / args.grid), round(xy[i, 1] / args.grid))
        if k not in keyed or dmin[i] < dmin[keyed[k]]:
            keyed[k] = i
    idx = sorted(keyed.values(), key=lambda i: dmin[i])[:args.n]

    print(f"\n  {'GOAL_XY':>16}   off-path   nearest recorded frame")
    for i in idx:
        print(f"  {xy[i, 0]:7.2f},{xy[i, 1]:<7.2f}   {dmin[i]:5.2f} m   "
              f"frame {int(nearest_frame[i])}")
    print("\n  Use as:  GOALXY=\"x,y\"  (training)  /  --goal_xy x,y  (eval, "
          "check_rewards)")
    print("  Spawn near the matching frame so the walk actually approaches it.")

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gnd = pts[band]
        gl = labs[band]
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.scatter(gnd[::20, 0], gnd[::20, 1], s=1, c="0.85", linewidths=0)
        m = gl == cid
        ax.scatter(gnd[m][::20, 0], gnd[m][::20, 1], s=1, c="tab:green",
                   linewidths=0, label=args.class_name)
        ax.plot(path[:, 0], path[:, 1], "-", c="tab:blue", lw=2, label="recorded walk")
        for i in idx:
            ax.plot(xy[i, 0], xy[i, 1], "*", c="tab:red", ms=14)
            ax.annotate(f"{xy[i, 0]:.1f},{xy[i, 1]:.1f}", (xy[i, 0], xy[i, 1]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_aspect("equal")
        ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{args.scene}: {args.class_name} goal candidates "
                     f"({args.min_off}-{args.max_off} m off path)")
        fig.savefig(args.png, dpi=130, bbox_inches="tight")
        print(f"==> {args.png}")


if __name__ == "__main__":
    main()
