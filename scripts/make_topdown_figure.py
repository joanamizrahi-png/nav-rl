"""Top-down avoidance figure: policy paths over the recorded traversable ribbon.

Why: eval numbers prove terrain avoidance statistically (collision steps), but
the advisor-facing claim "the policy avoids non-traversable terrain and reaches
the goal" needs one picture. This draws, in metric nav frame:
  - the recorded robot path as a ribbon (the known-traversable corridor),
  - every eval episode's trajectory (from metrics.json "traj", green=success,
    orange=failure), red dots where the collision penalty fired,
  - the straight spawn->goal chord (dashed) so detours are visible against
    the shortcut the policy chose NOT to take,
  - spawn circles / goal stars, frame-index ticks along the ribbon.
It also prints the highest-curvature windows of the recorded path — use those
to pin spawn_max_frame / goal_frame for a designed detour eval.

Runs locally (matplotlib only):
    python scripts/make_topdown_figure.py \
        --metrics ~/Documents/inference_runs/.../metrics.json \
        --poses_npz ~/Documents/inference_runs/rugd_trail_00_poses.npz \
        --out TOPDOWN_gated_noBC_goal70.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.env.real_calibrated import NavCalibration


# The recorded path is a LINE (robot center); the ribbon width is an assumption
# for display, not a measurement. Default = goal_radius (0.75 m). The honest
# avoidance evidence is the trajectories-vs-chords contrast and the collision
# dots, not the ribbon edge — shrink via --ribbon_halfwidth for a stricter look
# (Go2 half-width ~0.16 m).
DEFAULT_RIBBON_HALF_WIDTH_M = 0.75


def ribbon_polygon(p: np.ndarray, half_w: float) -> np.ndarray:
    d = np.gradient(p, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    return np.vstack([p + half_w * n, (p - half_w * n)[::-1]])


def bend_windows(p: np.ndarray, win: int = 10, top: int = 3) -> list:
    """Windows of the recorded path with the most heading change (the bends)."""
    d = np.gradient(p, axis=0)
    heading = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
    turn = np.abs(np.diff(heading))
    scores = [(float(np.degrees(turn[i:i + win].sum())), i, i + win)
              for i in range(0, len(turn) - win)]
    scores.sort(reverse=True)
    picked, used = [], set()
    for deg, lo, hi in scores:
        if any(lo < u_hi and hi > u_lo for u_lo, u_hi in used):
            continue
        picked.append((deg, lo, hi))
        used.add((lo, hi))
        if len(picked) == top:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--poses_npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("--ribbon_halfwidth", type=float,
                    default=DEFAULT_RIBBON_HALF_WIDTH_M)
    ap.add_argument("--cloud_npz", default=None,
                    help="scene cloud from dump_scene_cloud.py: draws the "
                         "MEASURED semantic map (class-colored Gaussians) "
                         "under the paths instead of the assumed ribbon — "
                         "trees/obstacles become findable by coordinate")
    ap.add_argument("--video_episode", type=int, default=None,
                    help="draw ONE episode from metrics.json's video_episodes "
                         "(the episode episode_<N>.mp4 shows) so the figure "
                         "pairs 1:1 with its rollout video")
    args = ap.parse_args()

    blob = json.loads(Path(args.metrics).read_text())
    summary = blob["summary"]
    if args.video_episode is not None:
        vids = blob.get("video_episodes", [])
        if args.video_episode >= len(vids):
            sys.exit(f"metrics.json has {len(vids)} video_episodes — re-run "
                     "eval with the updated scripts for video trajectories")
        episodes = [vids[args.video_episode]]
    else:
        episodes = blob["episodes"]
    if "traj" not in episodes[0]:
        sys.exit("metrics.json has no 'traj' — re-run eval with the updated "
                 "eval_policy.py (per-step trajectory logging)")

    cal = NavCalibration.from_npz(args.poses_npz)
    path = cal.positions[:, :2]

    fig, ax = plt.subplots(figsize=(9, 9))
    if args.cloud_npz:
        from src.eval.palette import CLASS_COLORS_V14_255, CLASS_COLORS_255
        c = np.load(args.cloud_npz)
        pts, labs = c["points"], c["labels"].astype(int)
        # Crop to the navigation arena (trajectory bbox + margin) and keep
        # only near-ground geometry — the raw cloud spans 30m+ of far-field
        # splats that squash the arena into a sliver.
        lo_xy = path.min(0) - 3.0
        hi_xy = path.max(0) + 3.0
        keep = ((pts[:, 0] > lo_xy[0]) & (pts[:, 0] < hi_xy[0])
                & (pts[:, 1] > lo_xy[1]) & (pts[:, 1] < hi_xy[1])
                & (pts[:, 2] > -0.3) & (pts[:, 2] < 2.2) & (labs > 0))
        pts, labs = pts[keep], labs[keep]
        if labs.max() < 14:
            pal = CLASS_COLORS_V14_255
        else:
            pal = CLASS_COLORS_255
            labs = labs.copy()
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=pal[np.clip(labs, 0, len(pal) - 1)] / 255.0,
                   s=1.2, alpha=0.5, zorder=0, rasterized=True)
        ax.set_xlim(lo_xy[0], hi_xy[0])
        ax.set_ylim(lo_xy[1], hi_xy[1])
    else:
        poly = ribbon_polygon(path, args.ribbon_halfwidth)
        ax.fill(poly[:, 0], poly[:, 1], color="#c8b48c", alpha=0.7, zorder=1,
                label="recorded traversable corridor")
    ax.plot(path[:, 0], path[:, 1], color="#8a744a", lw=1, zorder=2)
    for f in range(0, len(path), 10):
        ax.annotate(str(f), path[f], fontsize=7, color="#6b5a35",
                    ha="center", va="center", zorder=3)

    seen_labels = set()
    for ep in episodes:
        t = np.asarray(ep["traj"], dtype=float)
        ok = ep["success"]
        color = "#2e8b57" if ok else "#e07b00"
        lbl = ("success" if ok else "failure") if ok not in seen_labels else None
        seen_labels.add(ok)
        ax.plot(t[:, 0], t[:, 1], color=color, lw=1.6, alpha=0.85, zorder=4,
                label=lbl)
        gx, gy = ep["goal_xy"]
        ax.plot([t[0, 0], gx], [t[0, 1], gy], color="0.45", ls="--", lw=0.8,
                alpha=0.6, zorder=3)
        ax.scatter([t[0, 0]], [t[0, 1]], s=25, facecolor="white",
                   edgecolor="black", zorder=5)
        ax.scatter([gx], [gy], marker="*", s=110, color="#b03060", zorder=5)
        # traj[:, 3] = footprint fraction on non-traversable classes (older
        # metrics carry 0/1 flags, which land in the "collision" bucket).
        grazes = t[(t[:, 3] > 0.01) & (t[:, 3] <= 0.2)]
        hits = t[t[:, 3] > 0.2]
        if len(grazes):
            ax.scatter(grazes[:, 0], grazes[:, 1], s=14, color="#f0a000",
                       zorder=6,
                       label="graze (1-20% footprint)"
                       if "graze" not in seen_labels else None)
            seen_labels.add("graze")
        if len(hits):
            ax.scatter(hits[:, 0], hits[:, 1], s=36, color="red", zorder=6,
                       label="collision (>20% footprint)"
                       if "hit" not in seen_labels else None)
            seen_labels.add("hit")

    if args.video_episode is not None:
        ep0 = episodes[0]
        title = args.title or (
            f"{Path(summary['checkpoint']).stem} — {ep0.get('video', '')} "
            f"(success={ep0['success']})")
    else:
        title = args.title or (
            f"{Path(summary['checkpoint']).stem} — success {summary['success_rate']}, "
            f"{summary['mean_collision_steps']} collision steps/ep "
            f"({summary['episodes']} eps)")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")

    print("\nSharpest bends of the recorded path (for a designed detour eval):")
    for deg, lo, hi in bend_windows(path):
        print(f"  frames {lo:3d}-{hi:3d}: {deg:5.1f} deg of turn "
              f"-> spawn_max_frame<={lo - 2}, goal_frame>={hi + 5}")


if __name__ == "__main__":
    main()
