"""Diagnose the reconstructor's per-frame camera poses for our 4 test clips.

Purpose: BEFORE building RealWorldBackend (Milestone B), understand the
reconstructor's coordinate conventions:

  1. Are all 4 clips' first-frame poses at ~origin? (=> reconstructor normalizes
     the world frame per clip, first camera = origin.)
  2. What direction does the camera point at frame 0? (=> tells us which axis is
     "forward" in reconstructor's convention.)
  3. What are the position ranges? Consistent across clips, or per-clip variable?
     (Small ranges like 0.5m suggest scale ambiguity.)
  4. Is there a consistent "ground plane" in reconstructor coords? (Camera z
     should be roughly constant if the ground is flat and camera height is fixed.)

Outputs:
  outputs/pose_diag/summary.csv                 — per-clip stats
  outputs/pose_diag/trajectories_3d.png         — 3D scatter, one line per clip
  outputs/pose_diag/trajectories_topdown.png    — top-down x-y plot
  outputs/pose_diag/trajectories_z_over_time.png — z coord per frame (ground plane check)

Usage:
    python scripts/diagnose_poses.py --clips rugd_trail_00 rugd_creek_00 rugd_park-1_00 cityscapes_stuttgart_00_00
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import matplotlib.pyplot as plt


def load_poses(clip: str, data_dir: Path) -> dict:
    """Load a poses.npz file and return its arrays."""
    path = data_dir / f"{clip}_poses.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing pose file: {path}")
    d = np.load(path)
    return {
        "positions": d["positions"],   # (T, 3)
        "headings": d["headings"],     # (T, 3)
        "w2c": d["w2c"],               # (T, 4, 4)
        "K": d["K"],                   # (3, 3)
    }


def summarize(clip: str, poses: dict) -> dict:
    """Compute per-clip stats: ranges, start pose, heading of first frame."""
    p = poses["positions"]
    h = poses["headings"]

    return {
        "clip": clip,
        "n_frames": len(p),
        "x_min": p[:, 0].min(), "x_max": p[:, 0].max(), "x_range": p[:, 0].max() - p[:, 0].min(),
        "y_min": p[:, 1].min(), "y_max": p[:, 1].max(), "y_range": p[:, 1].max() - p[:, 1].min(),
        "z_min": p[:, 2].min(), "z_max": p[:, 2].max(), "z_range": p[:, 2].max() - p[:, 2].min(),
        "start_pos_x": p[0, 0], "start_pos_y": p[0, 1], "start_pos_z": p[0, 2],
        "start_heading_x": h[0, 0], "start_heading_y": h[0, 1], "start_heading_z": h[0, 2],
        "K_fx": poses["K"][0, 0], "K_fy": poses["K"][1, 1],
        "K_cx": poses["K"][0, 2], "K_cy": poses["K"][1, 2],
    }


def _get_axis_lims(all_positions_list):
    """Common axis limits across all clips for consistent 3D plots."""
    all_p = np.concatenate(all_positions_list)
    lo = all_p.min(axis=0) - 0.2
    hi = all_p.max(axis=0) + 0.2
    return lo, hi


def plot_trajectories_3d(clips_poses: dict, out_path: Path):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    lo, hi = _get_axis_lims([p["positions"] for p in clips_poses.values()])

    colors = plt.cm.tab10(np.linspace(0, 1, len(clips_poses)))
    for (clip, poses), color in zip(clips_poses.items(), colors):
        p = poses["positions"]
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linewidth=1.5, label=clip)
        # Mark start (green dot) and end (red dot)
        ax.scatter(*p[0], color="green", s=60, edgecolors="black", zorder=10)
        ax.scatter(*p[-1], color="red",   s=60, edgecolors="black", zorder=10)
        # Heading arrow at start
        h0 = poses["headings"][0] * 0.05  # tiny arrow proportional to scale
        ax.quiver(p[0, 0], p[0, 1], p[0, 2], h0[0], h0[1], h0[2], color=color, linewidth=2, arrow_length_ratio=0.4)

    ax.set_xlabel("recon x")
    ax.set_ylabel("recon y")
    ax.set_zlabel("recon z")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_title("Camera trajectories in reconstructor's world frame\n(green = start, red = end, arrow = frame-0 heading)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_trajectories_topdown(clips_poses: dict, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plots = [("xy", 0, 1, "recon x", "recon y"),
             ("xz", 0, 2, "recon x", "recon z"),
             ("yz", 1, 2, "recon y", "recon z")]

    colors = plt.cm.tab10(np.linspace(0, 1, len(clips_poses)))
    for ax, (name, i, j, xl, yl) in zip(axes, plots):
        for (clip, poses), color in zip(clips_poses.items(), colors):
            p = poses["positions"]
            ax.plot(p[:, i], p[:, j], color=color, linewidth=1.5, label=clip)
            ax.scatter(p[0, i], p[0, j], color="green", s=40, edgecolors="black", zorder=10)
            ax.scatter(p[-1, i], p[-1, j], color="red", s=40, edgecolors="black", zorder=10)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{name} projection")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="datalim")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("Trajectory projections — check for scale + which axis is 'up'")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_z_over_time(clips_poses: dict, out_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(clips_poses)))

    for i, (axis_name, ax) in enumerate(zip(["x", "y", "z"], axes)):
        for (clip, poses), color in zip(clips_poses.items(), colors):
            ax.plot(poses["positions"][:, i], color=color, linewidth=1.5, label=clip)
        ax.set_ylabel(f"recon {axis_name}")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("frame")
    fig.suptitle("Position component vs frame — flat axes suggest ground plane / constant camera height")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+",
                    default=["rugd_trail_00", "rugd_creek_00", "rugd_park-1_00", "cityscapes_stuttgart_00_00"])
    ap.add_argument("--data_dir", type=Path, default=REPO_ROOT / "data")
    ap.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "pose_diag")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all clips
    clips_poses = {}
    for clip in args.clips:
        try:
            clips_poses[clip] = load_poses(clip, args.data_dir)
        except FileNotFoundError as e:
            print(f"[diagnose_poses] SKIP: {e}")
    if not clips_poses:
        print("[diagnose_poses] no pose files found; nothing to plot.")
        return

    # ---- summary CSV ----
    rows = [summarize(clip, poses) for clip, poses in clips_poses.items()]
    csv_path = args.output_dir / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[diagnose_poses] wrote {csv_path}")

    # Print summary to stdout
    print("\nPer-clip position ranges (recon units):")
    for row in rows:
        print(f"  {row['clip']:32s} "
              f"n={row['n_frames']}  "
              f"x_range={row['x_range']:+.3f}  "
              f"y_range={row['y_range']:+.3f}  "
              f"z_range={row['z_range']:+.3f}  "
              f"start=({row['start_pos_x']:+.3f},{row['start_pos_y']:+.3f},{row['start_pos_z']:+.3f})  "
              f"heading0=({row['start_heading_x']:+.2f},{row['start_heading_y']:+.2f},{row['start_heading_z']:+.2f})")

    print("\nPer-clip intrinsics:")
    for row in rows:
        print(f"  {row['clip']:32s} "
              f"fx={row['K_fx']:.1f} fy={row['K_fy']:.1f} "
              f"cx={row['K_cx']:.1f} cy={row['K_cy']:.1f}")

    # ---- plots ----
    plot_trajectories_3d(clips_poses, args.output_dir / "trajectories_3d.png")
    plot_trajectories_topdown(clips_poses, args.output_dir / "trajectories_topdown.png")
    plot_z_over_time(clips_poses, args.output_dir / "components_over_time.png")

    print(f"\n[diagnose_poses] plots in {args.output_dir}/")
    print(
        "\nWhat to look for in the plots:\n"
        "  * Are all clips' start (green) at ~origin? => reconstructor normalizes per clip.\n"
        "  * Which axis has the LEAST variation across all clips' trajectories?\n"
        "      That's the ground plane's 'up' axis (camera stays at ~constant height).\n"
        "  * Do position ranges make sense? RUGD Jackal moves ~5m per clip realistically;\n"
        "      if range is 0.5m across the whole trajectory, scale is normalized/wrong.\n"
        "  * Are frame-0 heading arrows all pointing the same-ish way? => consistent axis convention.\n"
    )


if __name__ == "__main__":
    main()
