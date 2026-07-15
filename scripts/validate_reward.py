"""Offline reward validation on a recorded clip.

Loads (video + SAM3 labels + trajectory), sweeps the trajectory frame by
frame computing the decomposed reward, and produces:

  1. A CSV of per-frame reward components (for later analysis).
  2. A 4-panel matplotlib plot: total / semantic / goal / collision vs time.
  3. Optionally: a side-by-side video overlaying reward on each frame.

Usage:
    python scripts/validate_reward.py \\
        --video path/to/rugd_trail_00.mp4 \\
        --labels path/to/outputs/sam3_labels/rugd_trail_00.npz \\
        --output_dir outputs/reward_eval/rugd_trail_00/

Milestone A deliverable. See DECISIONS.md and OPTIONS.md at repo root.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


# Make the parent package importable regardless of where you run this from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Also make NeoVerse importable — load_clip imports diffsynth.utils.auxiliary.
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

from src.eval.traversability import load_traversability, load_class_names, NUM_CLASSES
from src.eval.load_clip import load_clip
from src.eval.reward_2d import (
    compute_reward, RewardWeights,
    _footprint_corners_world, _project_points, _fill_polygon,
    GO2_BODY_LENGTH, GO2_BODY_WIDTH,
)


def build_non_traversable_mask(traversability_scores: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Classes with score <= threshold are considered collision-worthy."""
    return traversability_scores <= threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--num_frames", type=int, default=81)
    ap.add_argument("--pose_source", default="synthetic", choices=["synthetic", "npz"])
    ap.add_argument("--poses_npz", type=Path, default=None)
    ap.add_argument("--look_ahead_dist", type=float, default=0.5,
                    help="meters ahead of the robot to place the footprint")
    ap.add_argument("--collision_threshold", type=float, default=0.1,
                    help="traversability score at/below which a class counts as collision")
    ap.add_argument("--overlay", action="store_true",
                    help="also produce an MP4 with the footprint polygon, reward, "
                         "and dominant class overlaid on each frame (for eyeballing)")
    # Weight knobs -- for ablation runs. Defaults match RewardWeights defaults.
    ap.add_argument("--w_sem", type=float, default=1.0)
    ap.add_argument("--w_goal", type=float, default=0.5)
    ap.add_argument("--w_col", type=float, default=5.0)
    ap.add_argument("--w_step", type=float, default=0.05,
                    help="constant per-frame negative reward (kills 'stand still on grass' gaming)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load ----
    print(f"[validate_reward] loading clip: {args.video}", flush=True)
    clip = load_clip(
        video_path=args.video,
        labels_path=args.labels,
        pose_source=args.pose_source,
        poses_npz_path=args.poses_npz,
        num_frames=args.num_frames,
    )
    T, H, W = clip.frames_rgb.shape[:3]
    print(f"[validate_reward] {T} frames at {W}x{H}", flush=True)

    trav = load_traversability()
    class_names = load_class_names()
    non_traversable = build_non_traversable_mask(trav, threshold=args.collision_threshold)
    print(f"[validate_reward] {int(non_traversable.sum())}/{NUM_CLASSES} classes are collision-worthy", flush=True)

    weights = RewardWeights(semantic=args.w_sem, goal=args.w_goal, collision=args.w_col, step_cost=args.w_step)
    print(f"[validate_reward] weights: sem={weights.semantic} goal={weights.goal} col={weights.collision} step={weights.step_cost}", flush=True)

    # ---- 2. Sweep the trajectory ----
    rows = []
    prev_position = None
    for t in range(T):
        breakdown = compute_reward(
            semantic_image=clip.labels[t],
            K=clip.K,
            w2c=clip.w2c[t],
            robot_position=clip.positions[t],
            robot_heading=clip.headings[t],
            goal=clip.goal,
            traversability_scores=trav,
            non_traversable_mask=non_traversable,
            previous_position=prev_position,
            look_ahead_dist=args.look_ahead_dist,
            weights=weights,
        )
        row = {"frame": t, **breakdown.to_dict()}
        row["dominant_class_name"] = (
            class_names[breakdown.dominant_class_id]
            if 0 <= breakdown.dominant_class_id < NUM_CLASSES else "n/a"
        )
        rows.append(row)
        prev_position = clip.positions[t]

    # ---- 3. Save CSV ----
    import csv
    csv_path = args.output_dir / "reward.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[validate_reward] wrote {csv_path}", flush=True)

    # ---- 4. Plot ----
    _plot_reward_curves(rows, out_path=args.output_dir / "reward_curves.png")
    print(f"[validate_reward] wrote {args.output_dir / 'reward_curves.png'}", flush=True)

    # ---- 4b. Overlay video (--overlay) ----
    if args.overlay:
        overlay_path = args.output_dir / "reward_overlay.mp4"
        _write_overlay_video(
            clip, rows, class_names, out_path=overlay_path,
            look_ahead_dist=args.look_ahead_dist,
        )
        print(f"[validate_reward] wrote {overlay_path}", flush=True)

    # ---- 5. Print a short summary ----
    total = np.array([r["total"] for r in rows])
    sem = np.array([r["semantic"] for r in rows])
    coll = np.array([r["collision"] for r in rows])
    print("[validate_reward] summary:")
    print(f"  total   mean={total.mean():+.3f}  min={total.min():+.3f}  max={total.max():+.3f}")
    print(f"  sem     mean={sem.mean():+.3f}")
    print(f"  coll    frac non-zero = {(coll < 0).mean():.2%}")
    print(f"  dominant classes: {_summarize_classes(rows)}")


def _plot_reward_curves(rows, out_path: Path):
    """4-panel plot: total / semantic / goal / collision vs frame."""
    import matplotlib.pyplot as plt

    frames = [r["frame"] for r in rows]
    total = [r["total"] for r in rows]
    sem = [r["semantic"] for r in rows]
    goal = [r["goal"] for r in rows]
    coll = [r["collision"] for r in rows]

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for ax, ys, label, color in [
        (axes[0], total, "total",     "black"),
        (axes[1], sem,   "semantic",  "tab:green"),
        (axes[2], goal,  "goal",      "tab:blue"),
        (axes[3], coll,  "collision", "tab:red"),
    ]:
        ax.plot(frames, ys, color=color, linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("frame")
    fig.suptitle("Reward-vs-time breakdown (offline validation)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _summarize_classes(rows) -> str:
    from collections import Counter
    counter = Counter(r["dominant_class_name"] for r in rows)
    top = counter.most_common(5)
    return ", ".join(f"{name}({count})" for name, count in top)


def _write_overlay_video(clip, rows, class_names, out_path: Path, look_ahead_dist: float):
    """Render an MP4 where each frame shows: RGB | RGB+labels+polygon | labels-only.

    Three-panel view side by side. Left = raw RGB. Middle = RGB with SAM3 labels
    tinted on top and the reward polygon drawn. Right = SAM3 labels colorized
    with the canonical palette. Lets you visually verify BOTH the SAM3 label
    quality AND where the reward polygon lands.
    """
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v3 as iio

    # Palette for colorizing SAM3 labels. Load from the neoverse CLASS_COLORS
    # if importable; otherwise build a random-but-fixed palette so classes are
    # visually distinct.
    palette = _load_palette(NUM_CLASSES)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
        except Exception:
            font = ImageFont.load_default()

    T = len(clip.frames_rgb)
    out_frames = []

    for t in range(T):
        rgb = clip.frames_rgb[t].copy()          # (H, W, 3) uint8
        labels = clip.labels[t]                   # (H, W) int
        H, W = rgb.shape[:2]

        # ---- Panel A: raw RGB
        panel_a = Image.fromarray(rgb)

        # ---- Panel B: RGB with labels tinted 50% + polygon
        colored = palette[np.clip(labels, 0, NUM_CLASSES - 1)]         # (H, W, 3) uint8
        blended = (0.5 * rgb + 0.5 * colored).clip(0, 255).astype(np.uint8)
        panel_b = Image.fromarray(blended)
        draw_b = ImageDraw.Draw(panel_b, "RGBA")

        corners_world = _footprint_corners_world(
            clip.positions[t], clip.headings[t],
            look_ahead_dist=look_ahead_dist,
            length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH,
        )
        corners_uv, in_front = _project_points(corners_world, clip.K, clip.w2c[t])
        if in_front.all():
            poly = [(float(u), float(v)) for u, v in corners_uv]
            r = rows[t]["total"]
            outline = (0, 255, 0, 255) if r >= 0 else (255, 0, 0, 255)
            draw_b.polygon(poly, outline=outline)
            # Second stroke for visibility
            for i in range(len(poly)):
                draw_b.line([poly[i], poly[(i + 1) % len(poly)]], fill=outline, width=2)

        # ---- Panel C: labels-only (full palette)
        panel_c = Image.fromarray(colored)

        # ---- Stack panels horizontally
        row_img = Image.new("RGB", (W * 3, H), (0, 0, 0))
        row_img.paste(panel_a, (0, 0))
        row_img.paste(panel_b, (W, 0))
        row_img.paste(panel_c, (W * 2, 0))

        # ---- Text banner on top
        draw = ImageDraw.Draw(row_img, "RGBA")
        lines = [
            f"frame {t:3d}   total={rows[t]['total']:+.2f}   sem={rows[t]['semantic']:+.2f}  goal={rows[t]['goal']:+.2f}  coll={rows[t]['collision']:+.2f}",
            f"dominant in polygon: {rows[t]['dominant_class_name']}  ({rows[t]['n_footprint_pixels']} px)   |   RGB   |   RGB+SAM3 labels+polygon   |   SAM3 labels only",
        ]
        line_h = 16
        box_h = line_h * len(lines) + 4
        draw.rectangle([(0, 0), (W * 3, box_h)], fill=(0, 0, 0, 190))
        for i, ln in enumerate(lines):
            draw.text((6, 2 + i * line_h), ln, fill=(255, 255, 255, 255), font=font)

        out_frames.append(np.array(row_img.convert("RGB")))

    out_arr = np.stack(out_frames, axis=0)
    iio.imwrite(str(out_path), out_arr, fps=8,
                codec="libx264", macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"])


def _load_palette(n: int) -> np.ndarray:
    """Return (n, 3) uint8 palette. Uses the local canonical palette which
    mirrors NeoVerse's CLASS_COLORS -- so colors are consistent across Mac
    and Marlowe regardless of whether torch/diffsynth are importable."""
    from src.eval.palette import CLASS_COLORS_255
    return CLASS_COLORS_255[:n]


if __name__ == "__main__":
    main()
