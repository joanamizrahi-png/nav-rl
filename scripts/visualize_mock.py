"""Visualize the MockWorldBackend scene + robot view + reward polygon.

Produces a 2-image side-by-side figure:
  1. Top-down scene map (colorized by class) with robot pose + goal marked
  2. First-person view from the robot's pose, with SAM3 labels tinted + reward polygon

Also runs one episode with a hand-scripted "walk forward" policy and saves an
MP4 of the RGB + semantic + reward-over-time.

Usage:
    python scripts/visualize_mock.py --output_dir outputs/mock_viz/
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.env.mock_backend import MockWorldBackend, MockSemanticBackend
from src.env.scene_env import SceneEnv, SceneEnvConfig
from src.eval.reward_2d import (
    RewardWeights, compute_reward,
    _footprint_corners_world, _project_points,
    GO2_BODY_LENGTH, GO2_BODY_WIDTH,
)
from src.eval.traversability import load_traversability, load_class_names
from src.eval.palette import CLASS_COLORS_255


def _draw_topdown(world: MockWorldBackend, out_path: Path,
                  robot_pose: np.ndarray, goal: np.ndarray):
    """Colorized top-down map with robot + goal markers."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, FancyArrow

    grid = world._scene_grid            # (gy, gx) class IDs
    palette = CLASS_COLORS_255[np.clip(grid, 0, len(CLASS_COLORS_255) - 1)]   # (gy, gx, 3)

    fig, ax = plt.subplots(figsize=(9, 6))
    extent = (world._x_min, world._x_max, world._y_min, world._y_max)
    ax.imshow(palette, extent=extent, origin="lower", aspect="equal")

    # Robot marker + heading arrow.
    rp = robot_pose[:3, 3]
    heading = robot_pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
    ax.add_patch(Circle((rp[0], rp[1]), radius=0.2, fc="magenta", ec="black", zorder=10))
    ax.add_patch(FancyArrow(rp[0], rp[1], heading[0] * 0.8, heading[1] * 0.8,
                            width=0.05, head_width=0.2, fc="magenta", ec="black", zorder=10))
    # Goal marker.
    ax.add_patch(Circle((goal[0], goal[1]), radius=0.3, fc="yellow", ec="black", zorder=10))
    ax.text(goal[0] + 0.4, goal[1], "GOAL", fontsize=10, verticalalignment="center", zorder=10)
    ax.text(rp[0] - 0.4, rp[1] - 0.3, "START", fontsize=10, ha="right", zorder=10)

    ax.set_title("Mock scene (top-down)")
    ax.set_xlabel("world x (meters, forward)")
    ax.set_ylabel("world y (meters, right)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _draw_pov_frame(world: MockWorldBackend, sem: MockSemanticBackend,
                    trav_scores: np.ndarray, class_names: list[str],
                    robot_pose: np.ndarray, goal: np.ndarray,
                    step_idx: int, reward_breakdown, out_path: Path):
    """RGB + labels + polygon overlay for one pose."""
    rgb, K, w2c = world.render(robot_pose)
    labels = sem.segment(rgb)

    palette = CLASS_COLORS_255[np.clip(labels, 0, len(CLASS_COLORS_255) - 1)]

    # ---- panel A: raw RGB ----
    panel_a = Image.fromarray(rgb)

    # ---- panel B: RGB with labels tinted + polygon ----
    blended = (0.5 * rgb.astype(np.float32) + 0.5 * palette).clip(0, 255).astype(np.uint8)
    panel_b = Image.fromarray(blended)
    draw = ImageDraw.Draw(panel_b, "RGBA")

    robot_position = robot_pose[:3, 3]
    robot_heading = robot_pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
    corners_world = _footprint_corners_world(
        robot_position, robot_heading,
        look_ahead_dist=1.5,
        length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH,
    )
    corners_uv, in_front = _project_points(corners_world, K, w2c)
    if in_front.all():
        poly = [(float(u), float(v)) for u, v in corners_uv]
        outline = (0, 255, 0, 255) if reward_breakdown.total >= 0 else (255, 0, 0, 255)
        for i in range(len(poly)):
            draw.line([poly[i], poly[(i + 1) % len(poly)]], fill=outline, width=2)

    # ---- panel C: pure labels ----
    panel_c = Image.fromarray(palette)

    # ---- stack + text ----
    H, W = rgb.shape[:2]
    canvas = Image.new("RGB", (W * 3, H + 40), (0, 0, 0))
    canvas.paste(panel_a, (0, 40))
    canvas.paste(panel_b, (W, 40))
    canvas.paste(panel_c, (W * 2, 40))

    d = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
    except Exception:
        font = ImageFont.load_default()
    dom = reward_breakdown.dominant_class_id
    dom_name = class_names[dom] if 0 <= dom < len(class_names) else "n/a"
    text = (f"step {step_idx}   total={reward_breakdown.total:+.2f}   "
            f"sem={reward_breakdown.semantic:+.2f}  goal={reward_breakdown.goal:+.2f}  "
            f"coll={reward_breakdown.collision:+.2f}  step_cost={reward_breakdown.step:+.2f}   "
            f"dominant: {dom_name}   |   RGB   |   RGB+labels+polygon   |   labels")
    d.text((6, 12), text, fill=(255, 255, 255), font=font)

    canvas.save(out_path)


def _rollout_episode(world, sem, trav_scores, non_trav, class_names, out_dir):
    """Walk the robot forward for 25 steps, save a frame per step, then bundle into MP4."""
    import imageio.v3 as iio

    cfg = SceneEnvConfig(max_steps=25, step_size_m=0.3, yaw_step_rad=0.3,
                         reward=RewardWeights(step_cost=0.05), look_ahead_dist=1.5)
    env = SceneEnv(world_backend=world, semantic_backend=sem, scene_ids=["mock"], cfg=cfg)
    obs, _ = env.reset(seed=0)

    frame_dir = out_dir / "episode_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Manually recompute reward breakdown for each pose (env.step returns scalar; we
    # want the full breakdown object for the visualization).
    robot_pose = world.start_pose("mock").copy()
    prev_position = None

    frames = []
    for t in range(25):
        rgb, K, w2c = world.render(robot_pose)
        labels = sem.segment(rgb)
        robot_position = robot_pose[:3, 3].copy()
        robot_heading = robot_pose[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float32)

        breakdown = compute_reward(
            semantic_image=labels, K=K, w2c=w2c,
            robot_position=robot_position, robot_heading=robot_heading,
            goal=world.goal_position("mock"),
            traversability_scores=trav_scores, non_traversable_mask=non_trav,
            previous_position=prev_position, look_ahead_dist=1.5,
            body_length=GO2_BODY_LENGTH, body_width=GO2_BODY_WIDTH,
            weights=RewardWeights(step_cost=0.05),
        )

        frame_path = frame_dir / f"frame_{t:03d}.png"
        _draw_pov_frame(world, sem, trav_scores, class_names, robot_pose,
                        world.goal_position("mock"), t, breakdown, frame_path)
        frames.append(np.array(Image.open(frame_path)))

        # Advance forward 0.3m, no yaw
        prev_position = robot_position
        forward_world = robot_pose[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
        robot_pose[:3, 3] += 0.3 * forward_world

    mp4_path = out_dir / "walk_forward_episode.mp4"
    iio.imwrite(str(mp4_path), np.stack(frames, axis=0), fps=6,
                codec="libx264", macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"[visualize_mock] wrote {mp4_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/mock_viz"))
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    world = MockWorldBackend(H=64, W=64)
    sem = MockSemanticBackend(world)
    trav = load_traversability()
    non_trav = trav <= 0.1
    class_names = load_class_names()

    start_pose = world.start_pose("mock")
    goal = world.goal_position("mock")

    # 1. Top-down map
    _draw_topdown(world, args.output_dir / "scene_topdown.png", start_pose, goal)
    print(f"[visualize_mock] wrote {args.output_dir / 'scene_topdown.png'}")

    # 2. First-person view at start pose (single-frame version)
    rgb, K, w2c = world.render(start_pose)
    labels = sem.segment(rgb)
    br = compute_reward(
        semantic_image=labels, K=K, w2c=w2c,
        robot_position=start_pose[:3, 3],
        robot_heading=start_pose[:3, :3] @ np.array([1.0, 0.0, 0.0]),
        goal=goal, traversability_scores=trav, non_traversable_mask=non_trav,
        previous_position=None, look_ahead_dist=1.5,
        body_length=GO2_BODY_LENGTH, body_width=GO2_BODY_WIDTH,
        weights=RewardWeights(step_cost=0.05),
    )
    _draw_pov_frame(world, sem, trav, class_names, start_pose, goal, 0, br,
                    args.output_dir / "pov_start.png")
    print(f"[visualize_mock] wrote {args.output_dir / 'pov_start.png'}")

    # 3. Rollout episode (walk forward)
    _rollout_episode(world, sem, trav, non_trav, class_names, args.output_dir)

    print(f"\n[visualize_mock] all outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
