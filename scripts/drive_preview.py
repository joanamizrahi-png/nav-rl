"""Moving-through-the-world diffusion preview at any resolution.

Walks the recorded trajectory pose-by-pose through BatchedLiveDiffusedBackend
(the exact machinery batched live training uses) and writes a side-by-side
rgb | semantic mp4 — the quality gate Joana judges before committing a
training run to a lower resolution / fewer sampler steps.

Usage (GPU node):
    python scripts/drive_preview.py \
        --scene rugd_trail_00 --clips_dir ... --poses_dir ... --labels_dir ... \
        --height 224 --width 336 --num_steps 2 --frames 40 \
        --out /scratch/.../outputs/drive_preview
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

from src.env.real_calibrated import CalibratedBackendConfig
from src.env.vec_live_env import BatchedLiveDiffusedBackend


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--live_ckpt",
                    default="/scratch/m000204-pm06b/joana/runs/train_semantic_v10/checkpoint-epoch-30.safetensors")
    ap.add_argument("--model_path", default="/scratch/m000204-pm06b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--height", type=int, default=336)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--num_steps", type=int, default=4)
    ap.add_argument("--frames", type=int, default=40,
                    help="drive length in poses along the recorded walk")
    ap.add_argument("--start", type=int, default=5)
    ap.add_argument("--heading", default="tangent",
                    choices=["tangent", "recorded"],
                    help="tangent: face the direction of motion (what a "
                         "driving robot does); recorded: the walker's camera "
                         "heading (pans and drifts off-path)")
    ap.add_argument("--target_xy", default=None,
                    help="'x,y': leave the recorded path at --start and walk "
                         "STRAIGHT AT this nav-frame point in 0.25 m policy "
                         "steps, camera facing it (obstacle visibility quest)")
    ap.add_argument("--goal_frame", type=int, default=None,
                    help="mark the pose at this frame index as the GOAL: "
                         "projected dot in both panels + topdown inset + "
                         "distance in the HUD (goal-placement design tool)")
    ap.add_argument("--goal_xy", default=None,
                    help="'x,y': mark this nav-frame point as the GOAL "
                         "(ignored if --goal_frame is set)")
    ap.add_argument("--spin", action="store_true",
                    help="J-spec spawn certification: hold position at --start "
                         "and rotate a full 360 over --frames steps; HUD shows "
                         "heading + reconstruction coverage per view")
    ap.add_argument("--spin_deg", type=float, default=360.0,
                    help="with --spin: sweep only +-this/2 around the path "
                         "tangent (visualize the goal cone's view range)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.height % 112 or args.width % 112:
        raise SystemExit("height/width must be multiples of 112 "
                         "(reconstructor patch 14 x VAE/DiT 16)")

    import cv2
    import torch
    from diffsynth.utils.class_taxonomy import v14_palette

    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        H=args.height,
        W=args.width,
    )
    world = BatchedLiveDiffusedBackend(cfg, checkpoint=args.live_ckpt)
    world.num_inference_steps = args.num_steps
    world.load_scene(args.scene)
    cal = world._calib[args.scene]

    pal = (v14_palette().numpy() * 255).astype(np.uint8)      # [14,3] RGB

    def tangent_pose(i: int) -> np.ndarray:
        pos = np.asarray(cal.positions)
        fwd = pos[min(i + 1, len(pos) - 1)] - pos[max(i - 1, 0)]
        fwd[2] = 0.0
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        pose = np.eye(4)
        pose[:3, 0] = fwd
        pose[:3, 1] = np.cross(up, fwd)
        pose[:3, 2] = up
        pose[:3, 3] = pos[i] * np.array([1.0, 1.0, 0.0])
        return pose.astype(np.float32)

    def pose_at(position_xy: np.ndarray, fwd_xy: np.ndarray) -> np.ndarray:
        fwd = np.array([fwd_xy[0], fwd_xy[1], 0.0])
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        pose = np.eye(4)
        pose[:3, 0] = fwd
        pose[:3, 1] = np.cross(up, fwd)
        pose[:3, 2] = up
        pose[:3, 3] = np.array([position_xy[0], position_xy[1], 0.0])
        return pose.astype(np.float32)

    target = (np.array([float(v) for v in args.target_xy.split(",")])
              if args.target_xy else None)

    goal = None
    if args.goal_frame is not None:
        goal = np.asarray(cal.positions[args.goal_frame], dtype=float).copy()
        goal[2] = 0.0
    elif args.goal_xy:
        gx, gy = (float(v) for v in args.goal_xy.split(","))
        goal = np.array([gx, gy, 0.0])

    path_xy = np.asarray(cal.positions, dtype=float)[:, :2]

    def topdown_inset(cur_xy, side):
        """Mini-map: recorded path, spawn end, current pose, goal. BGR."""
        allp = (np.vstack([path_xy, goal[None, :2]])
                if goal is not None else path_xy)
        lo = allp.min(0) - 1.0
        span = max(*(allp.max(0) + 1.0 - lo))
        def px(p):
            x = int((p[0] - lo[0]) / span * (side - 12)) + 6
            y = side - 1 - (int((p[1] - lo[1]) / span * (side - 12)) + 6)
            return x, y
        img = np.full((side, side, 3), 25, np.uint8)
        for a, b in zip(path_xy[:-1], path_xy[1:]):
            cv2.line(img, px(a), px(b), (190, 190, 190), 1, cv2.LINE_AA)
        cv2.circle(img, px(path_xy[0]), 3, (255, 255, 255), -1)   # pose 0
        if goal is not None:
            cv2.drawMarker(img, px(goal[:2]), (0, 255, 0),
                           cv2.MARKER_CROSS, 9, 2)
        cv2.circle(img, px(cur_xy), 4, (0, 140, 255), -1)         # current
        return img

    def project_goal(K, w2c):
        """Goal -> pixel via the render's own K/w2c (OpenCV convention)."""
        K = np.asarray(K, dtype=float).reshape(3, 3)
        M = np.asarray(w2c, dtype=float)
        if M.shape == (3, 4):
            M = np.vstack([M, [0.0, 0.0, 0.0, 1.0]])
        p = M @ np.array([goal[0], goal[1], goal[2], 1.0])
        if p[2] <= 0.1:
            return None
        u = K[0, 0] * p[0] / p[2] + K[0, 2]
        v = K[1, 1] * p[1] / p[2] + K[1, 2]
        if not (0 <= u < args.width and 0 <= v < args.height):
            return None
        return int(u), int(v)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"{args.scene}_{args.width}x{args.height}_s{args.num_steps}"
           + ("" if args.heading == "tangent" else "_rec")
           + ("" if args.start == 5 else f"_st{args.start}")
           + ("" if target is None else "_totgt")
           + (("" if not args.spin else
               ("_spin" if args.spin_deg >= 360.0 else f"_spin{args.spin_deg:.0f}")))
           + ("" if args.goal_frame is None else f"_gf{args.goal_frame}")
           + ("" if (args.goal_frame is not None or not args.goal_xy) else
              "_g" + args.goal_xy.replace(",", "_")))
    vw = None
    for step in range(args.frames):
        i = min(args.start + step, 80)
        if args.spin:
            i = args.start
            if args.spin_deg < 360.0:
                sweep = np.deg2rad(args.spin_deg)
                ang = sweep * (step / max(args.frames - 1, 1) - 0.5)
            else:
                ang = 2.0 * np.pi * step / max(args.frames, 1)
            base = tangent_pose(i)
            fwd = base[:2, 0]
            c, s = np.cos(ang), np.sin(ang)
            fwd_rot = np.array([c * fwd[0] - s * fwd[1],
                                s * fwd[0] + c * fwd[1]])
            pose = pose_at(base[:2, 3], fwd_rot)
        elif target is not None:
            origin = np.asarray(cal.positions[args.start])[:2]
            d = target - origin
            dist = np.linalg.norm(d)
            u = d / max(dist, 1e-6)
            cur = origin + u * min(0.25 * step, max(dist - 0.4, 0.0))
            pose = pose_at(cur, u)
        else:
            pose = (tangent_pose(i) if args.heading == "tangent"
                    else cal.robot_pose_nav(i))
        (rgb, K, w2c, lab) = world.render_batch([(0, pose)])[0]
        sem_rgb = pal[np.clip(lab, 0, 13).astype(int)]        # HxWx3 RGB
        frame = np.hstack([rgb, sem_rgb])[:, :, ::-1]         # to BGR
        frame = np.ascontiguousarray(frame)
        # In totgt mode the walk leaves the recorded path, so "pose i" would be
        # a lie (found 2026-08-30: the off-path tree got mislocated to "pose 25")
        # — label by steps/meters walked instead.
        if args.spin and args.spin_deg < 360.0:
            off = args.spin_deg * (step / max(args.frames - 1, 1) - 0.5)
            where = f"tangent{off:+.0f}deg @pose {i}"
        elif args.spin:
            where = f"heading {int(360 * step / max(args.frames, 1)):3d}deg @pose {i}"
        elif target is not None:
            where = f"step {step} ({0.25 * step:.1f}m walked)"
        else:
            where = f"pose {i}"
        hud = f"{tag}  {where}  {world.last_timings['total']:.2f}s"
        cov = getattr(world, "last_coverage", None)
        if cov is not None and cov == cov:
            hud += f"  cov {cov * 100:.0f}%"
        if goal is not None:
            gd = float(np.linalg.norm(goal[:2] - pose[:3, 3][:2]))
            hud += f"  goal {gd:.1f}m"
            uv = project_goal(K, w2c)
            if uv is not None:
                for dx in (0, args.width):
                    cv2.circle(frame, (uv[0] + dx, uv[1]), 8, (0, 255, 0), 2)
                    cv2.drawMarker(frame, (uv[0] + dx, uv[1]), (0, 255, 0),
                                   cv2.MARKER_CROSS, 10, 2)
            else:
                hud += " (goal off-screen)"
        side = max(96, args.height // 3)
        inset = topdown_inset(pose[:3, 3][:2], side)
        frame[frame.shape[0] - side - 8:frame.shape[0] - 8,
              args.width - side - 8:args.width - 8] = inset
        cv2.putText(frame, hud,
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                    cv2.LINE_AA)
        if vw is None:
            vw = cv2.VideoWriter(str(out_dir / f"DRIVE_{tag}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
                                 (frame.shape[1], frame.shape[0]))
        vw.write(frame)
        print(f"[{step + 1}/{args.frames}] pose {i} "
              f"{world.last_timings['total']:.2f}s", flush=True)
    vw.release()
    print(f"==> {out_dir / f'DRIVE_{tag}.mp4'}", flush=True)


if __name__ == "__main__":
    main()
