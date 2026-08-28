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
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.scene}_{args.width}x{args.height}_s{args.num_steps}"
    vw = None
    for step in range(args.frames):
        i = min(args.start + step, 80)
        pose = cal.robot_pose_nav(i)
        (rgb, K, w2c, lab) = world.render_batch([(0, pose)])[0]
        sem_rgb = pal[np.clip(lab, 0, 13).astype(int)]        # HxWx3 RGB
        frame = np.hstack([rgb, sem_rgb])[:, :, ::-1]         # to BGR
        frame = np.ascontiguousarray(frame)
        cv2.putText(frame, f"{tag}  pose {i}  {world.last_timings['total']:.2f}s",
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
