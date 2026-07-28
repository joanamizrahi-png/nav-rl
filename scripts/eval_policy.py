"""Evaluate a saved policy checkpoint: success rate, steps-to-goal, videos.

Why: training saves checkpoints every 2k steps, and PPO sometimes destroys its
own best policy late in training (observed twice). The final rollout.mp4 then
shows the corpse. This tool evaluates ANY checkpoint over N episodes with the
standard protocol, so we can (a) rescue the pre-collapse peak and (b) report
honest numbers: success rate, mean steps-to-goal, collision exposure.

Usage (Marlowe GPU):
    python scripts/eval_policy.py \
        --checkpoint /scratch/.../ppo_v3_bc_trail00/checkpoints/ppo_130000_steps.zip \
        --scene rugd_trail_00 --episodes 20 \
        --clips_dir ... --poses_dir ... --labels_dir ... \
        --out_dir /scratch/.../outputs/eval_v3_peak
Outputs: metrics.json, eval_summary printed, rollout videos for the first 3 episodes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
NEOVERSE_ROOT = REPO_ROOT.parent / "NeoVerse"
if NEOVERSE_ROOT.exists():
    sys.path.insert(0, str(NEOVERSE_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.env.scene_env import SceneEnv, SceneEnvConfig
from src.env.real_calibrated import (
    CalibratedRealWorldBackend, CalibratedBackendConfig, GaussianLabelBackend,
)
from src.eval.reward_2d import RewardWeights


def build_env(args):
    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        goal_frame=args.goal_frame,
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
    )
    world = CalibratedRealWorldBackend(cfg)
    sem = GaussianLabelBackend(world)
    env_cfg = SceneEnvConfig(
        max_steps=60, step_size_m=0.25, yaw_step_rad=0.3,
        reward=RewardWeights(semantic=1.0, goal=1.5, collision=1.0,
                             step_cost=0.05, void_cost=0.3),
        look_ahead_dist=1.5, goal_radius=0.75, collision_threshold=0.1,
        spin_cost=0.05, random_spawn=True,
    )
    return SceneEnv(world_backend=world, semantic_backend=sem,
                    scene_ids=[args.scene], cfg=env_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--goal_frame", type=int, default=30)
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm06b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--videos", type=int, default=3)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(build_env(args))
    model = PPO.load(args.checkpoint, env=env, device="cuda")
    print(f"loaded {args.checkpoint}", flush=True)

    # Reuse the HUD rollout recorder from the training script.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from train_ppo_real import save_rollout_video

    results = []
    for ep in range(args.episodes):
        if ep < args.videos:
            save_rollout_video(model, env, args.out_dir / f"episode_{ep}.mp4")
            # save_rollout_video runs its own episode; count it via a fresh one below
        obs, _ = env.reset()
        done, steps, collided, total_r = False, 0, 0, 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += float(r)
            steps += 1
            if info.get("collision", 0) < -0.01:
                collided += 1
            done = term or trunc
        results.append({"episode": ep, "success": bool(term),
                        "steps": steps, "return": round(total_r, 2),
                        "collision_steps": collided,
                        "final_dist": round(float(info.get("dist_to_goal", -1)), 2)})
        print(f"ep {ep:2d}: success={term} steps={steps} return={total_r:+.1f}", flush=True)

    succ = [r for r in results if r["success"]]
    summary = {
        "checkpoint": str(args.checkpoint),
        "episodes": args.episodes,
        "success_rate": round(len(succ) / args.episodes, 3),
        "mean_steps_to_goal": round(float(np.mean([r["steps"] for r in succ])), 1) if succ else None,
        "mean_return": round(float(np.mean([r["return"] for r in results])), 2),
        "mean_collision_steps": round(float(np.mean([r["collision_steps"] for r in results])), 2),
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
