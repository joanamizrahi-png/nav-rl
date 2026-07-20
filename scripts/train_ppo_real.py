"""PPO first version on a REAL scene (Milestone B/C glue — Thursday deliverable).

One RUGD scene, rasterizer-only observations (cheap, holey — fine for the loop
smoke), reward from Gaussian-rasterized labels (real observed geometry; never
generated RGB). All distances in REAL METERS via the extract_poses calibration.

Usage (Marlowe, GPU):
    python scripts/train_ppo_real.py \
        --scene rugd_trail_00 \
        --clips_dir /scratch/m000204-pm06b/joana/data/rugd_clips \
        --poses_dir /scratch/m000204-pm06b/joana/outputs/poses \
        --labels_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels \
        --total_steps 10000 \
        --output_dir outputs/ppo_real_trail00

Outputs: ppo_final.zip, checkpoints/, tensorboard/, rollout.mp4 (eval episode).
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

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from src.env.scene_env import SceneEnv, SceneEnvConfig
from src.env.real_calibrated import (
    CalibratedRealWorldBackend, CalibratedBackendConfig, GaussianLabelBackend,
)
from src.eval.reward_2d import RewardWeights


def make_env(args):
    cfg = CalibratedBackendConfig(
        scene_video_paths={args.scene: f"{args.clips_dir}/{args.scene}.mp4"},
        scene_poses_paths={args.scene: f"{args.poses_dir}/{args.scene}_poses.npz"},
        scene_labels_paths={args.scene: f"{args.labels_dir}/{args.scene}.npz"},
        goal_frame=args.goal_frame,
        render_mode="rasterizer_only",       # cheap per-step; diffusion later
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
    )
    world = CalibratedRealWorldBackend(cfg)
    sem = GaussianLabelBackend(world)
    env_cfg = SceneEnvConfig(
        max_steps=args.max_steps,
        step_size_m=0.3,                     # real robot pace (extract_poses: 0.2-0.4 m/frame)
        yaw_step_rad=0.3,
        reward=RewardWeights(semantic=1.0, goal=0.5, collision=5.0, step_cost=0.05),
        look_ahead_dist=2.0,                 # level-camera blind zone ends ~1.6 m
        goal_radius=1.0,
        collision_threshold=0.1,
    )
    env = SceneEnv(world_backend=world, semantic_backend=sem,
                   scene_ids=[args.scene], cfg=env_cfg)
    return Monitor(env)


def save_rollout_video(model, env, out_path: Path, max_frames=120):
    import imageio.v3 as iio
    frames = []
    obs, _ = env.reset()
    done = False
    while not done and len(frames) < max_frames:
        frames.append(env.render().copy())
        action, _ = model.predict(obs, deterministic=True)
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    iio.imwrite(str(out_path), np.stack(frames), fps=8,
                codec="libx264", macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"[train_ppo_real] rollout video: {out_path} "
          f"({len(frames)} frames, reached_goal={info.get('reached_goal')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm06b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--total_steps", type=int, default=10_000)
    ap.add_argument("--max_steps", type=int, default=60)
    ap.add_argument("--goal_frame", type=int, default=45,
                    help="goal = real-trajectory position at this frame (~45 => a dozen meters in)")
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/ppo_real"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_wandb", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ONE env: each env holds a reconstructed scene on the GPU. Parallel envs
    # would multiply VRAM; not worth it for the smoke.
    env = make_env(args)

    model = PPO(
        "MultiInputPolicy", env,
        verbose=1, seed=args.seed,
        tensorboard_log=str(args.output_dir / "tensorboard"),
        n_steps=128, batch_size=64,
        learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    )

    callbacks = [CheckpointCallback(save_freq=2_000,
                                    save_path=str(args.output_dir / "checkpoints"),
                                    name_prefix="ppo")]
    if args.use_wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            wandb.init(project="nav-rl", name=f"ppo_real_{args.scene}",
                       config=vars(args) | {"total_steps": args.total_steps},
                       sync_tensorboard=True, dir=str(args.output_dir))
            callbacks.append(WandbCallback())
        except Exception as e:
            print(f"[train_ppo_real] wandb unavailable ({e}); continuing without")

    print(f"[train_ppo_real] training {args.total_steps} steps on {args.scene} ...")
    model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=True)
    model.save(str(args.output_dir / "ppo_final.zip"))

    print("[train_ppo_real] eval rollout ...")
    save_rollout_video(model, env, args.output_dir / "rollout.mp4")


if __name__ == "__main__":
    main()
