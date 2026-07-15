"""Train PPO on SceneEnv using a mock world backend (Mac-local dev).

Purpose: exercise the RL training loop end-to-end BEFORE the real world model
backend exists. Same interface — swap MockWorldBackend for RealWorldBackend
(Milestone B) and the training script doesn't change.

Usage:
    python scripts/train_ppo.py \\
        --total_steps 20000 \\
        --output_dir outputs/ppo_mock/

Outputs:
    outputs/ppo_mock/
      ppo_final.zip          — trained policy
      monitor.csv            — episode returns per rollout (SB3 format)
      tensorboard/           — TB logs
      wandb/                 — wandb offline logs (if wandb installed)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# gymnasium + SB3 are required (soft dep of the rest of the repo).
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    from stable_baselines3.common.monitor import Monitor
except ImportError as e:
    raise SystemExit(
        f"stable-baselines3 required: pip install 'stable-baselines3[extra]'. Import failed with: {e}"
    )

from src.env.scene_env import SceneEnv, SceneEnvConfig
from src.env.mock_backend import MockWorldBackend, MockSemanticBackend
from src.eval.reward_2d import RewardWeights


def make_env(seed: int = 0):
    """Factory that builds one SceneEnv + Monitor wrapper."""
    world = MockWorldBackend(H=64, W=64, seed=seed)
    sem = MockSemanticBackend(world)
    cfg = SceneEnvConfig(
        max_steps=100,
        step_size_m=0.3,
        yaw_step_rad=0.3,
        reward=RewardWeights(semantic=1.0, goal=0.5, collision=5.0, step_cost=0.05),
        look_ahead_dist=1.5,
        goal_radius=0.5,
        collision_threshold=0.1,
    )
    env = SceneEnv(world_backend=world, semantic_backend=sem, scene_ids=["mock_scene"], cfg=cfg)
    env.reset(seed=seed)
    return Monitor(env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total_steps", type=int, default=20_000)
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/ppo_mock"))
    ap.add_argument("--n_envs", type=int, default=4,
                    help="parallel envs for rollout collection")
    ap.add_argument("--save_freq", type=int, default=5_000,
                    help="checkpoint every N env steps (across all parallel envs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_wandb", action="store_true",
                    help="enable wandb logging (nav-rl project)")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = args.output_dir / "tensorboard"

    print(f"[train_ppo] making {args.n_envs} parallel envs")
    vec_env = make_vec_env(make_env, n_envs=args.n_envs, seed=args.seed)

    print(f"[train_ppo] building PPO (MultiInputPolicy for Dict obs) ...")
    model = PPO(
        "MultiInputPolicy",   # handles Dict{"rgb", "goal"} observation space
        vec_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(tb_dir),
        # Sane defaults for a small env; tune later.
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    )

    # Wandb: enabled if flag set + wandb installed.
    if args.use_wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            run = wandb.init(
                project="nav-rl",
                name="ppo_mock_smoketest",
                config={"total_steps": args.total_steps, "n_envs": args.n_envs, **model.get_parameters()["policy.optimizer"]},
                sync_tensorboard=True,
                dir=str(args.output_dir),
            )
            callbacks = WandbCallback(model_save_path=str(args.output_dir / "wandb_checkpoints"))
        except Exception as e:
            print(f"[train_ppo] wandb setup failed ({e}); continuing without wandb.")
            callbacks = None
    else:
        callbacks = None

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, args.save_freq // args.n_envs),
        save_path=str(args.output_dir / "checkpoints"),
        name_prefix="ppo",
    )
    all_callbacks = [checkpoint_cb] + ([callbacks] if callbacks else [])

    print(f"[train_ppo] training for {args.total_steps} total env steps ...")
    model.learn(total_timesteps=args.total_steps, callback=all_callbacks, progress_bar=True)

    final_path = args.output_dir / "ppo_final.zip"
    model.save(str(final_path))
    print(f"[train_ppo] saved final model to {final_path}")

    # Quick eval — 5 episodes, print reward mean.
    print(f"[train_ppo] rolling 5 eval episodes with the trained policy ...")
    eval_env = make_env(seed=999)
    returns = []
    for ep in range(5):
        obs, _ = eval_env.reset()
        ep_return = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = eval_env.step(action)
            ep_return += float(r)
            done = terminated or truncated
        returns.append(ep_return)
        print(f"  episode {ep}: return {ep_return:+.3f}  reached_goal={info.get('reached_goal')}  dist={info.get('dist_to_goal'):.2f}")
    print(f"[train_ppo] mean return over 5 eval episodes: {sum(returns)/len(returns):+.3f}")


if __name__ == "__main__":
    main()
