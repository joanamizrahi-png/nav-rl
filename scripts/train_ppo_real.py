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
    # rung 7: --scenes trains one policy over several scenes (round-robin per
    # episode in SceneEnv). Single --scene remains the default path.
    scenes = args.scenes if getattr(args, "scenes", None) else [args.scene]
    cfg = CalibratedBackendConfig(
        scene_video_paths={s: f"{args.clips_dir}/{s}.mp4" for s in scenes},
        scene_poses_paths={s: f"{args.poses_dir}/{s}_poses.npz" for s in scenes},
        scene_labels_paths={s: f"{args.labels_dir}/{s}.npz" for s in scenes},
        goal_frame=args.goal_frame,
        goal_frame_range=tuple(args.goal_frame_range) if args.goal_frame_range else None,
        goal_min_sep_m=args.goal_min_sep,
        spawn_max_frame=args.spawn_max_frame,
        render_mode="rasterizer_only",       # cheap per-step; diffusion later
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
    )
    if getattr(args, "obs_cache", None):
        # Ribbon-cache mode: observations = precomputed diffused views (v10 +
        # reader), reward labels = the cache's alpha-masked diffused labels.
        # No GPU rendering during training at all.
        from src.env.cached_backend import CachedDiffusedBackend
        world = CachedDiffusedBackend(cfg, cache_root=args.obs_cache)
    else:
        world = CalibratedRealWorldBackend(cfg)
    sem = GaussianLabelBackend(world)
    # Shaping v2 (approved 2026-07-23): void split from obstacles (mild penalty),
    # collision 5->1, goal pull 0.5->1.5, anti-spin tax, spawn curriculum,
    # goals ~3 real meters. Corrected metric scale (camera 0.25 m): blind zone
    # starts ~0.6 m, so look_ahead 1.5 m is safely visible.
    env_cfg = SceneEnvConfig(
        max_steps=args.max_steps,
        step_size_m=0.25,                    # matches demo action scale; 0.15 clipped 43% of demos
        yaw_step_rad=0.3,
        reward=RewardWeights(semantic=1.0, goal=1.5, collision=1.0,
                             step_cost=0.05, void_cost=0.3,
                             terrain_as_cost=True),          # v4
        look_ahead_dist=1.5,
        goal_radius=0.75,
        collision_threshold=0.1,
        spin_cost=0.05,
        goal_bonus=50.0,                                     # v4
        random_spawn=True,
        trav_path=getattr(args, "trav_path", None),
    )
    env = SceneEnv(world_backend=world, semantic_backend=sem,
                   scene_ids=scenes, cfg=env_cfg)
    return Monitor(env)


def save_rollout_video(model, env, out_path: Path, max_frames=120):
    """Eval rollout with a HUD (step/action/reward/dist) + top-down map inset:
    agent path (white), recorded real trajectory (gray), goal (green). Makes
    off-manifold wandering legible — black frames = outside the reconstructed
    volume, and the map shows exactly where the agent is relative to the trail."""
    import imageio.v3 as iio
    from PIL import Image, ImageDraw

    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    frames, path_xy, rows = [], [], []
    obs, _ = env.reset()          # MUST precede calib access: reset loads the scene
    world = base_env.world_backend
    cal = world._calib[base_env.scene_ids[0]] if hasattr(world, "_calib") else None
    ref_traj = cal.positions[:, :2] if cal is not None else None
    goal = base_env._goal_world[:2].copy()
    done, r, info = False, 0.0, {}
    while not done and len(frames) < max_frames:
        pose = base_env._robot_pose_world
        path_xy.append(pose[:2, 3].copy())
        action, _ = model.predict(obs, deterministic=True)

        img = Image.fromarray(env.render().copy())
        draw = ImageDraw.Draw(img, "RGBA")
        H, W = img.height, img.width

        # --- top-down map inset (bottom-right), fixed world window ---
        m = 110
        all_pts = np.array(path_xy + ([goal] if goal is not None else []))
        if ref_traj is not None:
            all_pts = np.vstack([all_pts, ref_traj])
        lo, hi = all_pts.min(0) - 1.0, all_pts.max(0) + 1.0
        span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-3))
        def to_px(p):
            return (W - m - 6 + (p[0] - lo[0]) / span * m,
                    H - 6 - (p[1] - lo[1]) / span * m)
        draw.rectangle([W - m - 10, H - m - 10, W - 2, H - 2], fill=(0, 0, 0, 170))
        if ref_traj is not None:
            draw.line([to_px(p) for p in ref_traj[::4]], fill=(160, 160, 160, 255), width=1)
        if len(path_xy) > 1:
            draw.line([to_px(p) for p in path_xy], fill=(255, 255, 255, 255), width=2)
        gx, gy = to_px(goal)
        # success disc (goal_radius) so terminations look right, then the dot
        rr = base_env.cfg.goal_radius / span * m
        draw.ellipse([gx - rr, gy - rr, gx + rr, gy + rr],
                     outline=(0, 255, 0, 200), width=1)
        draw.ellipse([gx - 3, gy - 3, gx + 3, gy + 3], fill=(0, 255, 0, 255))
        ax, ay = to_px(path_xy[-1])
        draw.ellipse([ax - 2, ay - 2, ax + 2, ay + 2], fill=(255, 80, 80, 255))
        # heading arrow: where the robot LOOKS (local +x in world), 0.8 m long.
        # If this arrow ever disagrees with where the white path advances next
        # frame, the pose/obs frames are inconsistent — the bug Jing suspects.
        hd = pose[:2, 0] / (np.linalg.norm(pose[:2, 0]) + 1e-9)
        hx, hy = to_px(path_xy[-1] + hd * 0.8)
        draw.line([(ax, ay), (hx, hy)], fill=(255, 160, 0, 255), width=2)

        # goal-bearing compass on the FPV: obs dyaw, 0 = dead ahead (up),
        # positive = goal to the LEFT. The needle should swing to center as
        # the policy turns toward the goal.
        gr = base_env._goal_in_robot_frame()
        cx, cy = W // 2, H - 22
        tipx = cx - 16 * np.sin(gr[2])
        tipy = cy - 16 * np.cos(gr[2])
        draw.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=(0, 0, 0, 130))
        draw.line([(cx, cy), (tipx, tipy)], fill=(0, 255, 0, 255), width=2)
        draw.ellipse([tipx - 2, tipy - 2, tipx + 2, tipy + 2], fill=(0, 255, 0, 255))

        # --- HUD banner ---
        draw.rectangle([0, 0, W, 14], fill=(0, 0, 0, 180))
        draw.text((4, 2), f"t={len(frames):3d} v={float(action[0]):+.2f} "
                          f"w={float(action[1]):+.2f} r={float(r):+.2f} "
                          f"dist={info.get('dist_to_goal', float('nan')):.1f}m "
                          f"goal={np.degrees(gr[2]):+.0f}deg",
                  fill=(255, 255, 255, 255))
        frames.append(np.array(img.convert("RGB")))

        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Arrival frame: the loop above draws BEFORE stepping, so the terminal pose
    # (entering the goal disc) was never rendered — append it.
    final = Image.fromarray(env.render().copy())
    fd = ImageDraw.Draw(final, "RGBA")
    fd.rectangle([0, 0, final.width, 14], fill=(0, 0, 0, 180))
    fd.text((4, 2), f"t={len(frames):3d} DONE reached_goal={info.get('reached_goal')} "
                    f"r={float(r):+.2f}", fill=(0, 255, 0, 255))
    frames.append(np.array(final.convert("RGB")))

    iio.imwrite(str(out_path), np.stack(frames), fps=8,
                codec="libx264", macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"[train_ppo_real] rollout video: {out_path} "
          f"({len(frames)} frames, reached_goal={info.get('reached_goal')})")


def bc_pretrain(model, demos_path: Path, epochs: int = 25, batch_size: int = 64):
    """Behavior-clone the PPO policy on real-trajectory demonstrations.

    Maximizes log-prob of demo actions under the policy (SB3 evaluate_actions
    handles preprocessing internally, incl. image normalization). The policy
    then enters PPO already knowing roughly how to drive a trail.
    """
    import torch as th
    d = np.load(demos_path, allow_pickle=True)
    obs_rgb = d["obs"]                      # [N,H,W,3] uint8
    goal = d["goal"].astype(np.float32)
    act = d["act"].astype(np.float32)
    n = len(act)
    device = model.policy.device
    # Model observation space is channel-first (SB3 VecTransposeImage).
    rgb_chw = np.transpose(obs_rgb, (0, 3, 1, 2))
    print(f"[bc] {n} demos, {epochs} epochs", flush=True)
    opt = th.optim.Adam(model.policy.parameters(), lr=3e-4)
    idx = np.arange(n)
    for ep in range(epochs):
        np.random.shuffle(idx)
        losses = []
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            obs_t = {
                "rgb": th.as_tensor(rgb_chw[b]).to(device),
                "goal": th.as_tensor(goal[b]).to(device),
            }
            act_t = th.as_tensor(act[b]).to(device)
            _, log_prob, entropy = model.policy.evaluate_actions(obs_t, act_t)
            loss = -log_prob.mean() - 1e-3 * entropy.mean()
            opt.zero_grad(); loss.backward()
            th.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            opt.step()
            losses.append(float(loss))
        print(f"[bc] epoch {ep+1}/{epochs} loss {np.mean(losses):.3f}", flush=True)


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
    ap.add_argument("--spawn_max_frame", type=int, default=None,
                    help="cap spawn range; 3 = full traverses from the trail start (rung 5)")
    ap.add_argument("--goal_frame", type=int, default=30,
                    help="goal = real-trajectory position at this frame (~30 => ~3-4 real m)")
    ap.add_argument("--goal_frame_range", type=int, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="rung 6: sample the goal frame per episode from [LO, HI]")
    ap.add_argument("--obs_cache", default=None,
                    help="ribbon-cache root (outputs/ribbon_cache); enables cached diffused observations")
    ap.add_argument("--trav_path", default=None,
                    help="traversability yaml override (config/traversability_v14.yaml for cached runs)")
    ap.add_argument("--goal_min_sep", type=float, default=1.0,
                    help="spawn-goal separation guard in meters (v6 used 1.5)")
    ap.add_argument("--target_kl", type=float, default=0.02,
                    help="PPO KL trust region; 0 disables (rung 7b)")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="rung 7: train across these scenes (overrides --scene)")
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/ppo_real"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--bc_demos", type=Path, default=None,
                    help="npz from make_demo_dataset.py; if set, behavior-clone the "
                         "policy on real-trajectory demonstrations before PPO")
    ap.add_argument("--bc_epochs", type=int, default=25)
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
        # v6c: constant lr again. v6b's linear decay produced a NEVER-learned
        # policy (0% at 200k/306k/400k, goal-blind) — the random-goal task needs
        # full-rate learning throughout (v6 only found its peak at 306k). The KL
        # trust region stays as the sole collapse protection: it triggered just
        # 4x in v6b (harmless) yet caps the exact mechanism behind 3/3 observed
        # peak-then-collapse runs.
        learning_rate=1e-4,
        # target_kl: 0 disables. Single-scene: 0.02 harmless (4-110 triggers).
        # Multi-scene v7: 0.02 fired on 77% of update rounds (4837x) and
        # strangled learning -> rung 7b runs unleashed.
        target_kl=(args.target_kl if args.target_kl > 0 else None),
        gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.01,
        policy_kwargs=dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])]),
    )

    if args.bc_demos is not None:
        bc_pretrain(model, args.bc_demos, epochs=args.bc_epochs)
        model.save(str(args.output_dir / "policy_after_bc.zip"))

    callbacks = [CheckpointCallback(save_freq=2_000,
                                    save_path=str(args.output_dir / "checkpoints"),
                                    name_prefix="ppo")]
    if args.use_wandb:
        try:
            import wandb
            from wandb.integration.sb3 import WandbCallback
            wandb.init(project="nav-rl", name=args.output_dir.name,
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
