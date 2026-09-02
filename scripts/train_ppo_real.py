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
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


class GoalRadiusCurriculum(BaseCallback):
    """Anneal the goal-capture radius start -> end (advisor spec 2026-08-27).
    Attacks terminal-capture: a 1.0 m disc is reachable by a fresh policy;
    0.5 m is the graduation exam.

    mode="success" (default, Joana 2026-08-28): the radius only SHRINKS once
    the policy wins >= threshold of its recent episodes at the current radius
    (5 cm notches), and holds while it struggles — a timer would starve a
    policy that hasn't learned the big disc yet. mode="time": linear anneal
    over `steps` timesteps (predictable fallback)."""

    def __init__(self, start: float, end: float, steps: int,
                 mode: str = "success", window: int = 100,
                 threshold: float = 0.5, notch: float = 0.05):
        super().__init__()
        self.start, self.end, self.steps = start, end, steps
        self.mode, self.window, self.threshold, self.notch = (
            mode, window, threshold, notch)
        self.r = start
        self._wins: list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:            # episode ended at this step
                self._wins.append(1.0 if info.get("goal_bonus", 0.0) > 0
                                  else 0.0)
        return True

    def _on_rollout_start(self) -> None:
        if self.mode == "time":
            t = min(1.0, self.model.num_timesteps / max(1, self.steps))
            self.r = self.start + (self.end - self.start) * t
        else:
            recent = self._wins[-self.window:]
            if (len(recent) >= self.window // 2
                    and float(np.mean(recent)) >= self.threshold
                    and self.r > self.end):
                self.r = max(self.end, self.r - self.notch)
                self._wins.clear()           # re-earn the next notch
        self.training_env.env_method("set_goal_radius", self.r)
        if self.logger is not None:
            self.logger.record("curriculum/goal_radius", self.r)
            if self._wins:
                self.logger.record("curriculum/recent_success",
                                   float(np.mean(self._wins[-self.window:])))


class GoalDistCurriculum(BaseCallback):
    """Success-gated goal-DISTANCE growth (2026-08-29, Joana's catch: fixed
    6 m goals removed the easy-win bootstrap that made Arm B learn — B's
    frame-range goals included near-spawn gimmes that fed its curriculum).
    Goals start close and grow a notch each time the policy earns >= threshold
    recent wins, up to the target distance. Runs alongside the radius
    curriculum; both gates draw on the same win stream."""

    def __init__(self, start: float, end: float, window: int = 100,
                 threshold: float = 0.5, notch: float = 0.5):
        super().__init__()
        self.d, self.end = start, end
        self.window, self.threshold, self.notch = window, threshold, notch
        self._wins: list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._wins.append(1.0 if info.get("goal_bonus", 0.0) > 0
                                  else 0.0)
        return True

    def _on_rollout_start(self) -> None:
        recent = self._wins[-self.window:]
        if (len(recent) >= self.window // 2
                and float(np.mean(recent)) >= self.threshold
                and self.d < self.end):
            self.d = min(self.end, self.d + self.notch)
            self._wins.clear()               # re-earn the next notch
        self.training_env.env_method("set_goal_dist", self.d)
        if self.logger is not None:
            self.logger.record("curriculum/goal_dist", self.d)


class RewardComponentsCallback(BaseCallback):
    """Log per-component reward means each rollout (meeting item 2026-08-13:
    'visualize different reward vectors on wandb / see the range the reward
    falls in'). Would have exposed Run A's zero-semantic-collision anomaly
    during training instead of at eval."""

    KEYS = ("semantic", "goal", "collision", "step", "void", "spin",
            "backward", "smooth", "timeout", "crash", "proximity",
            "goal_bonus", "total",
            # 2026-09-01: the world model's own uncertainty, logged from day
            # one so any void THRESHOLD gets chosen from the measured
            # distribution instead of guessed. void_frac = footprint support;
            # image_void_frac = whole-view support (her measure). The spin
            # `cov` number is MEAN ALPHA, a different statistic — it cannot be
            # converted into these without the alpha histogram.
            "void_frac", "image_void_frac",
            # 2026-09-02: these were set on `info` by SceneEnv but MISSING from
            # this list, so 460539/460540 ran 17 h with the coherence term
            # inside `total` and invisible everywhere else. The only way to
            # recover its size was to subtract the other components from the
            # total by hand. `coverage` is the quantity the whole mechanism is
            # about -- log it whether or not the term is switched on, so the
            # thresholds get set from the distribution the policy ACTUALLY
            # visits rather than from a spin sweep.
            "coherence", "coherence_crash", "coverage")

    def __init__(self):
        super().__init__()
        self._sums = {k: 0.0 for k in self.KEYS}
        self._cnt = {k: 0 for k in self.KEYS}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "total" not in info:
                continue
            for k in self.KEYS:
                v = float(info.get(k, 0.0))
                # coverage is nan when the backend reports no per-frame alpha;
                # one nan would poison the running mean for the whole rollout.
                if v != v:
                    continue
                self._sums[k] += v
                self._cnt[k] += 1
        return True

    def _on_rollout_end(self) -> None:
        for k in self.KEYS:
            if self._cnt[k]:
                self.logger.record(f"reward/{k}", self._sums[k] / self._cnt[k])
        self._sums = {k: 0.0 for k in self.KEYS}
        self._cnt = {k: 0 for k in self.KEYS}

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
        goal_xy_override=(tuple(float(v) for v in args.goal_xy.split(","))
                          if getattr(args, "goal_xy", None) else None),
        goal_dir_360=getattr(args, "goal_dir_360", False),
        goal_cone_deg=getattr(args, "goal_cone_deg", 360.0),
        goal_dist_range=(tuple(float(v) for v in args.goal_dist_range.split(","))
                         if getattr(args, "goal_dist_range", None) else None),
        spawn_label_classes=(tuple(int(v) for v in args.spawn_classes.split(","))
                             if getattr(args, "spawn_classes", None) else None),
        spawn_yaw_jitter_deg=getattr(args, "spawn_yaw_jitter", 0.0),
        spawn_lat_jitter_m=getattr(args, "spawn_lat_jitter", 0.0),
        sem_palette_version=getattr(args, "sem_palette", 1),
        render_mode="rasterizer_only",       # cheap per-step; diffusion later
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        H=getattr(args, "render_height", None) or getattr(args, "obs_height", 336),
        W=getattr(args, "render_width", None) or getattr(args, "obs_width", 560),
        goal_dist_m=getattr(args, "goal_dist", None),
        spawn_min_frame=getattr(args, "spawn_min", 0),
    )
    if getattr(args, "live", False):
        # Live per-action diffusion: the policy queries the generative model at
        # its own CONTINUOUS pose every step — no cache, no grid snapping.
        # Observations and reward labels come from the same diffusion call.
        from src.env.live_backend import LiveDiffusedBackend
        world = LiveDiffusedBackend(cfg, checkpoint=args.live_ckpt,
                                    live_frames=args.live_frames,
                                    alpha_gate=not getattr(args, "no_alpha_gate", False),
        alpha_gate_tau=getattr(args, "alpha_gate_tau", 0.5))
    elif getattr(args, "obs_cache", None):
        # Ribbon-cache mode: observations = precomputed diffused views (v10 +
        # reader), reward labels = the cache's alpha-masked diffused labels.
        # No GPU rendering during training at all.
        from src.env.cached_backend import CachedDiffusedBackend
        world = CachedDiffusedBackend(cfg, cache_root=args.obs_cache,
                                      alpha_gate=not getattr(args, "no_alpha_gate", False),
        alpha_gate_tau=getattr(args, "alpha_gate_tau", 0.5))
    else:
        world = CalibratedRealWorldBackend(cfg)
    sem = GaussianLabelBackend(world)
    env_cfg = _scene_env_cfg(args)
    _dump_env_config(args, env_cfg)
    env = SceneEnv(world_backend=world, semantic_backend=sem,
                   scene_ids=scenes, cfg=env_cfg)
    return Monitor(env)


def _dump_env_config(args, cfg):
    """Record the env the policy is ACTUALLY learning in, beside its
    checkpoints. Evaluating with different kinematics runs the policy outside
    its own action model — on 2026-09-01 the eval used step 0.25 / yaw 0.3
    against training's 0.30 / 0.50 with reverse re-enabled, and it only
    surfaced when a rollout video showed the robot backing up. A knob needs a
    human to remember; a file does not."""
    try:
        import json as _json
        out = Path(args.output_dir) / "env_config.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_json.dumps({
            "step_size_m": cfg.step_size_m,
            "yaw_step_rad": cfg.yaw_step_rad,
            "forward_only": bool(getattr(cfg, "forward_only", False)),
            "look_ahead_dist": cfg.look_ahead_dist,
            "goal_radius": cfg.goal_radius,
            "collision_threshold": cfg.collision_threshold,
            "collision_terminate_frac": cfg.collision_terminate_frac,
            "collision_terminate_penalty": cfg.collision_terminate_penalty,
            "trav_path": cfg.trav_path,
            "action_chunk": getattr(cfg, "action_chunk", 1),
            "max_steps": cfg.max_steps,
            # Reward too, not just kinematics. Eval built its own weights
            # (semantic 1.0 / goal 1.5 / bonus 50, no reward_scale, no
            # proximity) against training's 5 / 10 / 1000 / 0.01 — so an eval
            # "return" was never on the same scale as anything in training and
            # could not be compared to it. Behaviour is unaffected (the policy
            # does not see the eval reward) but every reported number was.
            "semantic_weight": cfg.reward.semantic,
            "goal_weight": cfg.reward.goal,
            "collision_weight": cfg.reward.collision,
            "step_cost": cfg.reward.step_cost,
            "void_cost": cfg.reward.void_cost,
            "terrain_as_cost": bool(cfg.reward.terrain_as_cost),
            "spin_cost": cfg.spin_cost,
            "backward_cost": cfg.backward_cost,
            "action_smooth_cost": getattr(cfg, "action_smooth_cost", 0.0),
            "goal_bonus": cfg.goal_bonus,
            "timeout_penalty": getattr(cfg, "timeout_penalty", 0.0),
            "timeout_distance_scaled": bool(getattr(cfg, "timeout_distance_scaled", False)),
            "proximity_weight": getattr(cfg, "proximity_weight", 0.0),
            "proximity_margin": getattr(cfg, "proximity_margin", 1.0),
            "proximity_delta": bool(getattr(cfg, "proximity_delta", False)),
            "reward_scale": getattr(cfg, "reward_scale", 1.0),
            "coherence_cost_weight": getattr(cfg, "coherence_cost_weight", 0.0),
            "coherence_tau": getattr(cfg, "coherence_tau", 0.4),
            "coherence_terminate_tau": getattr(cfg, "coherence_terminate_tau", 0.0),
        }, indent=2))
        print(f"[train] env recorded for eval: {out}", flush=True)
    except Exception as e:
        print(f"[train] could not write env_config.json: {e}", flush=True)


def _scene_env_cfg(args):
    # Shaping v2 (approved 2026-07-23): void split from obstacles (mild penalty),
    # collision 5->1, goal pull 0.5->1.5, anti-spin tax, spawn curriculum,
    # goals ~3 real meters. Corrected metric scale (camera 0.25 m): blind zone
    # starts ~0.6 m, so look_ahead 1.5 m is safely visible.
    return SceneEnvConfig(
        max_steps=args.max_steps,
        step_size_m=0.25,                    # matches demo action scale; 0.15 clipped 43% of demos
        yaw_step_rad=0.3,
        reward=RewardWeights(semantic=getattr(args, "semantic_weight", 1.0),
                             goal=getattr(args, "goal_weight", 1.5),
                             collision=1.0,
                             step_cost=0.05, void_cost=0.3,
                             terrain_as_cost=True),          # v4
        look_ahead_dist=1.5,
        goal_radius=getattr(args, "goal_radius", 0.75),
        collision_threshold=0.1,
        spin_cost=getattr(args, "spin_cost", 0.05),
        backward_cost=getattr(args, "backward_cost", 0.0),
        collision_terminate_frac=getattr(args, "collision_terminate_frac", 0.0),
        collision_terminate_penalty=getattr(args, "collision_terminate_penalty", 20.0),
        void_terminate_frac=getattr(args, "void_terminate_frac", 0.0),
        void_terminate_penalty=getattr(args, "void_terminate_penalty", 100.0),
        image_void_terminate_frac=getattr(args, "image_void_terminate_frac", 0.0),
        coherence_cost_weight=getattr(args, "coherence_cost_weight", 0.0),
        coherence_tau=getattr(args, "coherence_tau", 0.4),
        coherence_terminate_tau=getattr(args, "coherence_terminate_tau", 0.0),
        coherence_terminate_penalty=getattr(args, "coherence_terminate_penalty", 100.0),
        proximity_weight=getattr(args, "proximity_weight", 0.0),
        proximity_margin=getattr(args, "proximity_margin", 1.0),
        clouds_dir=getattr(args, "clouds_dir", None),
        action_chunk=getattr(args, "action_chunk", 1),
        footprint_along_motion=getattr(args, "footprint_along_motion", False),
        forward_only=getattr(args, "forward_only", False),
        action_smooth_cost=getattr(args, "action_smooth_cost", 0.0),
        goal_bonus=getattr(args, "goal_bonus", 50.0),        # v4 default
        obs_out_hw=((getattr(args, "obs_height", 336), getattr(args, "obs_width", 560))
                    if (getattr(args, "render_height", None)
                        or getattr(args, "render_width", None)) else None),
        goal_noise_std=getattr(args, "goal_noise_std", 0.0),
        proximity_delta=getattr(args, "proximity_delta", False),
        timeout_penalty=getattr(args, "timeout_penalty", 0.0),
        timeout_distance_scaled=getattr(args, "timeout_distance_scaled", False),
        reward_scale=getattr(args, "reward_scale", 1.0),
        random_spawn=True,
        trav_path=getattr(args, "trav_path", None),
        failure_snap_dir=str(args.output_dir / "failures"),
    )


def make_live_vec_env(args):
    """Batched live training: N robots in ONE scene sharing one pipe. Builds
    the same configs as make_env, then N defer-render SceneEnvs around a
    single BatchedLiveDiffusedBackend and the LiveVecEnv that batches every
    render into one generation call."""
    from src.env.vec_live_env import (
        BatchedLiveDiffusedBackend, InjectedLabelBackend, LiveVecEnv,
    )
    from stable_baselines3.common.vec_env import VecMonitor

    # Multi-scene rotation (2026-08-29): all robots still share ONE resident
    # scene at a time; LiveVecEnv rotates it every --scene_rotate steps.
    scenes = args.scenes if getattr(args, "scenes", None) else [args.scene]
    cfg = CalibratedBackendConfig(
        scene_video_paths={s: f"{args.clips_dir}/{s}.mp4" for s in scenes},
        scene_poses_paths={s: f"{args.poses_dir}/{s}_poses.npz" for s in scenes},
        scene_labels_paths={s: f"{args.labels_dir}/{s}.npz" for s in scenes},
        goal_frame=args.goal_frame,
        goal_frame_range=tuple(args.goal_frame_range) if args.goal_frame_range else None,
        goal_min_sep_m=args.goal_min_sep,
        spawn_max_frame=args.spawn_max_frame,
        goal_xy_override=(tuple(float(v) for v in args.goal_xy.split(","))
                          if getattr(args, "goal_xy", None) else None),
        goal_dir_360=getattr(args, "goal_dir_360", False),
        goal_cone_deg=getattr(args, "goal_cone_deg", 360.0),
        goal_dist_range=(tuple(float(v) for v in args.goal_dist_range.split(","))
                         if getattr(args, "goal_dist_range", None) else None),
        spawn_label_classes=(tuple(int(v) for v in args.spawn_classes.split(","))
                             if getattr(args, "spawn_classes", None) else None),
        spawn_yaw_jitter_deg=getattr(args, "spawn_yaw_jitter", 0.0),
        spawn_lat_jitter_m=getattr(args, "spawn_lat_jitter", 0.0),
        sem_palette_version=getattr(args, "sem_palette", 1),
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        H=getattr(args, "render_height", None) or getattr(args, "obs_height", 336),
        W=getattr(args, "render_width", None) or getattr(args, "obs_width", 560),
        goal_dist_m=getattr(args, "goal_dist", None),
        spawn_min_frame=getattr(args, "spawn_min", 0),
    )
    world = BatchedLiveDiffusedBackend(
        cfg, checkpoint=args.live_ckpt, live_frames=args.live_frames,
        alpha_gate=not getattr(args, "no_alpha_gate", False),
        alpha_gate_tau=getattr(args, "alpha_gate_tau", 0.5))
    world.num_inference_steps = args.live_steps
    # Pre-reconstruct EVERY rotation scene now, while GPU headroom is maximal
    # (2026-08-30, G12 saga: first-visit reconstruction mid-training OOMs at
    # 560-render — training state + the resident scene's splats eat the ~7.3GB
    # the reconstructor needs). Each fresh cache is evicted to CPU immediately
    # so the next reconstruction also sees a clean GPU; rotation later never
    # reconstructs, it only uploads from the cache.
    if len(scenes) > 1:
        import gc as _gc
        import torch as _torch
        from src.env.real_backend import _move_tree_to
        done = []
        for s in scenes:
            free0 = _torch.cuda.mem_get_info()[0] / 1e9
            try:
                world.load_scene(s)
            except _torch.cuda.OutOfMemoryError:
                # 2026-08-30: ~2 GB/scene leaks through eviction at 560-render
                # (measured ladder 32.6->18.7 over 8 scenes). Until the leak is
                # found, survive: train on the scenes that fit, say so loudly.
                print(f"[make_live_vec_env] OOM pre-reconstructing {s} — "
                      f"PRUNING scene list to the {len(done)} that fit: "
                      f"{done}", flush=True)
                break
            world._cache[s] = _move_tree_to(world._cache[s], "cpu")
            # Clear the rasterizer's own splat reference (leak suspect #1) and
            # collect ref cycles before measuring.
            runner = getattr(getattr(getattr(world._reconstructor, "gs_renderer", None),
                                     "rasterizer", None), "runner", None)
            if runner is not None and hasattr(runner, "splats"):
                runner.splats = None
            _gc.collect()
            _torch.cuda.empty_cache()
            free1 = _torch.cuda.mem_get_info()[0] / 1e9
            done.append(s)
            print(f"[make_live_vec_env] pre-reconstructed {s}: free "
                  f"{free0:.1f} -> {free1:.1f} GB (leak {free0 - free1:.2f})",
                  flush=True)
        if done and len(done) < len(scenes):
            scenes = done
    world.load_scene(scenes[0])

    envs = []
    for _ in range(args.live_batch):
        env_cfg = _scene_env_cfg(args)
        env_cfg.defer_render = True
        e = SceneEnv(world_backend=world, semantic_backend=None,
                     scene_ids=scenes, cfg=env_cfg)
        e.semantic_backend = InjectedLabelBackend(e)
        envs.append(e)
    return VecMonitor(LiveVecEnv(envs, world, scenes=scenes,
                                 rotate_every=getattr(args, "scene_rotate", 0)))


def save_rollout_video(model, env, out_path: Path, max_frames=120):
    """Eval rollout with a HUD (step/action/reward/dist) + top-down map inset:
    agent path (white), recorded real trajectory (gray), goal (green). Makes
    off-manifold wandering legible — black frames = outside the reconstructed
    volume, and the map shows exactly where the agent is relative to the trail."""
    import imageio.v3 as iio
    from PIL import Image, ImageDraw
    from src.eval.palette import CLASS_COLORS_V14_255
    from src.eval.reward_2d import (
        _footprint_corners_world, _project_points, GO2_BODY_LENGTH, GO2_BODY_WIDTH,
    )

    def _colorize(lab, H, W, tag, footprint_uv=None):
        import cv2
        # v14 runs carry ids 0-13; legacy raster runs carry the 30-class ids —
        # pick the palette that actually matches the ids or colors lie.
        from src.eval.palette import CLASS_COLORS_255
        pal = CLASS_COLORS_V14_255 if int(np.max(lab)) < 14 else CLASS_COLORS_255
        col = pal[np.clip(lab, 0, len(pal) - 1)]
        if col.shape[:2] != (H, W):
            col = cv2.resize(col, (W, H), interpolation=cv2.INTER_NEAREST)
        col = col.copy()
        if footprint_uv is not None:
            # the exact quad the reward scores this step (look-ahead footprint)
            cv2.polylines(col, [footprint_uv.astype(np.int32)], True,
                          (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(col, tag, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return col

    def _footprint_uv(base_env, world):
        """Project the reward's footprint quad into the CURRENT view. Same math
        as compute_reward: corners from pose+heading, camera from the cached
        view actually served (so the box lands where the reward truly read)."""
        try:
            pose = base_env._robot_pose_world
            pos = pose[:3, 3]
            hd = getattr(base_env, "_last_fp_heading", None)
            if hd is None:
                hd = pose[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float64)
            corners = _footprint_corners_world(
                pos, hd, look_ahead_dist=base_env.cfg.look_ahead_dist,
                length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
            uv, in_front = _project_points(corners, base_env._last_K, base_env._last_w2c)
            if not in_front.all():
                return None
            return uv
        except Exception:
            return None

    def semantic_panel(world, H, W, footprint_uv=None):
        """Colorized semantics: what the reward reads, and (gated runs) also
        the model's RAW belief before voiding — the semantics-model test view."""
        used = getattr(world, "_last_semantic_image", None)
        if used is None:
            return None
        raw = getattr(world, "_last_semantic_raw", None)
        panels = []
        if raw is not None and raw is not used:
            panels.append(_colorize(raw, H, W, "sem RAW (model belief)", footprint_uv))
        panels.append(_colorize(used, H, W, "sem REWARD (gated)" if len(panels) else "sem REWARD", footprint_uv))
        return np.concatenate(panels, axis=1)

    base_env = env.unwrapped if hasattr(env, "unwrapped") else env
    frames, path_xy, rows = [], [], []
    obs, _ = env.reset()          # MUST precede calib access: reset loads the scene

    def _xyyaw():
        P = base_env._robot_pose_world
        return [round(float(P[0, 3]), 3), round(float(P[1, 3]), 3),
                round(float(np.arctan2(P[1, 0], P[0, 0])), 3)]
    traj = [_xyyaw() + [0]]       # [x, y, yaw, collision] — same format as eval
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
        # frame, the pose/obs frames are inconsistent — the suspected frame bug.
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
                          f"goal={np.degrees(gr[2]):+.0f}deg"
                          + (f" cache={world._last_lookup[0]*100:.0f}cm/"
                             f"{world._last_lookup[1]:.0f}deg"
                             if getattr(world, "_last_lookup", None) else ""),
                  fill=(255, 255, 255, 255))
        frame = np.array(img.convert("RGB"))
        fp_uv = _footprint_uv(base_env, world)
        if fp_uv is not None:
            import cv2 as _cv
            _cv.polylines(frame, [fp_uv.astype(np.int32)], True, (255, 255, 0), 2, _cv.LINE_AA)
        sem = semantic_panel(world, H, W, fp_uv)
        if sem is not None:
            frame = np.concatenate([frame, sem], axis=1)
        frames.append(frame)

        obs, r, terminated, truncated, info = env.step(action)
        traj.append(_xyyaw() +
                    [round(float(max(0.0, -info.get("collision", 0.0))), 3)])
        done = terminated or truncated

    # Arrival frame: the loop above draws BEFORE stepping, so the terminal pose
    # (entering the goal disc) was never rendered — append it.
    final = Image.fromarray(env.render().copy())
    fd = ImageDraw.Draw(final, "RGBA")
    fd.rectangle([0, 0, final.width, 14], fill=(0, 0, 0, 180))
    fd.text((4, 2), f"t={len(frames):3d} DONE reached_goal={info.get('reached_goal')} "
                    f"r={float(r):+.2f}", fill=(0, 255, 0, 255))
    last = np.array(final.convert("RGB"))
    sem = semantic_panel(world, last.shape[0], last.shape[1])
    if sem is not None:
        last = np.concatenate([last, sem], axis=1)
    frames.append(last)

    iio.imwrite(str(out_path), np.stack(frames), fps=8,
                codec="libx264", macro_block_size=1,
                ffmpeg_params=["-pix_fmt", "yuv420p"])
    print(f"[train_ppo_real] rollout video: {out_path} "
          f"({len(frames)} frames, reached_goal={info.get('reached_goal')})")
    return {"traj": traj,
            "goal_xy": [round(float(goal[0]), 3), round(float(goal[1]), 3)],
            "success": bool(info.get("reached_goal", False))}


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
    ap.add_argument("--live", action="store_true",
                    help="live per-action diffusion observations (no cache): one "
                         "5-frame generation per env step at the exact pose")
    ap.add_argument("--live_ckpt",
                    default="/scratch/m000204-pm06b/joana/runs/train_semantic_v10/checkpoint-epoch-30.safetensors",
                    help="semantic diffusion checkpoint the live backend serves")
    ap.add_argument("--live_frames", type=int, default=5,
                    help="frames per live call (4 history + current; 4k+1)")
    ap.add_argument("--warmstart", type=Path, default=None,
                    help="PPO checkpoint .zip to continue training from "
                         "(num_timesteps preserved; total_steps counts the NEW steps)")
    ap.add_argument("--void_terminate_frac", type=float, default=0.0,
                    help="end the episode when this fraction of the footprint "
                         "has no gaussian support (alpha-gated void) — stops "
                         "the policy exploiting the world model's blind spots")
    ap.add_argument("--image_void_terminate_frac", type=float, default=0.0,
                    help="end the episode when this fraction of the WHOLE "
                         "IMAGE has no gaussian support (= 1 - alpha coverage "
                         "with the gate on). Her measure: unsupported ground "
                         "is fine while the view stays paintable")
    ap.add_argument("--void_terminate_penalty", type=float, default=100.0,
                    help="cost applied on void termination (RW5 scale: a real "
                         "crash is 1000)")
    ap.add_argument("--timeout_distance_scaled", action="store_true",
                    help="scale the timeout penalty by remaining/initial "
                         "distance, so stopping close is cheap and stopping "
                         "far is not — makes 'approach as close as legally "
                         "possible and hold' the optimum without defining a "
                         "boundary")
    ap.add_argument("--coherence_cost_weight", type=float, default=0.0,
                    help="graded cost w*max(0, tau - coverage). Coverage is "
                         "mean alpha: how much of the render has real geometry "
                         "behind it. 0 = off")
    ap.add_argument("--coherence_tau", type=float, default=0.4,
                    help="read off the coverage ladder by eye, 2026-09-01")
    ap.add_argument("--coherence_terminate_tau", type=float, default=0.0,
                    help="end the episode below this coverage — frames with no "
                         "geometry at all. 0 = off")
    ap.add_argument("--coherence_terminate_penalty", type=float, default=100.0)
    ap.add_argument("--alpha_gate_tau", type=float, default=0.5,
                    help="alpha below this is relabelled void in the REWARD's "
                         "label map (never in the observation). Measured "
                         "2026-09-01: all phantom crashes gone by 0.1")
    ap.add_argument("--no_alpha_gate", action="store_true",
                    help="UNGATED reward: trust diffused labels in invented regions too "
                         "(coherence-justified; the gated run is the safety-anchored twin)")
    ap.add_argument("--collision_terminate_frac", type=float, default=0.0,
                    help="footprint overlap that ENDS the episode (0 = off). "
                         "Without it a whole episode of walking through a tree "
                         "costs ~1%% of the goal bonus, so crashing is optimal.")
    ap.add_argument("--collision_terminate_penalty", type=float, default=20.0)
    ap.add_argument("--backward_cost", type=float, default=0.0,
                    help="penalty * max(0,-v); tax the camera-blind backing gait")
    ap.add_argument("--footprint_along_motion", action="store_true",
                    help="score the footprint along the commanded motion "
                         "direction; motion onto unseen (rear) ground prices "
                         "as worst-case terrain instead of scoring free")
    ap.add_argument("--forward_only", action="store_true",
                    help="clamp negative velocity to 0 (stand-and-turn): kills "
                         "the backward-exploit class by construction")
    ap.add_argument("--spin_cost", type=float, default=0.05,
                    help="penalty * |yaw action| per step; raise to break the "
                         "bang-bang full-lock turning habit")
    ap.add_argument("--action_smooth_cost", type=float, default=0.0,
                    help="penalty * mean|a_t - a_{t-1}|; charges action "
                         "CHANGES (flip-flops), not turning itself")
    ap.add_argument("--goal_bonus", type=float, default=50.0)
    ap.add_argument("--goal_radius", type=float, default=0.75)
    ap.add_argument("--goal_radius_start", type=float, default=None,
                    help="curriculum: capture radius anneals from this down "
                         "to --goal_radius over --curriculum_steps")
    ap.add_argument("--curriculum_steps", type=int, default=100_000)
    ap.add_argument("--curriculum_mode", default="success",
                    choices=["success", "time"],
                    help="success: radius shrinks only when the policy earns "
                         "it (>=50%% recent wins); time: linear anneal")
    ap.add_argument("--goal_dist", type=float, default=None,
                    help="fixed spawn->goal distance in meters (advisor spec)")
    ap.add_argument("--encoder", default="nature",
                    choices=["nature", "dinov2", "resnet18", "both"],
                    help="policy visual encoder: SB3 NatureCNN (scratch) or "
                         "a frozen pretrained backbone (advisor ablation)")
    ap.add_argument("--goal_dist_start", type=float, default=None,
                    help="distance curriculum: goals start here and grow to "
                         "--goal_dist as the policy earns wins")
    ap.add_argument("--obs_height", type=int, default=336,
                    help="render/observation height (multiple of 112)")
    ap.add_argument("--obs_width", type=int, default=560,
                    help="render/observation width (multiple of 112)")
    ap.add_argument("--render_height", type=int, default=None,
                    help="render-high/observe-small: diffusion renders at this "
                         "height (multiple of 112) and obs['rgb'] is downsized "
                         "to --obs_height. Reward/labels stay at render size.")
    ap.add_argument("--render_width", type=int, default=None,
                    help="see --render_height")
    ap.add_argument("--live_steps", type=int, default=4,
                    help="diffusion sampler steps for live generation")
    ap.add_argument("--reward_scale", type=float, default=1.0,
                    help="uniform reward multiplier (e.g. 0.01 tames the "
                         "critic under +-1000 terminals; ratios preserved)")
    ap.add_argument("--spawn_min", type=int, default=0,
                    help="min spawn frame: keep out of the weak-recon clip "
                         "edges where live generation confabulates")
    ap.add_argument("--scene_rotate", type=int, default=0,
                    help="live multi-scene: rotate the resident world every "
                         "N robot-steps (0 = never; needs --scenes)")
    ap.add_argument("--goal_weight", type=float, default=1.5,
                    help="progress-to-goal shaping weight")
    ap.add_argument("--goal_dir_360", action="store_true",
                    help="J-spec: goals at random 360-degree bearing and "
                         "random distance (--goal_dist_range) from spawn; may "
                         "land on non-traversable ground BY DESIGN")
    ap.add_argument("--goal_dist_range", default=None,
                    help="'lo,hi' meters for --goal_dir_360 (e.g. 5,10)")
    ap.add_argument("--goal_cone_deg", type=float, default=360.0,
                    help="constrain --goal_dir_360 bearings to +-this/2 of "
                         "the path tangent (single-pass capture only renders "
                         "a forward cone; 360 once pano scenes exist)")
    ap.add_argument("--spawn_classes", default=None,
                    help="comma class ids; only spawn on frames whose ground "
                         "patch is one of these (e.g. 6,8 = sidewalk/pavement)")
    ap.add_argument("--spawn_yaw_jitter", type=float, default=0.0,
                    help="rotate each spawn heading by U(-x,+x) degrees — "
                         "anti-memorization: stop-at-grass, not stop-at-"
                         "deviation (her J-v2 spec)")
    ap.add_argument("--spawn_lat_jitter", type=float, default=0.0,
                    help="slide each spawn laterally by U(-x,+x) meters")
    ap.add_argument("--sem_palette", type=int, default=1,
                    help="v14 palette version for the live semantic decode — "
                         "must match --live_ckpt's training palette "
                         "(v21=1, v24/v25 line=4)")
    ap.add_argument("--goal_xy", default=None,
                    help="'x,y': FIX the goal at this nav-frame point every "
                         "episode — obstacle-encounter training (2026-08-31, "
                         "Jing's non-traversable-areas directive): goals "
                         "placed behind an obstacle force avoidance learning")
    ap.add_argument("--goal_noise_std", type=float, default=0.0,
                    help="std (m) of Gaussian noise on the goal-vector obs xy "
                         "each step — the anti-odometry lever; reward and "
                         "termination still use the true goal")
    ap.add_argument("--semantic_weight", type=float, default=1.0,
                    help="terrain/semantic footprint weight (0 disables)")
    ap.add_argument("--timeout_penalty", type=float, default=0.0,
                    help="subtracted at max_steps without goal — makes "
                         "freezing strictly worse than trying")
    ap.add_argument("--proximity_delta", action="store_true",
                    help="potential-shaped proximity: weight*(d_prev - d_now) "
                         "inside the margin; approaching costs, retreating "
                         "refunds, standing is free")
    ap.add_argument("--proximity_weight", type=float, default=0.0,
                    help="cost/step for being within proximity_margin of a "
                         "GEOMETRIC obstacle (scene cloud) — undreamable, "
                         "unlike the semantic footprint. 0 = off")
    ap.add_argument("--proximity_margin", type=float, default=1.0)
    ap.add_argument("--clouds_dir", default=None,
                    help="dir with <scene>_cloud.npz from dump_scene_cloud.py "
                         "(required for --proximity_weight > 0)")
    ap.add_argument("--action_chunk", type=int, default=1,
                    help="policy outputs k action pairs per decision and only "
                         "re-observes after all k execute (trajectory arm; "
                         "1 = per-action)")
    ap.add_argument("--trav_path", default=None,
                    help="traversability yaml override (config/traversability_v14.yaml for cached runs)")
    ap.add_argument("--goal_min_sep", type=float, default=1.0,
                    help="spawn-goal separation guard in meters (v6 used 1.5)")
    ap.add_argument("--target_kl", type=float, default=0.02,
                    help="PPO KL trust region; 0 disables (rung 7b)")
    ap.add_argument("--scenes", nargs="+", default=None,
                    help="rung 7: train across these scenes (overrides --scene)")
    ap.add_argument("--live_batch", type=int, default=1,
                    help="N robots sharing one pipe via batched generation "
                         "(live mode only; 1 = the classic single-robot path)")
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
    if getattr(args, "live", False) and getattr(args, "live_batch", 1) > 1:
        env = make_live_vec_env(args)
        if args.bc_demos:
            print("[live_batch] BC pretrain unsupported in batched mode — skipping")
            args.bc_demos = None
    else:
        env = make_env(args)

    if args.warmstart is not None:
        # Continue training an existing policy (e.g. the cache-trained champion
        # fine-tuning on live observations). SB3 keeps num_timesteps; learn()
        # below adds total_steps on top (reset_num_timesteps=False).
        model = PPO.load(str(args.warmstart), env=env,
                         tensorboard_log=str(args.output_dir / "tensorboard"))
        # The checkpoint carries its own target_kl — but a warm-start into a NEW
        # setting (new scenes / new obs source) shifts the policy hard on the
        # first updates, and the old leash aborts every round at step 0 (Run C
        # 2026-08-23: 300k steps, zero learning). Honor the CLI value instead.
        model.target_kl = (args.target_kl if args.target_kl > 0 else None)
        print(f"[train_ppo_real] warm-start from {args.warmstart} "
              f"(num_timesteps={model.num_timesteps}, target_kl={model.target_kl})")
    else:
        policy_kwargs = dict(net_arch=[dict(pi=[64, 64], vf=[64, 64])])
        if getattr(args, "encoder", "nature") != "nature":
            from src.policy.encoders import FrozenBackboneExtractor
            policy_kwargs.update(
                features_extractor_class=FrozenBackboneExtractor,
                features_extractor_kwargs=dict(backbone=args.encoder))
        model = PPO(
        "MultiInputPolicy", env,
        policy_kwargs=policy_kwargs,
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
    )

    if args.bc_demos is not None:
        bc_pretrain(model, args.bc_demos, epochs=args.bc_epochs)
        model.save(str(args.output_dir / "policy_after_bc.zip"))

    callbacks = [CheckpointCallback(save_freq=2_000,
                                    save_path=str(args.output_dir / "checkpoints"),
                                    name_prefix="ppo"),
                 RewardComponentsCallback()]
    if args.goal_radius_start is not None:
        callbacks.append(GoalRadiusCurriculum(
            args.goal_radius_start, args.goal_radius, args.curriculum_steps,
            mode=args.curriculum_mode))
    if args.goal_dist_start is not None and args.goal_dist is not None:
        callbacks.append(GoalDistCurriculum(
            args.goal_dist_start, args.goal_dist))
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
    model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=True,
                reset_num_timesteps=(args.warmstart is None))
    model.save(str(args.output_dir / "ppo_final.zip"))

    print("[train_ppo_real] eval rollout ...")
    if getattr(args, "live_batch", 1) > 1:
        print("[live_batch] end-of-training rollout video skipped (vec env)")
    else:
        save_rollout_video(model, env, args.output_dir / "rollout.mp4")


if __name__ == "__main__":
    main()
