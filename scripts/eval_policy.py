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
Outputs: metrics.json (incl. per-step [x, y, yaw, collision] trajectories for
top-down plotting via scripts/make_topdown_figure.py), eval_summary printed,
rollout videos for the first 3 episodes.
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
        goal_xy_override=(tuple(float(v) for v in args.goal_xy.split(","))
                          if args.goal_xy else None),
        # 2026-09-02: eval could ONLY use a fixed goal_xy or the default goal
        # frame, while every J arm trains on goals sampled in a cone at a
        # random distance from the spawn. So the eval had never once
        # reproduced the goal distribution the policy learned -- with a fixed
        # goal and a wide spawn range, d_start varied uncontrolled and some
        # episodes were unreachable inside the step budget before the policy
        # acted. Pass --goal_dir_360 to sample goals exactly as training does.
        goal_dir_360=args.goal_dir_360,
        # spawn jitter lives on the BACKEND config (it is applied in
        # sample_start_pose), not on SceneEnvConfig. Passing it to the wrong
        # dataclass killed six evals in four seconds on 2026-09-03.
        spawn_yaw_jitter_deg=getattr(args, "spawn_yaw_jitter", 0.0),
        spawn_lat_jitter_m=getattr(args, "spawn_lat_jitter", 0.0),
        goal_dist_range=(tuple(float(v) for v in args.goal_dist_range.split(","))
                         if args.goal_dist_range else None),
        goal_cone_deg=args.goal_cone_deg,
        goal_frame_range=(tuple(int(v) for v in args.goal_frame_range.split(","))
                          if args.goal_frame_range else None),
        spawn_max_frame=args.spawn_max_frame,
        spawn_min_frame=args.spawn_min_frame,
        render_mode="rasterizer_only",
        model_path=args.model_path,
        reconstructor_path=args.reconstructor_path,
        H=args.render_height or args.obs_height,
        W=args.render_width or args.obs_width,
    )
    if args.live:
        # Live-trained policies (--live runs) must be evaluated on live
        # observations too — cache lookups would be an obs-distribution shift.
        from src.env.live_backend import LiveDiffusedBackend
        world = LiveDiffusedBackend(
            cfg, checkpoint=args.live_ckpt,
            alpha_gate=not args.no_alpha_gate,
            alpha_gate_tau=float(getattr(args, "alpha_gate_tau", None) or 0.5))
    elif args.obs_cache:
        from src.env.cached_backend import CachedDiffusedBackend
        world = CachedDiffusedBackend(cfg, args.obs_cache,
                                      alpha_gate=not args.no_alpha_gate,
                                      sweep_switch_penalty_m=args.sweep_sticky)
    else:
        world = CalibratedRealWorldBackend(cfg)
    sem = GaussianLabelBackend(world)
    env_cfg = SceneEnvConfig(
        goal_support_radius_m=args.goal_support_radius,
        goal_support_min_frac=args.goal_support_min_frac,
        collision_look_ahead_m=args.collision_look_ahead,
        # GND/SCAND clips advance ~1 m per recorded frame vs RUGD's ~0.1 m, so
        # the same goal_frame is a far longer walk there — raise the budget
        # instead of moving the goal (moving it collapses the spawn range).
        # KINEMATICS MUST MATCH TRAINING (2026-09-01). These were hardcoded to
        # 0.25 / 0.3 while every J arm trains at 0.30 / 0.50, so the policy was
        # evaluated with 17% shorter steps and 40% weaker turning than the
        # action model it learned. Combined with forward_only defaulting off —
        # RW5 clamps reverse away during training — the robot could also drive
        # backwards in eval, which it never could while learning.
        max_steps=args.max_steps,
        step_size_m=args.step_size_m, yaw_step_rad=args.yaw_step_rad,
        reward=RewardWeights(
            semantic=(args.semantic_weight if getattr(args, "semantic_weight", None) is not None else 1.0),
            goal=getattr(args, "goal_weight", 1.5),
            collision=getattr(args, "collision_weight", 1.0),
            step_cost=getattr(args, "step_cost", 0.05),
            void_cost=getattr(args, "void_cost", 0.3),
            terrain_as_cost=bool(getattr(args, "terrain_as_cost", True))),
        # read from args so the adoption above actually takes effect
        look_ahead_dist=getattr(args, "look_ahead_dist", 1.5),
        goal_radius=(args.goal_radius if getattr(args, "goal_radius", None) is not None else 0.75),
        collision_threshold=getattr(args, "collision_threshold", 0.1),
        spin_cost=(getattr(args, "spin_cost", None) if getattr(args, "spin_cost", None) is not None else 0.05),
        backward_cost=getattr(args, "backward_cost", 0.0),
        action_smooth_cost=(getattr(args, "action_smooth_cost", None) if getattr(args, "action_smooth_cost", None) is not None else 0.0),
        goal_bonus=getattr(args, "goal_bonus", 50.0),
        timeout_penalty=getattr(args, "timeout_penalty", 0.0),
        proximity_weight=getattr(args, "proximity_weight", 0.0),
        proximity_margin=getattr(args, "proximity_margin", 1.0),
        proximity_delta=bool(getattr(args, "proximity_delta", False)),
        timeout_distance_scaled=bool(
            getattr(args, "timeout_distance_scaled", False)),
        clouds_dir=getattr(args, "clouds_dir", None),
        reward_scale=(getattr(args, "reward_scale", None) if getattr(args, "reward_scale", None) is not None else 1.0),
        coherence_cost_weight=getattr(args, "coherence_cost_weight", 0.0),
        coherence_tau=getattr(args, "coherence_tau", 0.4),
        coherence_terminate_tau=getattr(args, "coherence_terminate_tau", 0.0),
        coherence_terminate_penalty=getattr(
            args, "coherence_terminate_penalty", 100.0),
        void_terminate_frac=getattr(args, "void_terminate_frac", 0.0),
        void_terminate_penalty=getattr(args, "void_terminate_penalty", 100.0),
        halt_terminate_steps=getattr(args, "halt_terminate_steps", 0),
        halt_throttle_eps=getattr(args, "halt_throttle_eps", 0.05),
        halt_penalty_scale=getattr(args, "halt_penalty_scale", 1.0),
        random_spawn=True,
        trav_path=args.trav_path,
        collision_terminate_frac=args.collision_terminate_frac,
        collision_terminate_penalty=args.collision_terminate_penalty,
        action_chunk=args.action_chunk,
        footprint_along_motion=args.footprint_along_motion,
        forward_only=args.forward_only,
        failure_snap_dir=str(args.out_dir / "failures"),
        obs_out_hw=((args.obs_height, args.obs_width)
                    if (args.render_height or args.render_width) else None),
    )
    return SceneEnv(world_backend=world, semantic_backend=sem,
                    scene_ids=[args.scene], cfg=env_cfg)


# Every reward term SceneEnv puts on `info`. Summed per episode so the table
# shows WHY a return is what it is -- a -1000 return from one crash and a -1000
# return from a hundred bad steps are different diagnoses.
EVAL_COMPONENTS = ("semantic", "goal", "collision", "step", "spin", "backward",
                   "smooth", "timeout", "crash", "proximity", "goal_bonus",
                   "coherence", "coherence_crash")


def _pose_xyyaw(env) -> list:
    P = env.unwrapped._robot_pose_world
    return [round(float(P[0, 3]), 3), round(float(P[1, 3]), 3),
            round(float(np.arctan2(P[1, 0], P[0, 0])), 3)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--scene", default="rugd_trail_00")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--goal_frame", type=int, default=30)
    ap.add_argument("--max_steps", type=int, default=60,
                    help="episode step budget (60 = training default; raise "
                         "for scenes recorded with large per-frame motion)")
    ap.add_argument("--goal_xy", default=None,
                    help='designed obstacle test: pin the goal to "x,y" (nav-'
                         'frame meters, read off the top-down figure axes) — '
                         'e.g. just past a tree so the straight line crosses it')
    # 2026-09-03: these were readable ONLY through env_config.json adoption
    # (`getattr(args, ..., default)`), so when that file is missing there was no
    # way to make eval terminate the way training does. Every eval that day ran
    # with crash termination OFF (frac 0) and coherence termination OFF, which
    # is why five runs reported ZERO crashes against a 72% crash rate in
    # training, and why goal counts came back at 20/20. Exposed as real flags so
    # a run can be matched to its training by hand.
    # Reward weights. These do NOT change what a frozen policy does -- only the
    # `return=` it reports -- but a return that is not on training's scale is a
    # number nobody can compare to anything, so expose them all. Defaults here
    # are the OLD eval defaults, deliberately: passing nothing reproduces
    # previous evals exactly.
    ap.add_argument("--goal_weight", type=float, default=1.5,
                    help="training's RW5 value is 10")
    ap.add_argument("--collision_weight", type=float, default=1.0)
    ap.add_argument("--goal_bonus", type=float, default=50.0,
                    help="training's RW5 value is 1000")
    ap.add_argument("--timeout_penalty", type=float, default=0.0,
                    help="training's RW5 value is 100. At 0 a TIMEOUT costs "
                         "NOTHING, so --timeout_distance_scaled is also inert.")
    ap.add_argument("--timeout_distance_scaled", action="store_true",
                    help="scale the timeout by remaining/initial distance, as "
                         "training does with TIMEOUTDIST=1")
    ap.add_argument("--proximity_weight", type=float, default=0.0)
    ap.add_argument("--proximity_margin", type=float, default=1.0)
    ap.add_argument("--proximity_delta", action="store_true")
    ap.add_argument("--void_cost", type=float, default=0.3)
    ap.add_argument("--step_cost", type=float, default=0.05)
    ap.add_argument("--coherence_cost_weight", type=float, default=0.0,
                    help="graded cost below --coherence_tau. Training uses 10.")
    ap.add_argument("--coherence_tau", type=float, default=0.4)
    ap.add_argument("--coherence_terminate_tau", type=float, default=0.0,
                    help="END the episode below this coverage. Training uses "
                         "0.1 (0.05 on the COHTERM arm). 0 = never terminate.")
    ap.add_argument("--coherence_terminate_penalty", type=float, default=100.0)
    # Spawn jitter. Training spawns with +-20 deg of heading and +-0.4 m of
    # lateral offset; eval had no flag at all, so it spawned exactly on the
    # recorded pose -- an easier and different distribution (2026-09-03).
    ap.add_argument("--spawn_yaw_jitter", type=float, default=0.0)
    ap.add_argument("--spawn_lat_jitter", type=float, default=0.0)
    ap.add_argument("--void_terminate_frac", type=float, default=0.0,
                    help="END the episode when this fraction of the footprint "
                         "is void. The whole-frame coherence terminal does NOT "
                         "catch an unsupported ground patch under a "
                         "well-rendered view. 0 = off, as in training.")
    ap.add_argument("--void_terminate_penalty", type=float, default=100.0)
    ap.add_argument("--halt_terminate_steps", type=int, default=0,
                    help="must match training, or a policy trained to halt is "
                         "scored as if it merely timed out")
    ap.add_argument("--halt_throttle_eps", type=float, default=0.05)
    ap.add_argument("--halt_penalty_scale", type=float, default=1.0)
    ap.add_argument("--goal_radius", type=float, default=None,
                    help="arrival radius; training's FINAL value, not its start")
    ap.add_argument("--semantic_weight", type=float, default=None)
    ap.add_argument("--reward_scale", type=float, default=None)
    ap.add_argument("--action_smooth_cost", type=float, default=None)
    ap.add_argument("--spin_cost", type=float, default=None)
    ap.add_argument("--sem_palette", type=int, default=4,
                    help="colour table for the video semantic panels. MUST "
                         "match the semantics model (v26 = 4, v21 = 1) or the "
                         "panels are coloured with the wrong classes.")
    ap.add_argument("--eval_seed", type=int, default=7,
                    help="episode e uses seed eval_seed*10000+e, so two "
                         "policies evaluated at the same value see IDENTICAL "
                         "spawns and goals and can be compared pair by pair")
    ap.add_argument("--goal_support_min_frac", type=float, default=0.25,
                    help="adopted from env_config.json when present")
    ap.add_argument("--goal_support_radius", type=float, default=0.0,
                    help="reject goals with no cloud support within this "
                         "radius, as training does. Adopted from "
                         "env_config.json when present.")
    ap.add_argument("--collision_look_ahead", type=float, default=0.0,
                    help="0 = collision judged on the same box as shaping")
    ap.add_argument("--goal_dir_360", action="store_true",
                    help="sample goals the way training does (cone at a random "
                         "distance from the spawn) instead of a fixed goal")
    ap.add_argument("--goal_dist_range", default="",
                    help="e.g. 5,10 — must match training's --goal_dist_range")
    ap.add_argument("--goal_cone_deg", type=float, default=360.0)
    ap.add_argument("--goal_frame_range", default="",
                    help="e.g. 15,70 — must match training")
    ap.add_argument("--spawn_min_frame", type=int, default=0,
                    help="lower bound on the spawn frame. Without it a GOAL_XY "
                         "test spawns anywhere from the start of the path, so "
                         "most episodes are 10-17 m from the goal and cannot "
                         "reach it inside max_steps — the run measures reach, "
                         "not the terrain behaviour it was launched for.")
    ap.add_argument("--spawn_max_frame", type=int, default=None,
                    help="match the training rung: 3 = full traverses (rung 5)")
    ap.add_argument("--clips_dir", required=True)
    ap.add_argument("--poses_dir", required=True)
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--model_path", default="/scratch/m000204-pm06b/joana/NeoVerse/models")
    ap.add_argument("--reconstructor_path",
                    default="/scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt")
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--obs_cache", default=None,
                    help="ribbon-cache root; evaluate on cached diffused obs "
                         "(must match what the checkpoint trained on)")
    ap.add_argument("--no_alpha_gate", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="serve LIVE per-step diffusion observations (evals of "
                         "--live-trained policies; ~1.4 s/step -> raise --time)")
    ap.add_argument("--action_chunk", type=int, default=1,
                    help="match the policy's training chunk size (trajectory arm)")
    ap.add_argument("--footprint_along_motion", action="store_true",
                    help="match the policy's training rule: footprint follows "
                         "the commanded motion direction")
    ap.add_argument("--clouds_dir",
                    default="/scratch/m000204-pm06b/joana/outputs/scene_clouds/clouds",
                    help="proximity needs the scene clouds; without it that "
                         "term is silently zero in eval while active in training")
    ap.add_argument("--step_size_m", type=float, default=0.3,
                    help="metres per unit forward action — MUST match the "
                         "checkpoint's training value (J arms: 0.3)")
    ap.add_argument("--yaw_step_rad", type=float, default=0.5,
                    help="radians per unit yaw action — MUST match training "
                         "(J arms: 0.5)")
    ap.add_argument("--forward_only", action="store_true",
                    help="match the policy's training rule: negative velocity "
                         "clamps to 0")
    ap.add_argument("--live_ckpt",
                    default="/scratch/m000204-pm06b/joana/runs/train_semantic_v10/checkpoint-epoch-30.safetensors")
    ap.add_argument("--render_height", type=int, default=None,
                    help="render-high/observe-small: match the checkpoint's "
                         "training RENDERH (world at this res, obs downsized)")
    ap.add_argument("--render_width", type=int, default=None)
    ap.add_argument("--collision_terminate_frac", type=float, default=0.0,
                    help="match the policy's training rule; >0 ends the episode "
                         "on a real collision (and it does NOT count as success)")
    ap.add_argument("--collision_terminate_penalty", type=float, default=20.0)
    ap.add_argument("--sweep_sticky", type=float, default=0.0,
                    help="sticky-sweep lookup penalty in meters (0 = off, "
                         "matches training; see CachedDiffusedBackend)")
    ap.add_argument("--trav_path", default=None,
                    help="traversability yaml (v14 table for cached runs)")
    ap.add_argument("--obs_height", type=int, default=336,
                    help="render/obs height — MUST match the checkpoint's "
                         "training resolution (policy CNN is size-locked)")
    ap.add_argument("--obs_width", type=int, default=560)
    ap.add_argument("--blind", action="store_true",
                    help="zero the rgb observation (goal vector untouched): "
                         "does the policy actually use the world model's "
                         "frames, or is it odometry-only? Videos still show "
                         "the real frames — only the policy goes blind.")
    args = ap.parse_args()
    # TRAINING AND EVAL MUST MATCH. train_ppo_real writes env_config.json beside
    # the checkpoints; adopt it, and shout about anything that disagrees rather
    # than quietly running a policy outside the action model it learned.
    try:
        import json as _json
        _run = Path(args.checkpoint).resolve().parents[1]
        _ec = _run / "env_config.json"
        if _ec.exists():
            _tr = _json.loads(_ec.read_text())
            # collision_terminate_* belong here too: training ENDS the
            # episode at >=0.35 footprint non-traversable, eval defaulted to 0
            # and let the policy keep walking through terrain that would have
            # killed it — every episode then reports TIMEOUT and the crash
            # behaviour is invisible (2026-09-02).
            # Sampling first: rebuild the training goal/spawn distribution
            # unless the caller deliberately pinned a goal with --goal_xy.
            if not args.goal_xy:
                _gr = _tr.get("goal_dist_range")
                _fr = _tr.get("goal_frame_range")
                if _tr.get("goal_dir_360"):
                    args.goal_dir_360 = True
                if _gr and not args.goal_dist_range:
                    args.goal_dist_range = ",".join(str(v) for v in _gr)
                if _tr.get("goal_cone_deg") and args.goal_cone_deg >= 360.0:
                    args.goal_cone_deg = float(_tr["goal_cone_deg"])
                if _fr and not args.goal_frame_range:
                    args.goal_frame_range = ",".join(str(v) for v in _fr)
                print(f"[eval] goal sampling from training: dir360="
                      f"{args.goal_dir_360} range={args.goal_dist_range} "
                      f"cone={args.goal_cone_deg} frames={args.goal_frame_range}",
                      flush=True)
            for _k in ("step_size_m", "yaw_step_rad", "forward_only",
                       "look_ahead_dist", "goal_radius", "collision_threshold",
                       "collision_terminate_frac", "collision_terminate_penalty",
                       "action_chunk",
                       # 2026-09-02: training rejects goals with no
                       # reconstruction under them (14.5% of draws) and can
                       # judge collision on its own closer footprint. Eval did
                       # neither, so it was scoring a goal distribution
                       # training never sees.
                       "goal_support_radius_m", "collision_look_ahead_m",
                       # 2026-09-03: the ALPHA GATE. Training runs ungated;
                       # eval defaulted to gated, which turns low-coverage
                       # pixels into void -- and void leaves the collision
                       # fraction when void_cost > 0, so gated evals do not
                       # crash where training would. Every eval that day
                       # under-crashed for this reason.
                       "no_alpha_gate", "alpha_gate_tau",
                       # max_steps was NOT adopted: eval ran 60-step episodes
                       # against arms trained at 90, cutting every rollout a
                       # third short -- so "it never stopped at the boundary"
                       # could just mean the clock ran out (2026-09-02).
                       "max_steps", "goal_support_min_frac",
                       # reward, so a reported return means the same thing in
                       # both places (her ask 2026-09-02: "make eval like
                       # training please")
                       "semantic_weight", "goal_weight", "collision_weight",
                       "step_cost", "void_cost", "terrain_as_cost",
                       "spin_cost", "backward_cost", "action_smooth_cost",
                       "goal_bonus", "timeout_penalty", "proximity_weight",
                       "proximity_margin", "proximity_delta", "reward_scale",
                       "coherence_cost_weight", "coherence_tau",
                       "coherence_terminate_tau"):
                if _k not in _tr:
                    continue
                _have = getattr(args, _k, None)
                if _have is not None and _have != _tr[_k]:
                    print(f"[eval] MISMATCH {_k}: eval {_have} -> training "
                          f"{_tr[_k]} (adopting training)", flush=True)
                setattr(args, _k, _tr[_k])
            print(f"[eval] adopted training env from {_ec}", flush=True)
        else:
            print(f"[eval] WARNING: no env_config.json at {_ec} — this run "
                  f"predates it, so kinematics come from the CLI and may not "
                  f"match what the policy trained with", flush=True)
        # Read OUTSIDE the env_config block. Nesting it there meant a run with no
        # env_config.json -- which on 2026-09-03 was EVERY run -- also lost its
        # curriculum range, the one number eval most needs.
        # env_config.json records the range the run STARTED at. With a
        # distance curriculum the policy finishes somewhere else, and how
        # far it got is not predictable from the config -- notches are
        # EARNED. curriculum_state.json is rewritten every rollout with the
        # live range, so it is the only honest answer to "what was this
        # policy actually training on when we stopped it".
        _cs = _run / "curriculum_state.json"
        if _cs.exists():
            try:
                _st = json.loads(_cs.read_text())
                _fin = _st.get("goal_dist_range")
                if _fin:
                    _fin_s = ",".join(str(v) for v in _fin)
                    if args.goal_dist_range != _fin_s:
                        print(f"[eval] curriculum FINISHED at "
                              f"{_fin_s} m (env_config says "
                              f"{args.goal_dist_range}, its START). "
                              f"Evaluating at the finished range.",
                              flush=True)
                    args.goal_dist_range = _fin_s
            except Exception as _e2:
                print(f"[eval] could not read {_cs}: {_e2}", flush=True)
        else:
            print(f"[eval] no curriculum_state.json — either no distance "
                  f"curriculum, or a run from before it was recorded; "
                  f"using the env_config range {args.goal_dist_range}",
                  flush=True)
    except Exception as _e:
        print(f"[eval] could not read training env config: {_e}", flush=True)

    print(f"[eval] kinematics: step {args.step_size_m} m, yaw "
          f"{args.yaw_step_rad} rad, forward_only={args.forward_only} "
          f"— these MUST match the training run", flush=True)

    # The policy CNN is size-locked, so an eval at the wrong resolution dies
    # with "Observation spaces do not match" — after queueing, waiting for a
    # node, and loading the 14B pipeline. The checkpoint already KNOWS its
    # resolution, so stop making a human retype it from the run name.
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        _d, _, _ = load_from_zip_file(args.checkpoint, device="cpu",
                                      print_system_info=False)
        _rgb = _d["observation_space"]["rgb"]            # channels-first
        _h, _w = int(_rgb.shape[1]), int(_rgb.shape[2])
        if (_h, _w) != (args.obs_height, args.obs_width):
            print(f"[eval] obs resolution from checkpoint: {_w}x{_h} "
                  f"(overriding {args.obs_width}x{args.obs_height})", flush=True)
        args.obs_height, args.obs_width = _h, _w
    except Exception as e:
        print(f"[eval] could not read obs resolution from checkpoint ({e}); "
              f"using {args.obs_width}x{args.obs_height}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    inner_env = build_env(args)
    if args.blind:
        import gymnasium as gym

        class BlindObs(gym.ObservationWrapper):
            def observation(self, obs):
                obs = dict(obs)
                obs["rgb"] = np.zeros_like(obs["rgb"])
                return obs
        inner_env = BlindObs(inner_env)
    env = Monitor(inner_env)
    model = PPO.load(args.checkpoint, env=env, device="cuda")
    print(f"loaded {args.checkpoint}", flush=True)

    # Reuse the HUD rollout recorder from the training script.
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from train_ppo_real import save_rollout_video

    results, video_records = [], []
    for ep in range(args.episodes):
        if ep < args.videos:
            rec = save_rollout_video(model, env,
                                     args.out_dir / f"episode_{ep}.mp4",
                                     seed=args.eval_seed * 10000 + ep,
                                     sem_palette=args.sem_palette)
            if rec:
                rec["video"] = f"episode_{ep}.mp4"
                video_records.append(rec)
            # save_rollout_video runs its own episode; count it via a fresh one below
        # Same seed as the video rollout above, so the video IS this episode,
        # and so a second policy evaluated at the same --eval_seed sees
        # identical spawns and goals. Unseeded, blind and sighted runs drew
        # different goals and could only be compared in aggregate.
        obs, _ = env.reset(seed=args.eval_seed * 10000 + ep)
        goal = env.unwrapped._goal_world
        # 5 fields, matching every later row: [x, y, yaw, collision_frac,
        # dominant_class]. It used to be 4 here and 5 below, which made `traj`
        # ragged and np.array() on it raise (2026-09-03).
        traj = [_pose_xyyaw(env) + [0.0, -1]]
        done, steps, collided, total_r = False, 0, 0, 0.0
        ground_counts: dict = {}
        # 2026-09-02: a bare success=False conflated "crashed at step 8",
        # "walked to the boundary and held" and "never moved" -- three
        # completely different policies with identical summaries. Everything
        # below exists to separate them.
        d_start = float(getattr(env.unwrapped, "_initial_goal_dist", 0.0) or 0.0)
        comps = {k: 0.0 for k in EVAL_COMPONENTS}
        cov_sum, cov_n, trespass, min_dist = 0.0, 0, 0, float("inf")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(action)
            total_r += float(r)
            steps += 1
            for k in EVAL_COMPONENTS:
                v = float(info.get(k, 0.0))
                if v == v:
                    comps[k] += v
            _c = float(info.get("coverage", float("nan")))
            if _c == _c:
                cov_sum += _c
                cov_n += 1
            min_dist = min(min_dist, float(info.get("dist_to_goal", min_dist)))
            # trespass = the footprint's dominant class is grass. This is the
            # number that says whether "did not reach the goal" means restraint
            # or failure, when the goal itself sits on grass.
            if int(info.get("dominant_class_id", -1)) == 3:
                trespass += 1
            # collision magnitude = footprint fraction on non-traversable
            # classes (eval reward weight is 1.0, so |term| = the fraction).
            frac = round(float(max(0.0, -info.get("collision", 0.0))), 3)
            hit = int(frac > 0.01)
            collided += hit
            # terrain occupancy: which class was under the footprint this step
            # (the sidewalk-preference metric, 2026-08-28)
            gc = int(info.get("dominant_class_id", -1))
            ground_counts[gc] = ground_counts.get(gc, 0) + 1
            traj.append(_pose_xyyaw(env) + [frac, gc])
            done = term or trunc
        d_final = float(info.get("dist_to_goal", -1))
        if term:
            outcome = "GOAL"
        elif float(info.get("crash", 0.0)) != 0.0:
            outcome = "CRASH"
        elif float(info.get("coherence_crash", 0.0)) != 0.0:
            outcome = "INCOHERENT"
        elif float(info.get("halted", 0.0)) != 0.0:
            # Stopped deliberately, somewhere safe, having closed distance.
            # Distinct from TIMEOUT on purpose: "walked to the boundary and
            # correctly refused an unreachable goal" and "wandered until the
            # clock ran out" were previously the same outcome, so the behaviour
            # this project is about had no metric.
            outcome = "HALTED"
        else:
            outcome = "TIMEOUT"
        results.append({"episode": ep, "success": bool(term),
                        "outcome": outcome,
                        "steps": steps, "return": round(total_r, 2),
                        "collision_steps": collided,
                        "d_start": round(d_start, 2),
                        "closed_frac": (round(1.0 - d_final / d_start, 3)
                                        if d_start > 1e-6 else None),
                        "min_dist": round(min_dist, 2),
                        "trespass_steps": trespass,
                        "mean_coverage": (round(cov_sum / cov_n, 3)
                                          if cov_n else None),
                        "reward_components": {k: round(v, 2)
                                              for k, v in comps.items()},
                        "final_dist": round(d_final, 2),
                        "goal_xy": [round(float(goal[0]), 3), round(float(goal[1]), 3)],
                        "ground_class_counts": {str(k): v for k, v
                                                in sorted(ground_counts.items())},
                        "traj": traj})
        print(f"ep {ep:2d}: {outcome:<10} steps={steps:3d} "
              f"d {d_start:5.1f} -> {d_final:5.1f} m  closest {min_dist:5.1f}  "
              f"grass {trespass:3d}  return={total_r:+.1f}", flush=True)

    succ = [r for r in results if r["success"]]
    # aggregate terrain occupancy across all episodes (share of ALL steps)
    agg: dict = {}
    for rr in results:
        for k, v in rr.get("ground_class_counts", {}).items():
            agg[k] = agg.get(k, 0) + v
    n_all = max(1, sum(agg.values()))
    class_names = {
        "0": "void", "1": "sky", "2": "trail", "3": "grass", "4": "rough",
        "5": "water", "6": "sidewalk", "7": "road", "8": "pavement",
        "9": "stairs", "10": "obstacle", "11": "vegetation", "12": "person",
        "13": "vehicle", "-1": "none"}
    ground_share = {class_names.get(k, k): round(v / n_all, 3)
                    for k, v in sorted(agg.items(), key=lambda kv: -kv[1])}
    outcomes: dict = {}
    for rr in results:
        outcomes[rr["outcome"]] = outcomes.get(rr["outcome"], 0) + 1
    closed = [r["closed_frac"] for r in results if r["closed_frac"] is not None]
    covs = [r["mean_coverage"] for r in results if r["mean_coverage"] is not None]
    comp_mean = {k: round(float(np.mean([r["reward_components"][k]
                                         for r in results])), 2)
                 for k in EVAL_COMPONENTS}
    summary = {
        "checkpoint": str(args.checkpoint),
        "scene": args.scene,
        "episodes": args.episodes,
        "outcomes": outcomes,
        "success_rate": round(len(succ) / args.episodes, 3),
        "mean_d_start": round(float(np.mean([r["d_start"] for r in results])), 2),
        "mean_d_final": round(float(np.mean([r["final_dist"] for r in results])), 2),
        "mean_closed_frac": round(float(np.mean(closed)), 3) if closed else None,
        "mean_min_dist": round(float(np.mean([r["min_dist"] for r in results])), 2),
        "trespass_rate": round(float(np.mean(
            [1.0 if r["trespass_steps"] > 0 else 0.0 for r in results])), 3),
        "mean_trespass_steps": round(float(np.mean(
            [r["trespass_steps"] for r in results])), 2),
        "mean_coverage": round(float(np.mean(covs)), 3) if covs else None,
        "mean_steps_to_goal": round(float(np.mean([r["steps"] for r in succ])), 1) if succ else None,
        "mean_return": round(float(np.mean([r["return"] for r in results])), 2),
        "mean_collision_steps": round(float(np.mean([r["collision_steps"] for r in results])), 2),
        "reward_components_mean": comp_mean,
        "ground_share": ground_share,
    }
    with open(args.out_dir / "metrics.json", "w") as f:
        json.dump({"summary": summary, "episodes": results,
                   "video_episodes": video_records}, f, indent=2)
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
