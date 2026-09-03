"""SceneEnv — Gym environment wrapping the NeoVerse world model as an RL simulator.

Interface (Gymnasium API):
  reset() -> (observation, info)
  step(action) -> (observation, reward, terminated, truncated, info)

Observation:
  Dict({
    "rgb":  Box(0, 255, (H, W, 3), uint8)   -- the current camera view
    "goal": Box(-inf, inf, (3,), float32)   -- (dx, dy, dyaw) to goal, robot frame
  })

Action:
  Box(-1, 1, (2,), float32) = (v_forward, omega_yaw), each in [-1, 1].
  Scaled inside step() by config.step_size and config.yaw_step_rad.

This file is deliberately thin. The two things that actually do work live in
their own modules and are injected here:
  - `world_backend`  — turns (scene, pose) into an RGB image + camera intrinsics/extrinsics.
                       Real backend calls NeoVerse; the mock backend returns cached images.
  - `semantic_backend` — turns an RGB image into a per-pixel class-id map.
                         Default = SAM3-on-RGB. Later = the fine-tuned diffusion output.

Nothing in this file is NeoVerse-specific, which is what lets us develop the
env, the reward, and the PPO loop entirely on the Mac before cluster access.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # gym is a soft dep at import time so this file can be read anywhere
    gym = None
    spaces = None

from .reward import SemanticBackend                             # keep the Protocol only

from ..eval.reward_2d import (
    RewardWeights, RewardBreakdown, compute_reward, GO2_BODY_LENGTH, GO2_BODY_WIDTH,
)
from ..eval.traversability import load_traversability, NUM_CLASSES


# ---------------------------------------------------------------------------
# World-model backend abstraction
# ---------------------------------------------------------------------------

class WorldBackend(Protocol):
    """Anything that can render an RGB view at a requested pose."""
    H: int
    W: int

    def load_scene(self, scene_id: str) -> None:
        """Prepare a scene (may build a pose cache, load Gaussians, etc.)."""
        ...

    def render(self, pose_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render at the given robot pose in world frame.

        Args:
            pose_world: (4, 4) camera-to-world matrix.
        Returns:
            rgb:  (H, W, 3) uint8
            K:    (3, 3)    float32 — camera intrinsics for this render
            w2c:  (4, 4)    float32 — world->camera for this render
        """
        ...

    def start_pose(self, scene_id: str) -> np.ndarray:
        """Where the robot starts in this scene, as a (4, 4) pose."""
        ...

    def goal_position(self, scene_id: str) -> np.ndarray:
        """Where the robot should reach. (3,) world coords."""
        ...


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------

_ENV_SEQ = 0    # per-process env counter, tags crash snapshots


@dataclass
class SceneEnvConfig:
    max_steps: int = 100
    step_size_m: float = 0.3            # meters per unit forward action
    yaw_step_rad: float = 0.5           # radians per unit yaw action
    reward: RewardWeights = field(default_factory=RewardWeights)
    look_ahead_dist: float = 1.5        # meters; passed to the reward
    # 2026-09-02: separate the LETHAL test from the GRADED one. 0.0 = keep the
    # old single-box behaviour (collision judged at look_ahead_dist), so the
    # control arm is bit-identical to every run before today. Set ~0.8-1.0 to
    # move only crash/collision in to the body while the graded semantic score
    # keeps its 1.5 m of warning. Watch reward/collision_off_frame: if the near
    # box misses the frame it falls back and the split is not in effect.
    collision_look_ahead_m: float = 0.0
    # 2026-09-02, measured: 14.5% of sampled goals have NO reconstruction under
    # them at all (scripts/goal_audit.py, 2000 episodes x 6 scenes). The goal
    # sampler is pure geometry -- spawn + d*(cos th, sin th) -- and never
    # consults the cloud, so the robot is regularly sent to a place that does
    # not exist in the world model: it cannot see the goal, cannot reach it,
    # and eats the timeout. Those episodes are pure noise in the return.
    # > 0 = resample until the goal has ground points within this radius.
    goal_support_radius_m: float = 0.0
    goal_support_tries: int = 12
    # "Enough reconstruction" cannot be "at least one point" -- a single stray
    # gaussian passes that and the goal is still in a void. Calibrate against
    # the scene itself: count ground points within the radius at every RECORDED
    # path position (places we know are well reconstructed, because someone
    # walked them and filmed it), take the median, and require the goal to
    # reach this fraction of it. Self-scaling across scenes of different
    # density. 0 = fall back to the any-point test.
    goal_support_min_frac: float = 0.25
    goal_radius: float = 0.5            # meters; within this counts as "reached goal"
    collision_threshold: float = 0.1    # class score at/below which counts as collision
    spin_cost: float = 0.0              # shaping v2: penalty * |yaw action| (taxes circling)
    backward_cost: float = 0.0          # penalty * max(0, -v): backing to the goal is
                                        # reward-rational (goals spawn behind, footprint
                                        # projects ahead) but camera-blind on the robot
    goal_bonus: float = 0.0             # v4: one-time reward on reaching the goal — must
                                        # exceed what an episode could farm (~+35 in v3)
    trav_path: "str | None" = None      # traversability yaml override (v14 table for cached runs)
    random_spawn: bool = False          # shaping v2: spawn along the real trajectory if the
                                        # backend offers sample_start_pose(scene_id, rng)
    failure_snap_dir: "str | None" = None  # save a figure at each collision
                                        # (obs + semantics + reward numbers)
    failure_snap_max: int = 200         # cap so long runs don't fill the disk
    failure_snap_min_frac: float = 0.2  # only snapshot substantial overlaps —
                                        # tiny footprint grazes are risk, not collision
    collision_terminate_frac: float = 0.0    # >0: footprint overlap at/above this
                                        # ENDS the episode (no goal bonus) — see step().
    collision_terminate_penalty: float = 20.0  # subtracted on crash
    # Void termination: >0 ends the episode when this fraction of the footprint
    # has NO gaussian support (alpha-gated to class 0). Guards the blind-spot
    # exploit that alpha-gating would otherwise open. 0 = off.
    void_terminate_frac: float = 0.0
    void_terminate_penalty: float = 100.0     # ~1/10 of a crash at RW5 scale
    # IMAGE-coverage version (her spec 2026-09-01): the footprint measure above
    # asks "is the ground under me observed"; this asks "does the WHOLE VIEW
    # have enough gaussian support for the world model to paint something
    # coherent". With the alpha gate on, (semantic_image == 0).mean() IS
    # 1 - alpha coverage — the same `cov` number the spin certificates print.
    # Her reasoning: stepping onto locally-unsupported ground is fine as long
    # as the view around it is well supported. 0 = off.
    image_void_terminate_frac: float = 0.0
    # ---- COHERENCE (2026-09-01, her design) --------------------------------
    # `coverage` = MEAN ALPHA over the frame: how much of the diffusion's
    # conditioning input has real gaussian support behind it. Pure rasterizer
    # geometry, no diffusion in it, and NOT the same statistic as the void
    # fraction the gate produces.
    #
    # Her claim, which the panels support: coherence is a WHOLE-FRAME property.
    # A frame is either the world or it is not; there is no half-trustworthy
    # frame. So the decision belongs at frame level, not per pixel.
    #
    # Two thresholds, two jobs:
    #   coherence_tau (0.4, read off the ladder by eye)  -> GRADED COST. The
    #     render is degrading; pull the policy back, do not kill it. Measured:
    #     the wander regime averages 0.397, the corridor 0.645 — terminating
    #     at 0.4 would end most exploratory episodes and teach path-hugging,
    #     which is the failure the spawn jitter exists to prevent.
    #   coherence_terminate_tau (~0.1)                   -> TERMINATE. Frames
    #     with essentially no geometry anywhere (GEOvoid = 1.000 was observed).
    #     Nothing there can be scored and nothing can be learned from it.
    coherence_cost_weight: float = 0.0      # 0 = off (every run before today)
    coherence_tau: float = 0.4
    coherence_terminate_tau: float = 0.0    # 0 = off
    coherence_terminate_penalty: float = 100.0
    # Proximity cost (2026-08-24, after Run A): charge for being NEAR obstacles,
    # measured against the STATIC reconstructed geometry (scene cloud from
    # dump_scene_cloud.py) — the diffusion cannot dream this away, unlike the
    # semantic footprint (Run A scored 0.0 semantic collisions on the tree test
    # at 0.10 m median true clearance: live renders diverge at grazing range).
    proximity_weight: float = 0.0       # 0 = off (every run before this date)
    proximity_margin: float = 1.0       # meters; cost ramps linearly inside this
    # v2 (2026-08-25): charge only SOLID classes by default. v1 included
    # vegetation (11) — trail corridors are walled with it, so the charge was
    # always-on everywhere: a flat tax the policy paid instead of a gradient
    # it followed (measured: solid clearance 0.06 m vs Run C's 0.45 m).
    proximity_classes: tuple = (10, 13)  # obstacle, vehicle
    # Potential-shaped proximity (2026-08-27, advisor reward spec): charge
    # weight*(d_{t-1}-d_t) while within proximity_margin of an obstacle —
    # approaching costs, retreating refunds, standing is free.
    proximity_delta: bool = False
    # Timeout penalty (2026-08-27, advisor reward spec): subtracted when the
    # episode hits max_steps without reaching the goal. Makes freezing/stalling
    # strictly worse than trying — the missing counterweight to termination
    # penalties (the -20 crash penalty froze the policy when standing was free).
    timeout_penalty: float = 0.0
    # PROPORTIONAL TIMEOUT (2026-09-02, her call: "proportional timeout seems
    # genius"). A flat timeout penalty charges the same for stopping 1 m short
    # as for stopping 10 m short, so there is NO gradient toward "as close as
    # the terrain legally allows" — and when the goal sits on grass, stopping
    # short is the CORRECT behaviour being punished at full price.
    #
    # Scaling it by remaining/initial distance makes the reward's optimum
    # "approach as close as legally possible and hold", without anyone having
    # to define where the boundary is — which is unanswerable anyway, since
    # any definition rests on cloud labels that are ~17% wrong.
    #
    # Note the goal term already telescopes to goal_weight*(d_start - d_final),
    # so only the ENDING distance matters; this fixes the counterweight that
    # was flat. It also prices drifting back out after approaching (warm went
    # 1.14 m -> 1.89 m and paid nothing for it).
    timeout_distance_scaled: bool = False
    # uniform reward multiplier applied at the very end of step() — tames the
    # critic's value targets under +-1000-scale terminals; 1.0 = off.
    reward_scale: float = 1.0
    # Render-high / observe-small (2026-08-30, her design after the w270
    # resolution ladder: 336-render is mush on campus scenes, 560 is coherent).
    # When set to (h, w): the WORLD keeps the backend's native render size for
    # reward/footprint/labels, and only the policy-facing obs["rgb"] is
    # downsampled to (h, w). None = obs at render size (every run before this).
    obs_out_hw: "tuple | None" = None
    # Anti-odometry lever (2026-08-31): Gaussian noise (std in meters) added to
    # the goal-vector observation's xy each step. A perfect goal vector makes
    # vision optional for the dominant reward terms — a noisy compass forces
    # visual navigation. 0 = off (every run before this). Reward/termination
    # still use the TRUE goal; only the policy's observation is corrupted.
    goal_noise_std: float = 0.0
    clouds_dir: "str | None" = None     # dir holding <scene>_cloud.npz files
    # Trajectory output (plan-B arm, 2026-08-25): the policy emits k action
    # pairs per decision and only observes again after all k execute. Rewards
    # still accrue per sub-step, so the world stays action-conditioned; only
    # the POLICY's decision rate changes. 1 = per-action (default, all runs
    # before this date).
    action_chunk: int = 1
    # Action smoothness (2026-08-27): penalty * mean|a_t - a_{t-1}|. Targets
    # the bang-bang habit (PPO saturates clipped Gaussians at the bounds and
    # flip-flops the turn sign -> minimum-turn-circle capture loops); charges
    # CHANGES, so sustained gentle turns stay free. 0 = off (all runs before).
    action_smooth_cost: float = 0.0
    # Batched live training (2026-08-27): when True the env NEVER renders
    # itself — after each step/reset the owning vectorized env batches all
    # robots' poses into one diffusion call and injects the frames back via
    # inject_render(). Observations must be rebuilt (env._obs()) after
    # injection; the obs returned by step()/reset() are stale placeholders.
    defer_render: bool = False
    # Forward-only motion (2026-08-27): negative velocity actions clamp to 0
    # (stand-and-turn). Kills the entire backward-exploit class BY CONSTRUCTION
    # instead of pricing it; clamping at the env keeps the action space shape,
    # so warm-starts from older checkpoints still load.
    forward_only: bool = False
    # Motion-direction footprint (2026-08-26): score the ground THIS step's
    # action moves onto — forward actions score ahead (unchanged), backward
    # actions point the footprint behind the camera, where there are no
    # labels; compute_reward's no-info branch then prices the step as
    # worst-case terrain under terrain_as_cost. Motion onto unseen ground is
    # never cheaper than the worst visible ground, so the camera-blind
    # reversing exploit (prox-v2 100% backward, chunk-5 all-backward arcs)
    # stops paying. False = heading-pinned footprint (every run before).
    footprint_along_motion: bool = False


# ---------------------------------------------------------------------------
# The env itself
# ---------------------------------------------------------------------------

class SceneEnv(gym.Env if gym is not None else object):
    """RL environment where the world model is our simulator."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        world_backend: WorldBackend,
        semantic_backend: SemanticBackend,
        scene_ids: list[str],
        cfg: SceneEnvConfig = SceneEnvConfig(),
    ):
        if gym is None:
            raise ImportError("gymnasium is required to instantiate SceneEnv")
        super().__init__()
        self.world_backend = world_backend
        self.semantic_backend = semantic_backend
        self.scene_ids = list(scene_ids)
        self.cfg = cfg

        # Pre-load traversability table + collision mask (rewards use these each step).
        from pathlib import Path as _Path
        self._trav_scores = load_traversability(
            _Path(cfg.trav_path) if cfg.trav_path else None)         # (NUM_CLASSES,) float32
        self._non_trav = self._trav_scores <= cfg.collision_threshold

        H, W = world_backend.H, world_backend.W
        if cfg.obs_out_hw is not None:
            H, W = int(cfg.obs_out_hw[0]), int(cfg.obs_out_hw[1])
        self.observation_space = spaces.Dict({
            "rgb":  spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8),
            "goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2 * max(1, cfg.action_chunk),), dtype=np.float32)

        # Populated on reset()
        self._scene_id: Optional[str] = None
        self._robot_pose_world: Optional[np.ndarray] = None
        self._goal_world: Optional[np.ndarray] = None
        self._prev_position: Optional[np.ndarray] = None
        self._steps: int = 0

        # Cache the latest render so step() can compute the reward without a fresh render.
        # (We render at the START of each step because that's what the policy sees;
        # after action, the NEXT render is done for the NEXT observation.)
        self._last_rgb: Optional[np.ndarray] = None
        self._last_K: Optional[np.ndarray] = None
        self._last_w2c: Optional[np.ndarray] = None
        self._failure_snaps = 0
        # x4/x8 runs put N envs behind one failure_snap_dir, each with its own
        # counter — so robot 0 and robot 1 both wrote collision_0000_... and
        # silently overwrote each other. Every env gets a tag (2026-09-01).
        global _ENV_SEQ
        _ENV_SEQ += 1
        self._env_tag = _ENV_SEQ

    def _save_failure_snapshot(self, breakdown, semantic_image) -> None:
        """One PNG per collision: [obs RGB | v14-colorized semantics], with the
        reward numbers burned in. Files: <dir>/collision_<n>_<scene>_step<k>.png"""
        import cv2
        from pathlib import Path as _Path
        from ..eval.palette import CLASS_COLORS_V14_255
        out = _Path(self.cfg.failure_snap_dir)
        out.mkdir(parents=True, exist_ok=True)
        rgb = self._last_rgb
        pal = CLASS_COLORS_V14_255
        col = pal[np.clip(semantic_image, 0, len(pal) - 1)]
        if col.shape[:2] != rgb.shape[:2]:
            col = cv2.resize(col, (rgb.shape[1], rgb.shape[0]),
                             interpolation=cv2.INTER_NEAREST)
        # draw the exact footprint quad the reward scored (her ask 2026-08-19)
        try:
            from ..eval.reward_2d import (
                _footprint_corners_world, _project_points,
                GO2_BODY_LENGTH, GO2_BODY_WIDTH,
            )
            pose = self._robot_pose_world
            hd = getattr(self, "_last_fp_heading", None)
            if hd is None:
                hd = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
            corners = _footprint_corners_world(
                pose[:3, 3], hd, look_ahead_dist=self.cfg.look_ahead_dist,
                length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
            uv, in_front = _project_points(corners, self._last_K, self._last_w2c)
            if in_front.all():
                rgb = rgb.copy()
                cv2.polylines(rgb, [uv.astype(np.int32)], True, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.polylines(col, [uv.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass
        panel = np.concatenate([rgb, col], axis=1)
        bar = np.zeros((44, panel.shape[1], 3), dtype=np.uint8)
        x, y = self._robot_pose_world[0, 3], self._robot_pose_world[1, 3]
        cv2.putText(bar, f"COLLISION  {self._scene_id}  step {self._steps}  "
                         f"pose=({x:+.2f},{y:+.2f})", (6, 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, f"collision={breakdown.collision:+.2f}  "
                         f"dominant_class={breakdown.dominant_class_id}  "
                         f"mean_score={breakdown.mean_class_score:.2f}  "
                         f"footprint_px={breakdown.n_footprint_pixels}  "
                         f"trav_px={breakdown.n_traversable_pixels}", (6, 37),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        img = np.concatenate([bar, panel], axis=0)
        cv2.imwrite(str(out / f"collision_e{self._env_tag:02d}_"
                              f"{self._failure_snaps:04d}_"
                              f"{self._scene_id}_step{self._steps:03d}.png"),
                    img[:, :, ::-1])

        # Machine-readable sidecar (2026-09-01). The pose used to live only in
        # the burned-in text, so 1304 crash snapshots could be looked at but
        # never REPLAYED — we could not re-render a crash to ask whether the
        # obstacle was real. One CSV line per crash fixes that for good.
        try:
            hd_w = pose[:3, :3] @ np.array([1.0, 0.0, 0.0])
            csv_p = out / "crash_poses.csv"
            if not csv_p.exists():
                csv_p.write_text(
                    "env,n,scene,step,x,y,z,hx,hy,goal_x,goal_y,collision,"
                    "void_frac,dominant,footprint_px,trav_px\n")
            with open(csv_p, "a") as fh:
                fh.write(
                    f"{self._env_tag},{self._failure_snaps},"
                    f"{self._scene_id},{self._steps},"
                    f"{pose[0, 3]:.4f},{pose[1, 3]:.4f},{pose[2, 3]:.4f},"
                    f"{hd_w[0]:.4f},{hd_w[1]:.4f},"
                    f"{self._goal_world[0]:.4f},{self._goal_world[1]:.4f},"
                    f"{breakdown.collision:.4f},{breakdown.void_frac:.4f},"
                    f"{breakdown.dominant_class_id},"
                    f"{breakdown.n_footprint_pixels},"
                    f"{breakdown.n_traversable_pixels}\n")
        except Exception:
            pass

        self._failure_snaps += 1

    # ---------------- gym API ----------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._last_action = None
        self._prev_obstacle_dist = None
        # Choose a scene (round-robin for now; can be random later).
        idx = self.np_random.integers(0, len(self.scene_ids))
        self._scene_id = self.scene_ids[idx]
        self.world_backend.load_scene(self._scene_id)
        self._load_obstacle_points(self._scene_id)

        if self.cfg.random_spawn and hasattr(self.world_backend, "sample_start_pose"):
            self._robot_pose_world = self.world_backend.sample_start_pose(
                self._scene_id, self.np_random).copy()
        else:
            self._robot_pose_world = self.world_backend.start_pose(self._scene_id).copy()
        if hasattr(self.world_backend, "sample_goal_position"):
            # per-episode goal when the backend has goal_frame_range set;
            # falls back to the fixed goal otherwise
            # Centre the goal cone on the RECORDED heading at the spawn
            # frame -- not the jittered pose, and not a tangent re-derived
            # from whichever frame is nearest the jittered spawn. Falls back
            # to the actual pose yaw only if the backend cannot report it.
            _yaw = None
            if hasattr(self.world_backend, "last_spawn_base_yaw"):
                _yaw = self.world_backend.last_spawn_base_yaw()
            if _yaw is None:
                _yaw = float(np.arctan2(self._robot_pose_world[1, 0],
                                        self._robot_pose_world[0, 0]))
            self._goal_world = self.world_backend.sample_goal_position(
                self._scene_id, self.np_random,
                self._robot_pose_world[:2, 3], cone_yaw=_yaw).copy()
            # Reject goals with no reconstruction under them and draw again.
            # Keeps the LAST draw if every try fails, so a scene whose goals
            # are mostly off-cloud degrades to the old behaviour instead of
            # looping forever or crashing.
            r = self.cfg.goal_support_radius_m
            if r > 0.0:
                gp = getattr(self, "_ground_pts", {}).get(self._scene_id)
                if gp is not None and len(gp):
                    ref = getattr(self, "_support_ref", {}).get(self._scene_id, 0.0)
                    need = max(1, int(self.cfg.goal_support_min_frac * ref))
                    for _try in range(max(1, self.cfg.goal_support_tries)):
                        d2 = np.abs(gp - self._goal_world[:2]).max(axis=1)
                        near = gp[d2 <= r]
                        # `need` points inside the radius, not merely one --
                        # and NOTE this deliberately says nothing about
                        # traversability: a goal ON GRASS is legitimate and is
                        # the whole stop-at-the-boundary curriculum. The test
                        # is only "does this place exist in the reconstruction".
                        if len(near) >= need and float(np.linalg.norm(
                                near - self._goal_world[:2], axis=1).min()) <= r:
                            break
                        self._goal_world = self.world_backend.sample_goal_position(
                            self._scene_id, self.np_random,
                            self._robot_pose_world[:2, 3],
                            cone_yaw=_yaw).copy()
        else:
            self._goal_world = self.world_backend.goal_position(self._scene_id).copy()
        self._prev_position = None
        # a reset teleports the robot, so the first frame difference of an
        # episode is meaningless -- drop it rather than log a false jump
        self._rgb_delta = float("nan")
        self._steps = 0
        self._initial_goal_dist = float(np.linalg.norm(
            self._robot_pose_world[:3, 3] - self._goal_world))

        if self.cfg.defer_render:
            self._needs_render = True
            if self._last_rgb is None:      # first-ever reset: placeholder obs
                H, W = self.observation_space["rgb"].shape[:2]
                self._last_rgb = np.zeros((H, W, 3), dtype=np.uint8)
                self._last_K = np.eye(3, dtype=np.float32)
                self._last_w2c = np.eye(4, dtype=np.float32)
        else:
            self._render_current()
        return self._obs(), {"scene_id": self._scene_id}

    def set_goal_radius(self, r: float) -> None:
        """Curriculum hook: training callback anneals the capture radius
        (e.g. 1.0 m -> 0.5 m) via vec_env.env_method("set_goal_radius", r)."""
        self.cfg.goal_radius = float(r)

    _goal_dist_lo0: "float | None" = None

    def goal_cone_probe(self, n: int = 500, seed: int = 0) -> dict:
        """Is the goal ever BEHIND the robot? Sample n spawn/goal pairs per
        scene exactly as reset() does -- same backend, same live config, same
        scenes -- but WITHOUT rendering, so this costs milliseconds and can run
        in the startup banner of every job.

        Standalone tests can pass while the real thing is broken: the goal cone
        was centred on a path tangent re-derived from whichever frame was
        nearest the JITTERED spawn, which on a walk that doubles back can be a
        frame on the other leg pointing the opposite way. 2.25% of episodes
        across the six AU scenes put the goal more than 90 deg behind a
        forward-only robot, which can then only time out. Fixed by centring on
        the recorded heading at the spawn frame; this probe is what proves the
        fix is live in THIS run rather than in a test file.

        Returns {scene: (mean_deg, max_deg, behind_pct)}.
        """
        rng = np.random.default_rng(seed)
        wb = self.world_backend
        out = {}
        for sid in self.scene_ids:
            errs = []
            for _ in range(int(n)):
                pose = wb.sample_start_pose(sid, rng)
                yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
                cone = (wb.last_spawn_base_yaw()
                        if hasattr(wb, "last_spawn_base_yaw") else None)
                g = wb.sample_goal_position(sid, rng, pose[:2, 3],
                                            cone_yaw=cone)
                d = g[:2] - pose[:2, 3]
                e = float(np.arctan2(d[1], d[0])) - yaw
                errs.append(abs(np.degrees((e + np.pi) % (2 * np.pi) - np.pi)))
            e = np.asarray(errs)
            out[sid] = (float(e.mean()), float(e.max()),
                        float(100.0 * (e > 90).mean()))
        return out

    def set_goal_dist(self, d: float) -> "tuple | None":
        """Distance-curriculum hook (2026-08-29, Joana: E must bootstrap like
        B did): goals START close (~3 m) and GROW as the policy earns wins.
        Backends sample goals, so the knob lives on the backend cfg.

        2026-09-02: this only ever wrote `goal_dist_m`, which
        `sample_goal_position` reads in the RECORDED-FRAME branch. Every arm
        since 458724 passes `--goal_dir_360`, and that branch returns from
        `goal_dist_range` before `goal_dist_m` is ever consulted -- so the
        curriculum has been silently inert on every run for weeks, including
        the ones whose names carry `gds3`. It moves the UPPER bound now: the
        near end of the range stays put (close goals must remain in the mix,
        they are the easy wins that bootstrap learning) and the far end grows
        as the policy earns them.
        """
        cfg = getattr(self.world_backend, "cfg", None)
        if cfg is None:
            return None
        cfg.goal_dist_m = float(d)
        if getattr(cfg, "goal_dir_360", False):
            lo, _hi = getattr(cfg, "goal_dist_range", None) or (3.0, float(d))
            lo = float(lo)
            # Remember where the near end STARTED -- it is a floor the window
            # can never slide below, and reading it off the live range would
            # let it ratchet.
            if self._goal_dist_lo0 is None:
                self._goal_dist_lo0 = lo
            win = getattr(cfg, "goal_dist_window_m", None)
            if win:
                # Sliding window: retire the trivially close goals as the far
                # end grows, instead of pinning the near end forever. At
                # width 5 the range walks (2,3) -> (3,8) rather than (2,8), so
                # the last band the policy has already mastered stops
                # consuming rollout steps.
                lo = max(self._goal_dist_lo0, float(d) - float(win))
            cfg.goal_dist_range = (lo, max(lo + 0.5, float(d)))
        # Returned so the callback can log the ACTUAL range on wandb. Logging
        # only `d` hid the near end, which the sliding window moves.
        return getattr(cfg, "goal_dist_range", None)

    def inject_render(self, rgb: np.ndarray, K: np.ndarray, w2c: np.ndarray,
                      labels: "np.ndarray | None" = None,
                      coverage: "float | None" = None) -> None:
        """Batched-live path: the vec-env pushes this robot's frame in after
        the shared batched diffusion call. `labels` feeds the injected
        semantic backend (reward source for the NEXT step); `coverage` is the
        mean alpha of that frame, which only the backend can see."""
        # Frame-to-frame |dRGB|: Joana's measure of world-model coherence
        # (2026-09-02). Alpha coverage is a proxy -- it says how much geometry
        # backs the view -- but the failure she actually cares about is the
        # model CUTTING to a different scene, which is exactly a large frame
        # difference. It is also deployment-safe: real camera frames are
        # temporally continuous, so a term built on this can never fire on the
        # robot, whereas a coverage term teaches "avoid low-alpha headings",
        # which is a property of this reconstruction and not of the world.
        # Logged as a DIAGNOSTIC first -- it also grows with fast turning, so
        # its distribution has to be looked at before it becomes a cost.
        if self._last_rgb is not None and rgb is not None \
                and self._last_rgb.shape == rgb.shape:
            self._rgb_delta = float(
                np.abs(rgb.astype(np.float32)
                       - self._last_rgb.astype(np.float32)).mean() / 255.0)
        else:
            self._rgb_delta = float("nan")
        self._last_rgb, self._last_K, self._last_w2c = rgb, K, w2c
        if labels is not None:
            self._injected_labels = labels
        if coverage is not None:
            self._last_coverage = float(coverage)
        self._needs_render = False

    def _load_obstacle_points(self, scene_id: str) -> None:
        """Body-height obstacle points (xy) from the scene cloud, cached per
        scene. Serves the proximity cost; silently absent -> cost is 0."""
        if not hasattr(self, "_obstacle_pts"):
            self._obstacle_pts = {}
        if not hasattr(self, "_ground_pts"):
            self._ground_pts = {}
        if not hasattr(self, "_support_ref"):
            self._support_ref = {}
        if self.cfg.clouds_dir is None or scene_id in self._obstacle_pts:
            return
        from pathlib import Path
        p = Path(self.cfg.clouds_dir) / f"{scene_id}_cloud.npz"
        if not p.exists():
            self._obstacle_pts[scene_id] = None
            if self.cfg.proximity_weight > 0:
                print(f"[SceneEnv] WARNING: proximity cost ON but no cloud at {p}")
            return
        d = np.load(p)
        pts, labs = d["points"], d["labels"].astype(int)
        # Robot-height band only (0.15-1.2 m): canopy/arches overhang the
        # walkway without blocking a knee-high robot — charging for them
        # taxed walking under trees (audit finding on gnd_AU_60).
        ob = pts[np.isin(labs, list(self.cfg.proximity_classes))
                 & (pts[:, 2] > 0.15) & (pts[:, 2] < 1.2)][:, :2]
        self._obstacle_pts[scene_id] = ob[::4].astype(np.float32) if len(ob) else None
        # GROUND points, separately: `_obstacle_pts` is an obstacle-class slice
        # in a height band and says nothing about whether a patch of ground was
        # reconstructed at all. Goal support needs the latter.
        gr = pts[(pts[:, 2] < 0.15) & (labs >= 0)][:, :2]
        self._ground_pts[scene_id] = gr[::4].astype(np.float32) if len(gr) else None
        # Reference density: how many ground points sit within the goal radius
        # at a place we KNOW is reconstructed -- the recorded walk itself.
        ref = 0.0
        g = self._ground_pts[scene_id]
        r = self.cfg.goal_support_radius_m
        if g is not None and len(g) and r > 0.0 and "traj_positions" in d:
            path = (np.asarray(d["traj_positions"], dtype=float)
                    * np.array([1.0, -1.0, 1.0]))[:, :2]
            cnt = [int((np.abs(g - q).max(axis=1) <= r).sum())
                   for q in path[::max(1, len(path) // 40)]]
            ref = float(np.median(cnt)) if cnt else 0.0
        self._support_ref[scene_id] = ref
        if r > 0.0:
            print(f"[SceneEnv] {scene_id}: goal support reference "
                  f"{ref:.0f} ground pts within {r} m on the recorded path; "
                  f"goals need >= {self.cfg.goal_support_min_frac:.0%} of that",
                  flush=True)

    def _proximity_term(self, robot_xy: np.ndarray) -> float:
        if self.cfg.proximity_weight <= 0.0:
            return 0.0
        ob = getattr(self, "_obstacle_pts", {}).get(self._scene_id)
        if ob is None or len(ob) == 0:
            return 0.0
        dmin = float(np.sqrt(((ob - robot_xy[None, :2]) ** 2).sum(1)).min())
        if self.cfg.proximity_delta:
            # Potential-shaped variant: charge APPROACHING an obstacle and
            # refund retreating, weight * (d_{t-1} - d_t), active inside the
            # margin (a wide horizon, e.g. 5 m). No standing charge, so the
            # flat-tax failure mode is impossible by construction.
            prev = getattr(self, "_prev_obstacle_dist", None)
            self._prev_obstacle_dist = dmin
            if prev is None or min(prev, dmin) >= self.cfg.proximity_margin:
                return 0.0
            return -self.cfg.proximity_weight * (prev - dmin)
        if dmin >= self.cfg.proximity_margin:
            return 0.0
        return -self.cfg.proximity_weight * (self.cfg.proximity_margin - dmin) / self.cfg.proximity_margin

    def step(self, action: np.ndarray):
        """Chunk-aware step: executes cfg.action_chunk sub-actions (1 = the
        per-action default). Sub-step rewards and components are summed; the
        observation returned is the one AFTER the whole chunk."""
        k = max(1, self.cfg.action_chunk)
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        if k == 1:
            return self._step_single(action)
        total_r, agg = 0.0, None
        obs = terminated = truncated = info = None
        for i in range(k):
            obs, r, terminated, truncated, info = self._step_single(
                action[2 * i:2 * i + 2])
            total_r += float(r)
            if agg is None:
                agg = {kk: float(vv) for kk, vv in info.items()
                       if isinstance(vv, (int, float))
                       and not isinstance(vv, bool)
                       and kk != "dist_to_goal"}
            else:
                for kk in agg:
                    agg[kk] += float(info.get(kk, 0.0))
            if terminated or truncated:
                break
        info = dict(info)          # last sub-step's flags/dist survive...
        info.update(agg)           # ...numeric reward components are summed
        info["total"] = total_r
        return obs, total_r, terminated, truncated, info

    def _step_single(self, action: np.ndarray):
        assert self._robot_pose_world is not None, "call reset() first"
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)
        if self.cfg.forward_only and action[0] < 0.0:
            action = action.copy()
            action[0] = 0.0

        # Semantic labels for the current view (from mock cache OR real segmenter).
        semantic_image = self.semantic_backend.segment(self._last_rgb)

        # Extract robot position + heading from the 4x4 pose. Heading = local +x
        # of the robot (its "forward"), transformed to world coords.
        robot_position = self._robot_pose_world[:3, 3].copy()
        robot_heading = self._robot_pose_world[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # Footprint direction follows the commanded motion, not the nose
        # (see cfg.footprint_along_motion). Spin-only steps keep forward.
        fp_heading = robot_heading
        if self.cfg.footprint_along_motion and float(action[0]) < 0.0:
            fp_heading = -robot_heading
        self._last_fp_heading = fp_heading

        # ---- decomposed reward on the CURRENT view + robot pose ----
        breakdown = compute_reward(
            semantic_image=semantic_image,
            K=self._last_K,
            w2c=self._last_w2c,
            robot_position=robot_position,
            robot_heading=fp_heading,
            goal=self._goal_world,
            traversability_scores=self._trav_scores,
            non_traversable_mask=self._non_trav,
            previous_position=self._prev_position,
            look_ahead_dist=self.cfg.look_ahead_dist,
            collision_look_ahead_dist=(self.cfg.collision_look_ahead_m
                                       if self.cfg.collision_look_ahead_m > 0.0
                                       else None),
            body_length=GO2_BODY_LENGTH,
            body_width=GO2_BODY_WIDTH,
            weights=self.cfg.reward,
        )

        # Collision snapshot: the view + semantics + numbers behind every
        # non-traversable footprint, saved BEFORE the pose advances so the
        # figure shows exactly what the reward punished.
        if (self.cfg.failure_snap_dir is not None
                and breakdown.collision < -self.cfg.failure_snap_min_frac
                and self._failure_snaps < self.cfg.failure_snap_max):
            self._save_failure_snapshot(breakdown, semantic_image)

        # ---- apply the action to advance the robot pose ----
        self._prev_position = robot_position
        self._advance_pose(action)
        self._steps += 1

        # ---- render the new view for the NEXT step's observation ----
        if self.cfg.defer_render:
            self._needs_render = True     # owning vec-env batches the render
        else:
            self._render_current()

        # Reached goal?
        dist_to_goal = float(np.linalg.norm(self._robot_pose_world[:3, 3] - self._goal_world))
        terminated = dist_to_goal < self.cfg.goal_radius
        truncated = self._steps >= self.cfg.max_steps

        spin_term = -self.cfg.spin_cost * abs(float(action[1]))
        back_term = -self.cfg.backward_cost * max(0.0, -float(action[0]))
        smooth_term = 0.0
        if self.cfg.action_smooth_cost > 0.0:
            prev = getattr(self, "_last_action", None)
            if prev is not None:
                smooth_term = -self.cfg.action_smooth_cost * float(
                    np.abs(action - prev).mean())
            self._last_action = action.copy()
        bonus = self.cfg.goal_bonus if terminated else 0.0

        # Crash termination (2026-08-20). Measured on the tree test: a whole
        # episode of walking through a trunk cost -0.60 against a +50 arrival
        # bonus (1.2%), so driving through obstacles was reward-optimal — the
        # policy passed within 0.03-0.21 m of trunk centres while scoring
        # "success 1.0". A real collision must END the episode, like a real
        # robot's would. 0 = off (every result before this date).
        crash = 0.0
        if self.cfg.collision_terminate_frac > 0.0:
            frac = -float(breakdown.collision) / max(self.cfg.reward.collision, 1e-6)
            if frac >= self.cfg.collision_terminate_frac:
                crash = -self.cfg.collision_terminate_penalty
                truncated = True          # ends the episode, and NOT a success
                bonus = 0.0
        # VOID TERMINATION (2026-09-01, her concern): with the alpha gate ON,
        # unobserved regions become void and stop counting as collisions — but
        # then "walk into the unobserved" becomes a way to dodge real terrain:
        # the policy learns to exploit the world model's blind spots, and that
        # behavior means nothing on the real robot. Pessimism under uncertainty
        # (MOReL-style HALT): leaving the known world ENDS the episode at a
        # moderate cost — clearly worse than walking correctly, far cheaper
        # than a real crash. 0 = off (every run before this date).
        if (self.cfg.void_terminate_frac > 0.0 and crash == 0.0
                and float(breakdown.void_frac) >= self.cfg.void_terminate_frac):
            crash = -self.cfg.void_terminate_penalty
            truncated = True
            bonus = 0.0
        # Whole-image coverage version (her spec): unsupported GROUND is fine
        # if the VIEW is still well supported — the world model can paint a
        # coherent scene and the robot can reasonably step there. Only end the
        # episode when the view itself is mostly invention.
        image_void_frac = float((semantic_image == 0).mean())
        if (self.cfg.image_void_terminate_frac > 0.0 and crash == 0.0
                and image_void_frac >= self.cfg.image_void_terminate_frac):
            crash = -self.cfg.void_terminate_penalty
            truncated = True
            bonus = 0.0
        prox_term = self._proximity_term(self._robot_pose_world[:2, 3])

        # ---- coherence: is this frame still the world? ----
        coverage = getattr(self, "_last_coverage", None)
        coh_term, coh_crash = 0.0, 0.0
        if coverage is None and (self.cfg.coherence_cost_weight > 0.0
                                 or self.cfg.coherence_terminate_tau > 0.0):
            # Only the BATCHED live backend reports per-frame alpha. Asking for
            # coherence on a path that cannot supply it would silently train a
            # run with the mechanism switched off and nobody the wiser — the
            # exact failure shape as the SANPO length filter. Say so, loudly.
            if not getattr(self, "_coh_warned", False):
                print("[SceneEnv] WARNING: coherence requested but this backend "
                      "reports no per-frame alpha — the term is INERT.", flush=True)
                self._coh_warned = True
        if coverage is not None:
            if self.cfg.coherence_cost_weight > 0.0:
                coh_term = -self.cfg.coherence_cost_weight * max(
                    0.0, self.cfg.coherence_tau - coverage)
            if (self.cfg.coherence_terminate_tau > 0.0 and crash == 0.0
                    and coverage < self.cfg.coherence_terminate_tau):
                coh_crash = -self.cfg.coherence_terminate_penalty
                # truncated, NOT terminated — same convention as crash/void
                # above. info["reached_goal"] mirrors `terminated`, so setting
                # it here would score every coherence kill as a goal arrival.
                truncated = True
                bonus = 0.0
        timeout_term = 0.0
        if (truncated and not terminated and crash == 0.0 and coh_crash == 0.0
                and self.cfg.timeout_penalty > 0.0):
            timeout_term = -self.cfg.timeout_penalty
            if self.cfg.timeout_distance_scaled:
                d0 = float(getattr(self, "_initial_goal_dist", 0.0) or 0.0)
                frac = 1.0 if d0 <= 1e-6 else min(1.0, max(0.0, dist_to_goal / d0))
                timeout_term *= frac
        reward = (breakdown.total + spin_term + back_term + smooth_term
                  + bonus + crash + prox_term + timeout_term
                  + coh_term + coh_crash)

        info = breakdown.to_dict()
        info["spin"] = spin_term
        info["backward"] = back_term
        info["smooth"] = smooth_term
        # Always logged (even with both void terminations off) so the coverage
        # the policy actually experiences is visible in wandb from day one.
        info["image_void_frac"] = image_void_frac
        info["timeout"] = timeout_term
        info["goal_dist_frac"] = (
            dist_to_goal / self._initial_goal_dist
            if getattr(self, "_initial_goal_dist", 0.0) else float("nan"))
        info["crash"] = crash
        info["proximity"] = prox_term
        # Always logged, on or off, so the coverage the policy actually
        # experiences shows up in wandb from the first run.
        info["coverage"] = float("nan") if coverage is None else coverage
        info["rgb_delta"] = float(getattr(self, "_rgb_delta", float("nan")))
        info["coherence"] = coh_term
        info["coherence_crash"] = coh_crash
        info["goal_bonus"] = bonus
        info["total"] = reward
        info["dist_to_goal"] = dist_to_goal
        info["reached_goal"] = terminated
        # reward_scale (2026-08-28): shrink the WHOLE reward uniformly (ratios
        # preserved) so +-1000 terminals stop blowing up the critic's targets
        # (value_loss ~1e5 in the batch-1 RW5 trio). Info keys stay unscaled
        # so component logging remains in Jing-spec units.
        if self.cfg.reward_scale != 1.0:
            reward = float(reward) * self.cfg.reward_scale
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        return self._last_rgb

    # ---------------- helpers ----------------

    def _obs(self) -> dict:
        goal_robot = self._goal_in_robot_frame()
        if self.cfg.goal_noise_std > 0.0:
            goal_robot = goal_robot.copy()
            goal_robot[:2] += self.np_random.normal(
                0.0, self.cfg.goal_noise_std, size=2)
        rgb = self._last_rgb
        if self.cfg.obs_out_hw is not None:
            oh, ow = int(self.cfg.obs_out_hw[0]), int(self.cfg.obs_out_hw[1])
            if rgb.shape[0] != oh or rgb.shape[1] != ow:
                import cv2
                rgb = cv2.resize(rgb, (ow, oh), interpolation=cv2.INTER_AREA)
        return {
            "rgb":  rgb.copy(),
            "goal": goal_robot.astype(np.float32),
        }

    def _render_current(self) -> None:
        rgb, K, w2c = self.world_backend.render(self._robot_pose_world)
        # Frame-to-frame |dRGB|: Joana's measure of world-model coherence
        # (2026-09-02). Alpha coverage is a proxy -- it says how much geometry
        # backs the view -- but the failure she actually cares about is the
        # model CUTTING to a different scene, which is exactly a large frame
        # difference. It is also deployment-safe: real camera frames are
        # temporally continuous, so a term built on this can never fire on the
        # robot, whereas a coverage term teaches "avoid low-alpha headings",
        # which is a property of this reconstruction and not of the world.
        # Logged as a DIAGNOSTIC first -- it also grows with fast turning, so
        # its distribution has to be looked at before it becomes a cost.
        if self._last_rgb is not None and rgb is not None \
                and self._last_rgb.shape == rgb.shape:
            self._rgb_delta = float(
                np.abs(rgb.astype(np.float32)
                       - self._last_rgb.astype(np.float32)).mean() / 255.0)
        else:
            self._rgb_delta = float("nan")
        self._last_rgb, self._last_K, self._last_w2c = rgb, K, w2c

    def _advance_pose(self, action: np.ndarray) -> None:
        v_forward = float(action[0]) * self.cfg.step_size_m
        omega_yaw = float(action[1]) * self.cfg.yaw_step_rad

        # Rotate yaw first (about world +z), then translate forward.
        c, s = np.cos(omega_yaw), np.sin(omega_yaw)
        R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        self._robot_pose_world[:3, :3] = R_yaw @ self._robot_pose_world[:3, :3]

        forward_world = self._robot_pose_world[:3, :3] @ np.array([1.0, 0.0, 0.0])
        self._robot_pose_world[:3, 3] += v_forward * forward_world

    def _goal_in_robot_frame(self) -> np.ndarray:
        # (dx, dy, dyaw_to_goal) in the robot's local frame.
        dp_world = self._goal_world - self._robot_pose_world[:3, 3]
        R_wr = self._robot_pose_world[:3, :3]                                 # world->robot rotation
        dp_robot = R_wr.T @ dp_world
        dyaw = float(np.arctan2(dp_robot[1], dp_robot[0]))
        return np.array([dp_robot[0], dp_robot[1], dyaw])
