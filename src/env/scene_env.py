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

@dataclass
class SceneEnvConfig:
    max_steps: int = 100
    step_size_m: float = 0.3            # meters per unit forward action
    yaw_step_rad: float = 0.5           # radians per unit yaw action
    reward: RewardWeights = field(default_factory=RewardWeights)
    look_ahead_dist: float = 1.5        # meters; passed to the reward
    goal_radius: float = 0.5            # meters; within this counts as "reached goal"
    collision_threshold: float = 0.1    # class score at/below which counts as collision
    spin_cost: float = 0.0              # shaping v2: penalty * |yaw action| (taxes circling)
    goal_bonus: float = 0.0             # v4: one-time reward on reaching the goal — must
                                        # exceed what an episode could farm (~+35 in v3)
    trav_path: "str | None" = None      # traversability yaml override (v14 table for cached runs)
    random_spawn: bool = False          # shaping v2: spawn along the real trajectory if the
                                        # backend offers sample_start_pose(scene_id, rng)
    failure_snap_dir: "str | None" = None  # save a figure at each collision
                                        # (obs + semantics + reward numbers)
    failure_snap_max: int = 200         # cap so long runs don't fill the disk


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
        self.observation_space = spaces.Dict({
            "rgb":  spaces.Box(low=0, high=255, shape=(H, W, 3), dtype=np.uint8),
            "goal": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

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
        cv2.imwrite(str(out / f"collision_{self._failure_snaps:04d}_"
                              f"{self._scene_id}_step{self._steps:03d}.png"),
                    img[:, :, ::-1])
        self._failure_snaps += 1

    # ---------------- gym API ----------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Choose a scene (round-robin for now; can be random later).
        idx = self.np_random.integers(0, len(self.scene_ids))
        self._scene_id = self.scene_ids[idx]
        self.world_backend.load_scene(self._scene_id)

        if self.cfg.random_spawn and hasattr(self.world_backend, "sample_start_pose"):
            self._robot_pose_world = self.world_backend.sample_start_pose(
                self._scene_id, self.np_random).copy()
        else:
            self._robot_pose_world = self.world_backend.start_pose(self._scene_id).copy()
        if hasattr(self.world_backend, "sample_goal_position"):
            # per-episode goal when the backend has goal_frame_range set;
            # falls back to the fixed goal otherwise
            self._goal_world = self.world_backend.sample_goal_position(
                self._scene_id, self.np_random,
                self._robot_pose_world[:2, 3]).copy()
        else:
            self._goal_world = self.world_backend.goal_position(self._scene_id).copy()
        self._prev_position = None
        self._steps = 0

        self._render_current()
        return self._obs(), {"scene_id": self._scene_id}

    def step(self, action: np.ndarray):
        assert self._robot_pose_world is not None, "call reset() first"
        action = np.asarray(action, dtype=np.float32).clip(-1.0, 1.0)

        # Semantic labels for the current view (from mock cache OR real segmenter).
        semantic_image = self.semantic_backend.segment(self._last_rgb)

        # Extract robot position + heading from the 4x4 pose. Heading = local +x
        # of the robot (its "forward"), transformed to world coords.
        robot_position = self._robot_pose_world[:3, 3].copy()
        robot_heading = self._robot_pose_world[:3, :3] @ np.array([1.0, 0.0, 0.0], dtype=np.float32)

        # ---- decomposed reward on the CURRENT view + robot pose ----
        breakdown = compute_reward(
            semantic_image=semantic_image,
            K=self._last_K,
            w2c=self._last_w2c,
            robot_position=robot_position,
            robot_heading=robot_heading,
            goal=self._goal_world,
            traversability_scores=self._trav_scores,
            non_traversable_mask=self._non_trav,
            previous_position=self._prev_position,
            look_ahead_dist=self.cfg.look_ahead_dist,
            body_length=GO2_BODY_LENGTH,
            body_width=GO2_BODY_WIDTH,
            weights=self.cfg.reward,
        )

        # Collision snapshot: the view + semantics + numbers behind every
        # non-traversable footprint, saved BEFORE the pose advances so the
        # figure shows exactly what the reward punished.
        if (self.cfg.failure_snap_dir is not None and breakdown.collision < 0
                and self._failure_snaps < self.cfg.failure_snap_max):
            self._save_failure_snapshot(breakdown, semantic_image)

        # ---- apply the action to advance the robot pose ----
        self._prev_position = robot_position
        self._advance_pose(action)
        self._steps += 1

        # ---- render the new view for the NEXT step's observation ----
        self._render_current()

        # Reached goal?
        dist_to_goal = float(np.linalg.norm(self._robot_pose_world[:3, 3] - self._goal_world))
        terminated = dist_to_goal < self.cfg.goal_radius
        truncated = self._steps >= self.cfg.max_steps

        spin_term = -self.cfg.spin_cost * abs(float(action[1]))
        bonus = self.cfg.goal_bonus if terminated else 0.0
        reward = breakdown.total + spin_term + bonus

        info = breakdown.to_dict()
        info["spin"] = spin_term
        info["goal_bonus"] = bonus
        info["total"] = reward
        info["dist_to_goal"] = dist_to_goal
        info["reached_goal"] = terminated
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        return self._last_rgb

    # ---------------- helpers ----------------

    def _obs(self) -> dict:
        goal_robot = self._goal_in_robot_frame()
        return {
            "rgb":  self._last_rgb.copy(),
            "goal": goal_robot.astype(np.float32),
        }

    def _render_current(self) -> None:
        rgb, K, w2c = self.world_backend.render(self._robot_pose_world)
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
