"""Traversability reward for the RL policy.

The reward answers one question: is the robot's NEXT foot placement on
traversable terrain? We compute it in three steps:

  1. Given the current robot pose and the chosen action, predict where the
     robot's feet will be one timestep from now (in world coords).
  2. Project that 3D position back into the currently-rendered camera view.
  3. Look up the semantic class at that pixel; return a reward that's
     positive on traversable classes (grass, road, sidewalk, ...) and
     negative on non-traversable ones (building, water, ...).

The semantic-lookup part is decoupled from the projection math via
`SemanticBackend`, so we can swap "SAM3-on-rendered-RGB" for
"NeoVerse-diffusion-semantic" without touching the reward logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


# ---------------------------------------------------------------------------
# Traversability lookup (29-class Go2W taxonomy + void; ids 0..29)
# ---------------------------------------------------------------------------

# Mirror of sam3_precompute_labels.CLASSES / diffsynth.utils.semantics.CLASS_COLORS.
# We only need the boolean "traversable" flag here. Index order MUST stay in
# lockstep with those two files.
TRAVERSABLE = np.array([
    False,  # 0  void
    False,  # 1  sky
    True,   # 2  dirt          FLAG: revisit — loose dirt may not be Go2W-safe
    True,   # 3  sand          FLAG: revisit — loose sand may not be Go2W-safe
    True,   # 4  grass
    True,   # 5  gravel
    True,   # 6  mulch
    False,  # 7  mud           hazard
    False,  # 8  water         hazard
    False,  # 9  rock          obstacle
    True,   # 10 asphalt
    True,   # 11 concrete
    True,   # 12 road
    True,   # 13 sidewalk
    True,   # 14 crosswalk
    False,  # 15 building
    False,  # 16 wall
    False,  # 17 fence
    True,   # 18 bridge
    False,  # 19 tree
    False,  # 20 vegetation
    False,  # 21 log
    True,   # 22 stairs        Go2W handles stairs
    False,  # 23 pole
    False,  # 24 traffic-sign
    False,  # 25 traffic-light
    False,  # 26 vehicle
    False,  # 27 motorcycle
    False,  # 28 bicycle
    False,  # 29 person
], dtype=bool)


def is_traversable(class_id: int) -> bool:
    """Class ID -> traversable? Uses the 30-class Go2W taxonomy (0=void, 1..29 named)."""
    if class_id < 0 or class_id >= len(TRAVERSABLE):
        return False
    return bool(TRAVERSABLE[class_id])


# ---------------------------------------------------------------------------
# Semantic backend — swappable source of per-pixel class labels
# ---------------------------------------------------------------------------

class SemanticBackend(Protocol):
    """Anything that turns an RGB image into a per-pixel class-id map."""
    def segment(self, rgb: np.ndarray) -> np.ndarray:
        """RGB (H, W, 3) uint8 -> labels (H, W) int32 in RUGD's 0..24 space."""
        ...


# ---------------------------------------------------------------------------
# Reward config + main entry point
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    r_traversable: float = 1.0     # reward for stepping onto a traversable class
    r_non_traversable: float = -5.0  # penalty for stepping onto anything else
    r_step: float = -0.05          # per-step cost, encourages efficient trajectories
    r_goal_reached: float = 100.0  # bonus on reaching the goal
    r_out_of_frame: float = -10.0  # penalty if next foot lands off the rendered image
    goal_radius: float = 0.5       # meters: within this counts as "reached"
    step_size: float = 0.3         # meters per forward action unit — matches SceneEnv


def compute_reward(
    *,
    rgb: np.ndarray,               # current camera view, (H, W, 3) uint8
    K: np.ndarray,                 # (3, 3) camera intrinsics for the current view
    w2c: np.ndarray,               # (4, 4) world->camera for the current view
    robot_pose_world: np.ndarray,  # (4, 4) robot's current pose in world frame
    action: np.ndarray,            # (2,) continuous action: (v_forward, omega_yaw), each in [-1, 1]
    goal_world: np.ndarray,        # (3,) goal position in world frame
    semantic_backend: SemanticBackend,
    cfg: RewardConfig = RewardConfig(),
) -> tuple[float, dict]:
    """Compute the reward for THIS step and return (reward, info_dict).

    The info_dict is for logging / debugging: whether the goal was reached,
    which class the next foot landed on, etc.
    """
    # --- 1. Predict next foot position in the WORLD frame -------------------
    #
    # For Stage 1 we treat the robot as a point mass on the ground plane.
    # The action is (v_forward in [-1,1], omega_yaw in [-1,1]); we scale to
    # a physical step size and yaw change per step.
    v_forward = float(action[0]) * cfg.step_size            # meters this step
    # omega_yaw applied at the current pose; simplest: rotate the forward
    # direction by omega_yaw*(some angle) before stepping. For a first pass we
    # just step in the current forward direction (yaw handled by env, not reward).
    forward_world = robot_pose_world[:3, :3] @ np.array([1.0, 0.0, 0.0])   # x-forward convention
    next_foot_world = robot_pose_world[:3, 3] + v_forward * forward_world

    # --- 2. Project the next foot position into the current camera view ----
    next_foot_h = np.append(next_foot_world, 1.0)                          # homogeneous
    next_foot_cam = w2c @ next_foot_h                                       # camera frame
    if next_foot_cam[2] <= 1e-3:
        # Behind the camera or ~zero depth => can't reason about traversability.
        return cfg.r_out_of_frame, {"reason": "next_foot_behind_camera"}

    u = K[0, 0] * next_foot_cam[0] / next_foot_cam[2] + K[0, 2]
    v = K[1, 1] * next_foot_cam[1] / next_foot_cam[2] + K[1, 2]
    H, W = rgb.shape[:2]
    if not (0 <= u < W and 0 <= v < H):
        return cfg.r_out_of_frame, {"reason": "next_foot_out_of_image", "u": u, "v": v}

    # --- 3. Look up semantic class at that pixel ---------------------------
    labels = semantic_backend.segment(rgb)                                  # (H, W) int32
    class_id = int(labels[int(v), int(u)])
    trav = is_traversable(class_id)

    # --- 4. Assemble reward -----------------------------------------------
    reward = cfg.r_step
    reward += cfg.r_traversable if trav else cfg.r_non_traversable

    # Goal bonus
    dist_to_goal = float(np.linalg.norm(next_foot_world - goal_world))
    reached = dist_to_goal < cfg.goal_radius
    if reached:
        reward += cfg.r_goal_reached

    return reward, {
        "next_foot_world": next_foot_world.tolist(),
        "next_foot_pixel": (float(u), float(v)),
        "class_id": class_id,
        "traversable": trav,
        "dist_to_goal": dist_to_goal,
        "reached_goal": reached,
    }
