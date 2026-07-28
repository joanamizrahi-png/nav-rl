"""Gaussian-based reward function for outdoor navigation.

Given labeled 3D gaussians (position + class) and a robot pose + goal, compute
a decomposed reward with 4 components:

    R = w_sem  * semantic_traversability      # traversability score of ground ahead
      + w_goal * goal_progress                # motion toward goal
      + w_clear * clearance                   # headroom above ground (branches etc)
      - w_col  * collision                    # 1 if any gaussian in robot body volume

Returned as a `RewardBreakdown` so each component can be plotted separately
(the requested preliminary reward evaluation).

Note: goal_progress is only meaningful when we have a previous position to
compare against, so the reward function takes an optional `previous_position`.
When None (first timestep), goal_progress contribution is 0.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

from .footprint import (
    RobotPose,
    gaussians_in_body_volume,
    gaussians_in_lookahead,
    gaussians_in_clearance_column,
    GO2_BODY_HEIGHT,
    GO2_STEP_CLEARANCE,
)
from .ground_plane import GroundPlane


@dataclass
class RewardWeights:
    semantic: float = 1.0
    goal: float = 0.5
    clearance: float = 0.5
    collision: float = 5.0     # LARGE penalty on collision


@dataclass
class RewardBreakdown:
    total: float
    semantic: float
    goal: float
    clearance: float
    collision: float
    # Debug info for logging/plotting:
    dominant_class_id: int = -1
    dominant_class_score: float = 0.0
    lookahead_count: int = 0
    body_count: int = 0
    max_column_height: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def compute_reward(
    *,
    positions: np.ndarray,        # (N, 3) all gaussian world positions
    labels: np.ndarray,           # (N,) int class ids
    pose: RobotPose,              # robot's current pose
    goal: np.ndarray,             # (3,) goal position in world
    ground_plane: GroundPlane,
    traversability_scores: np.ndarray,  # (num_classes,) from load_traversability()
    previous_position: Optional[np.ndarray] = None,  # (3,) for goal_progress delta
    weights: RewardWeights = RewardWeights(),
) -> RewardBreakdown:
    """Compute the decomposed reward at ONE timestep.

    Args are all in world-frame coordinates aligned with the gaussian positions.
    """
    # --- 1. Semantic traversability of look-ahead footprint ---
    la_idx = gaussians_in_lookahead(positions, pose, ground_plane)
    if len(la_idx) == 0:
        sem_score = 0.0
        dominant_class_id = -1
        dominant_class_score = 0.0
    else:
        la_labels = labels[la_idx]
        # Weighted mode: average traversability score over gaussians in region.
        # Cleaner than picking a single mode; robust to fringe classes.
        la_scores = traversability_scores[la_labels]
        sem_score = float(la_scores.mean())
        # Also report the dominant class for logging/plotting.
        counts = np.bincount(la_labels, minlength=len(traversability_scores))
        dominant_class_id = int(np.argmax(counts))
        dominant_class_score = float(traversability_scores[dominant_class_id])

    # --- 2. Goal progress ---
    # Positive when the robot moved closer to the goal since the previous step.
    if previous_position is None:
        goal_score = 0.0
    else:
        prev_dist = float(np.linalg.norm(previous_position - goal))
        curr_dist = float(np.linalg.norm(pose.position - goal))
        goal_score = prev_dist - curr_dist   # positive = closed distance

    # --- 3. Clearance ---
    # Look at max-z gaussian in the vertical column above the look-ahead footprint.
    col_idx = gaussians_in_clearance_column(positions, pose, ground_plane)
    if len(col_idx) == 0:
        max_col_height = 0.0
        clearance_score = 1.0   # no overhead obstacles => full clearance
    else:
        heights = ground_plane.height_above(positions[col_idx])
        max_col_height = float(heights.max())
        # Score is 1.0 when obstacle is above robot head; drops to 0 when at
        # step-clearance height (would block the robot's body).
        if max_col_height >= GO2_BODY_HEIGHT:
            clearance_score = 1.0
        elif max_col_height <= GO2_STEP_CLEARANCE:
            clearance_score = 0.0    # obstacle at step height => blocked
        else:
            clearance_score = (max_col_height - GO2_STEP_CLEARANCE) / (GO2_BODY_HEIGHT - GO2_STEP_CLEARANCE)

    # --- 4. Collision ---
    # 1.0 if any gaussian is inside the robot's current body volume.
    body_idx = gaussians_in_body_volume(positions, pose, ground_plane)
    collision_score = 1.0 if len(body_idx) > 0 else 0.0

    # --- Combine ---
    total = (
        weights.semantic  * sem_score +
        weights.goal      * goal_score +
        weights.clearance * clearance_score -
        weights.collision * collision_score
    )

    return RewardBreakdown(
        total=float(total),
        semantic=float(weights.semantic * sem_score),
        goal=float(weights.goal * goal_score),
        clearance=float(weights.clearance * clearance_score),
        collision=float(-weights.collision * collision_score),
        dominant_class_id=dominant_class_id,
        dominant_class_score=dominant_class_score,
        lookahead_count=int(len(la_idx)),
        body_count=int(len(body_idx)),
        max_column_height=float(max_col_height),
    )
