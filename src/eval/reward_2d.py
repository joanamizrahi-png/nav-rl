"""2D-image-based reward for offline validation on recorded clips.

Given a semantic label image (from SAM3 or Track A) and a robot pose in world
coords, compute a decomposed reward:

    R = w_sem  * semantic_traversability      # class scores at the projected footprint
      + w_goal * goal_progress                # motion component toward goal
      - w_col  * collision                    # fraction of footprint pixels on non-traversable classes

No 3D / Gaussians needed. Uses camera projection to figure out where the robot's
next body position falls in the current image.

Design note: this is Milestone A's reward. For Milestone C (RL training) we may
swap this for the Gaussian-based version at `reward_gaussian.py`. See
`../../OPTIONS.md` for the alternatives.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np


# Go2 body dimensions used to build the footprint rectangle in the world frame.
GO2_BODY_LENGTH = 0.6   # meters, front-to-back
GO2_BODY_WIDTH = 0.3    # meters, side-to-side


@dataclass
class RewardWeights:
    semantic: float = 1.0
    goal: float = 0.5
    collision: float = 5.0     # collision term already scaled to ~[0,1], so this is the actual weight
    step_cost: float = 0.05    # constant per-frame negative; kills "stand still and accumulate reward"
    # Shaping v2 (2026-07-23): void (unknown/unseen) split from true obstacles.
    # Default 0.0 keeps old behavior (void stays inside `collision` via the
    # non_traversable mask); setting > 0 moves void into its own mild penalty
    # so exploring near the unknown isn't as punishing as hitting a tree.
    void_cost: float = 0.0


@dataclass
class RewardBreakdown:
    total: float
    semantic: float
    goal: float
    collision: float
    step: float = 0.0               # constant per-frame step cost (negative when weights.step_cost > 0)
    void: float = 0.0               # unknown-region term (only when weights.void_cost > 0)
    # Debug info for plots/logs:
    n_footprint_pixels: int = 0
    n_traversable_pixels: int = 0
    mean_class_score: float = 0.0
    dominant_class_id: int = -1
    off_frame_frac: float = 0.0     # fraction of footprint that projected outside the image

    def to_dict(self) -> dict:
        return asdict(self)


def _footprint_corners_world(
    position: np.ndarray, heading: np.ndarray,
    look_ahead_dist: float,
    length: float, width: float,
) -> np.ndarray:
    """Return the 4 corners (in world frame) of a rectangle centered
    look_ahead_dist ahead of `position` along `heading`."""
    # Right = 90-deg clockwise from heading in the ground plane (heading assumed
    # roughly horizontal). For a first pass we treat everything as if the ground
    # plane is z=0 (world Y-up or Z-up convention doesn't matter as long as
    # heading is in the ground plane).
    right = np.array([heading[1], -heading[0], 0.0], dtype=np.float32)
    right /= max(np.linalg.norm(right), 1e-8)

    center = position + look_ahead_dist * heading
    hl = length / 2.0
    hw = width / 2.0
    return np.stack([
        center + hl * heading + hw * right,
        center + hl * heading - hw * right,
        center - hl * heading - hw * right,
        center - hl * heading + hw * right,
    ], axis=0)   # (4, 3)


def _project_points(
    points_world: np.ndarray,   # (N, 3)
    K: np.ndarray,              # (3, 3) intrinsics
    w2c: np.ndarray,            # (4, 4) world -> camera
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points to pixel coords. Returns (uv, in_front_mask).

    uv: (N, 2) float pixel coords
    in_front_mask: (N,) bool — True where the point is in front of the camera
                   (positive depth); such points have a meaningful projection.
    """
    n = len(points_world)
    pts_h = np.concatenate([points_world, np.ones((n, 1), dtype=points_world.dtype)], axis=1)
    pts_cam = (w2c @ pts_h.T).T[:, :3]   # (N, 3), camera frame
    z = pts_cam[:, 2]
    in_front = z > 1e-3
    uv = np.zeros((n, 2), dtype=np.float32)
    valid = in_front
    uv[valid, 0] = K[0, 0] * pts_cam[valid, 0] / z[valid] + K[0, 2]
    uv[valid, 1] = K[1, 1] * pts_cam[valid, 1] / z[valid] + K[1, 2]
    return uv, in_front


def _fill_polygon(H: int, W: int, corners_uv: np.ndarray) -> np.ndarray:
    """Rasterize a convex polygon (4 corners in image coords) to a boolean mask.

    Uses PIL for the fill — no scipy dependency required.
    """
    from PIL import Image, ImageDraw
    img = Image.new("L", (W, H), 0)
    ImageDraw.Draw(img).polygon(
        [(float(u), float(v)) for u, v in corners_uv], outline=1, fill=1
    )
    return np.array(img, dtype=bool)


def compute_reward(
    *,
    semantic_image: np.ndarray,       # (H, W) int class ids
    K: np.ndarray,                     # (3, 3) intrinsics
    w2c: np.ndarray,                   # (4, 4) world -> camera
    robot_position: np.ndarray,        # (3,) current world position
    robot_heading: np.ndarray,         # (3,) unit vector, robot's forward direction in world frame
    goal: np.ndarray,                  # (3,) goal position in world
    traversability_scores: np.ndarray, # (num_classes,)
    non_traversable_mask: np.ndarray,  # (num_classes,) bool — True = non-traversable (collision)
    previous_position: Optional[np.ndarray] = None,
    look_ahead_dist: float = 0.5,
    body_length: float = GO2_BODY_LENGTH,
    body_width: float = GO2_BODY_WIDTH,
    weights: RewardWeights = RewardWeights(),
) -> RewardBreakdown:
    """Compute the reward at one timestep. All world-frame inputs align with the
    scene the recorded video was shot in.
    """
    H, W = semantic_image.shape[:2]

    # --- 1. Semantic traversability + collision from projected footprint ---
    corners_world = _footprint_corners_world(
        robot_position, robot_heading,
        look_ahead_dist=look_ahead_dist,
        length=body_length, width=body_width,
    )
    corners_uv, in_front = _project_points(corners_world, K, w2c)

    if not in_front.all():
        # Footprint straddles / is behind the camera => skip semantic; treat as no info.
        sem_score = 0.0
        n_footprint_pixels = 0
        n_traversable_pixels = 0
        mean_class_score = 0.0
        dominant_class_id = -1
        off_frame_frac = 1.0
        collision_frac = 0.0
        void_frac = 0.0
    else:
        mask = _fill_polygon(H, W, corners_uv)
        # Clip mask to image bounds — polygon may extend past edges.
        in_bounds = mask
        n_footprint_pixels = int(in_bounds.sum())

        if n_footprint_pixels == 0:
            sem_score = 0.0
            n_traversable_pixels = 0
            mean_class_score = 0.0
            dominant_class_id = -1
            off_frame_frac = 1.0
            collision_frac = 0.0
            void_frac = 0.0
        else:
            classes = semantic_image[in_bounds]
            scores = traversability_scores[classes]
            mean_class_score = float(scores.mean())
            sem_score = mean_class_score

            n_traversable_pixels = int((~non_traversable_mask[classes]).sum())
            if weights.void_cost > 0:
                # Void (class 0 = unknown/unseen) is its own mild term; collision
                # counts only REAL obstacle classes. Otherwise void rides inside
                # collision (original behavior).
                void_sel = classes == 0
                collision_frac = float((non_traversable_mask[classes] & ~void_sel).mean())
                void_frac = float(void_sel.mean())
            else:
                collision_frac = float(non_traversable_mask[classes].mean())
                void_frac = 0.0

            counts = np.bincount(classes, minlength=len(traversability_scores))
            dominant_class_id = int(np.argmax(counts))
            off_frame_frac = 0.0

    # --- 2. Goal progress ---
    if previous_position is None:
        goal_score = 0.0
    else:
        prev_dist = float(np.linalg.norm(previous_position - goal))
        curr_dist = float(np.linalg.norm(robot_position - goal))
        goal_score = prev_dist - curr_dist    # positive => closed distance this step

    # --- 3. Combine ---
    semantic_term = weights.semantic * sem_score
    goal_term = weights.goal * goal_score
    collision_term = -weights.collision * collision_frac
    void_term = -weights.void_cost * void_frac
    step_term = -weights.step_cost                      # constant per-frame negative
    total = semantic_term + goal_term + collision_term + void_term + step_term

    return RewardBreakdown(
        total=float(total),
        semantic=float(semantic_term),
        goal=float(goal_term),
        collision=float(collision_term),
        step=float(step_term),
        void=float(void_term),
        n_footprint_pixels=n_footprint_pixels,
        n_traversable_pixels=n_traversable_pixels,
        mean_class_score=mean_class_score,
        dominant_class_id=dominant_class_id,
        off_frame_frac=off_frame_frac,
    )
