"""Footprint volume queries against a scene of labeled gaussians.

Given a robot pose (position + heading) and a ground plane, compute:
- Which gaussians are inside the robot's body footprint on the ground.
- Which gaussians are in the vertical column above that footprint (for clearance).
- Which gaussians are inside a "next-step" region ahead of the robot (for reward).

Terminology:
- Footprint = 2D rectangle on the ground plane where the robot's body sits.
- Body volume = 3D box (footprint × robot_height) above the ground plane.
- Look-ahead region = footprint translated forward by `look_ahead_dist`, in the
  robot's heading direction. This is what the reward evaluates.

We express everything in the world frame (same as the gaussian positions), using
the ground plane to project onto/off of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .ground_plane import GroundPlane


# Go2-specific dimensions. Tunable.
GO2_BODY_LENGTH = 0.6   # meters, front-to-back
GO2_BODY_WIDTH = 0.3    # meters, side-to-side
GO2_BODY_HEIGHT = 0.4   # meters, standing height
GO2_STEP_CLEARANCE = 0.15   # meters, max obstacle height robot can step over


@dataclass
class RobotPose:
    """Robot pose in the world frame.

    position: (3,) center of body on the ground plane.
    heading: (3,) unit vector, robot's forward direction (in ground plane).
    """
    position: np.ndarray   # (3,)
    heading: np.ndarray    # (3,), unit vector, in ground plane (perpendicular to gp.normal)


def _rect_indices_on_plane(
    positions: np.ndarray,
    center: np.ndarray,
    forward: np.ndarray,
    right: np.ndarray,
    length: float,
    width: float,
    gp: GroundPlane,
    height_range: tuple[float, float],
) -> np.ndarray:
    """Return indices of `positions` inside a 3D box.

    The box is aligned to (forward, right, gp.normal). Its footprint on the
    ground plane is length x width centered at `center`. It extends
    `height_range[0]` to `height_range[1]` in the ground-normal direction.
    """
    d = positions - center                                     # (N, 3)
    fwd_coord = d @ forward                                    # signed forward offset
    right_coord = d @ right                                    # signed lateral offset
    height = d @ gp.normal                                     # signed height above center

    return np.where(
        (np.abs(fwd_coord) <= length / 2) &
        (np.abs(right_coord) <= width / 2) &
        (height >= height_range[0]) &
        (height <= height_range[1])
    )[0]


def gaussians_in_body_volume(
    positions: np.ndarray,
    pose: RobotPose,
    gp: GroundPlane,
    *,
    length: float = GO2_BODY_LENGTH,
    width: float = GO2_BODY_WIDTH,
    height: float = GO2_BODY_HEIGHT,
) -> np.ndarray:
    """Indices of gaussians inside the robot's body volume.

    Used for collision detection: any gaussian in this volume => robot is
    physically inside something.
    """
    right = np.cross(gp.normal, pose.heading)
    right /= np.linalg.norm(right)
    return _rect_indices_on_plane(
        positions, pose.position, pose.heading, right,
        length, width, gp, height_range=(0.0, height),
    )


def gaussians_in_lookahead(
    positions: np.ndarray,
    pose: RobotPose,
    gp: GroundPlane,
    *,
    look_ahead_dist: float = 0.5,
    length: float = GO2_BODY_LENGTH,
    width: float = GO2_BODY_WIDTH,
    height_below_gp: float = 0.1,
    height_above_gp: float = 0.05,
) -> np.ndarray:
    """Indices of gaussians in a footprint-sized region AHEAD of the robot.

    This is the "where the robot's next body position will be" region. Used for
    semantic traversability: what class dominates the ground under this region.

    `height_below_gp` and `height_above_gp` bracket the ground plane to catch
    slightly-below (imperfect plane fit) and slightly-above (grass tips) gaussians.
    """
    right = np.cross(gp.normal, pose.heading)
    right /= np.linalg.norm(right)
    look_ahead_center = pose.position + look_ahead_dist * pose.heading
    return _rect_indices_on_plane(
        positions, look_ahead_center, pose.heading, right,
        length, width, gp,
        height_range=(-height_below_gp, height_above_gp),
    )


def gaussians_in_clearance_column(
    positions: np.ndarray,
    pose: RobotPose,
    gp: GroundPlane,
    *,
    look_ahead_dist: float = 0.5,
    length: float = GO2_BODY_LENGTH,
    width: float = GO2_BODY_WIDTH,
    max_ceiling: float = 2.0,
) -> np.ndarray:
    """Indices of gaussians in the VERTICAL COLUMN above the look-ahead footprint.

    Used for clearance: max-z of these gaussians tells us if a low-hanging
    obstacle (branch, awning) is in the way. Excludes gaussians on the ground
    itself.
    """
    right = np.cross(gp.normal, pose.heading)
    right /= np.linalg.norm(right)
    look_ahead_center = pose.position + look_ahead_dist * pose.heading
    return _rect_indices_on_plane(
        positions, look_ahead_center, pose.heading, right,
        length, width, gp,
        # Column starts ABOVE the ground (excluding ground gaussians) and goes up.
        height_range=(0.05, max_ceiling),
    )
