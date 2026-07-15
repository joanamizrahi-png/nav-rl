"""Fit a ground plane to a set of 3D gaussian positions.

Why: "under the robot" and "clearance above the robot" only make sense once we
know which way is up. We fit a plane to the gaussians that are most likely to
be ground (lowest ~30% by z-coordinate), using RANSAC to be robust to outliers
(e.g. gaussians on the underside of overhangs).

Alternative we may switch to later: if NeoVerse's reconstructor outputs gaussians
in a gravity-aligned world frame (z-axis = up, ground at z ~ 0), we can skip
this entirely and just use `z` directly. To be verified when we start using
real reconstructor output.

Returns a `GroundPlane` with `normal` (unit vector, points UP) and `offset`
(signed distance from origin such that `normal @ point + offset = 0` on the plane).
Height-above-ground of any point: `normal @ point + offset` (positive means above).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class GroundPlane:
    normal: np.ndarray   # shape (3,), unit vector pointing UP (away from ground)
    offset: float        # scalar; plane equation: normal @ point + offset = 0

    def height_above(self, points: np.ndarray) -> np.ndarray:
        """Signed distance above the ground plane for each point.

        Args:
            points: (N, 3) array of 3D positions.
        Returns:
            (N,) array of heights (positive = above ground, negative = below).
        """
        return points @ self.normal + self.offset


def fit_ground_plane_ransac(
    positions: np.ndarray,
    *,
    lowest_frac: float = 0.30,
    ransac_iters: int = 200,
    inlier_thresh: float = 0.05,   # meters — a point is "on" the plane if within 5cm
    rng: Optional[np.random.Generator] = None,
) -> GroundPlane:
    """Fit a plane to the lowest `lowest_frac` of gaussians by naive z.

    RANSAC picks 3 random points, fits a plane, counts inliers within
    `inlier_thresh`. Keeps the plane with the most inliers.

    Assumes the raw z-axis is roughly aligned with "down" — good enough as a
    starting heuristic for outdoor scenes where most gaussians ARE ground level.

    Args:
        positions: (N, 3) gaussian positions in world frame.
        lowest_frac: fraction of lowest-z gaussians to use as candidate ground.
        ransac_iters: number of random 3-point plane trials.
        inlier_thresh: meters; distance under which a point counts as on-plane.

    Returns:
        GroundPlane with normal oriented so the majority of gaussians are ABOVE.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(positions)
    if n < 100:
        raise ValueError(f"need >= 100 gaussians to fit ground plane, got {n}")

    # Take the lowest-z fraction as the candidate ground subset.
    z_sorted_idx = np.argsort(positions[:, 2])
    k = int(lowest_frac * n)
    ground_candidates = positions[z_sorted_idx[:k]]

    best_normal = None
    best_offset = None
    best_inliers = -1

    for _ in range(ransac_iters):
        # Sample 3 non-collinear points
        idx = rng.choice(len(ground_candidates), 3, replace=False)
        p0, p1, p2 = ground_candidates[idx]
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue  # collinear
        normal = normal / norm
        offset = -normal @ p0

        # Count inliers over the full population (not just candidates) — the plane
        # should agree with most gaussians in the scene, not just the ones we picked.
        distances = np.abs(positions @ normal + offset)
        n_inliers = int((distances < inlier_thresh).sum())

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_normal = normal
            best_offset = offset

    if best_normal is None:
        raise RuntimeError("RANSAC failed to find any valid plane")

    # Orient the normal so that the MAJORITY of gaussians are above the plane
    # (positive height). Otherwise our sign convention is backwards.
    signed = positions @ best_normal + best_offset
    if (signed < 0).sum() > (signed > 0).sum():
        best_normal = -best_normal
        best_offset = -best_offset

    return GroundPlane(normal=best_normal, offset=float(best_offset))
