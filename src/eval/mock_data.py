"""Synthetic labeled gaussians + fake trajectory for Mac-local dev.

Lets us exercise the reward pipeline without needing the NeoVerse reconstructor
(GPU-only) to produce real gaussians. The scene layout:

    ------------------ tree (obstacle) at (2, 0)
                     |
    -----------------|-------------------  <-- trail (dirt)
                     |                     robot walks along +x
    ------------ vegetation at (4, -1) ----
    -------- grass everywhere else --------

Robot trajectory: walks from (0, 0) to (5, 0) along +x, one step every 0.5m.
Should get high semantic reward on the trail, drop when it passes the tree /
vegetation regions.

Use `make_mock_scene()` for gaussians+labels and `make_mock_trajectory()` for
the sequence of poses.
"""
from __future__ import annotations

import numpy as np

from .footprint import RobotPose
from .ground_plane import GroundPlane


# Class IDs — must match the taxonomy in config/traversability.yaml.
CID_DIRT = 2
CID_GRASS = 4
CID_TREE = 19
CID_VEGETATION = 20


def make_mock_scene(rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic outdoor scene: grass ground + a trail + a tree + a bush.

    Returns:
        positions: (N, 3) float32 world-frame gaussian centers
        labels:    (N,)   int32   class IDs
    """
    if rng is None:
        rng = np.random.default_rng(0)
    parts = []

    # -- Ground plane: grass, scattered over 20m x 20m ----------------------
    n_grass = 6000
    x = rng.uniform(-5.0, 15.0, n_grass)
    y = rng.uniform(-5.0, 5.0, n_grass)
    z = rng.normal(0.0, 0.02, n_grass)     # ~2cm surface noise
    parts.append((np.column_stack([x, y, z]), np.full(n_grass, CID_GRASS, dtype=np.int32)))

    # -- Trail: dirt strip along +x centered on y=0, width 1m -------------
    n_trail = 3000
    x = rng.uniform(0.0, 10.0, n_trail)
    y = rng.uniform(-0.5, 0.5, n_trail)
    z = rng.normal(-0.02, 0.01, n_trail)   # slightly lower than grass
    parts.append((np.column_stack([x, y, z]), np.full(n_trail, CID_DIRT, dtype=np.int32)))

    # -- Tree at (3, 0.5): trunk column ------------------------------------
    n_tree = 500
    x = rng.normal(3.0, 0.1, n_tree)
    y = rng.normal(0.5, 0.1, n_tree)
    z = rng.uniform(0.2, 4.0, n_tree)      # trunk goes up 4m
    parts.append((np.column_stack([x, y, z]), np.full(n_tree, CID_TREE, dtype=np.int32)))

    # -- Vegetation bush at (5, -0.8) --------------------------------------
    n_veg = 400
    x = rng.normal(5.0, 0.3, n_veg)
    y = rng.normal(-0.8, 0.3, n_veg)
    z = rng.uniform(0.0, 0.4, n_veg)       # bush 0-40cm tall
    parts.append((np.column_stack([x, y, z]), np.full(n_veg, CID_VEGETATION, dtype=np.int32)))

    positions = np.concatenate([p for p, _ in parts]).astype(np.float32)
    labels = np.concatenate([l for _, l in parts]).astype(np.int32)
    return positions, labels


def make_mock_ground_plane() -> GroundPlane:
    """Ground plane matching the mock scene (world XY plane, z=0)."""
    return GroundPlane(normal=np.array([0.0, 0.0, 1.0], dtype=np.float32), offset=0.0)


def make_mock_trajectory(n_steps: int = 20) -> list[RobotPose]:
    """Robot walks from (0, 0) to ~(10, 0) along +x, one step every 0.5m.

    Passes near the tree (at x=3) and the vegetation bush (at x=5). Reward
    should reflect: high on dirt, drops when tree/bush enters look-ahead.
    """
    poses = []
    step = 10.0 / (n_steps - 1)
    for i in range(n_steps):
        x = i * step
        pos = np.array([x, 0.0, 0.05], dtype=np.float32)   # ~5cm above ground
        heading = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        poses.append(RobotPose(position=pos, heading=heading))
    return poses


def mock_goal() -> np.ndarray:
    return np.array([10.0, 0.0, 0.0], dtype=np.float32)
