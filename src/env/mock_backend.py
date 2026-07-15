"""Mock world + semantic backends for local (Mac) RL development.

Idea: build a tiny "trail" scenario as a top-down grid of class IDs (grass in
the middle, trees on the sides, dirt path along +x). At each `render()` call,
ray-cast from the camera into the grid to produce (a) a colorized RGB view and
(b) the corresponding semantic label image. The SemanticBackend just reads back
the semantic image the world backend computed.

Enough to exercise the whole RL loop (SceneEnv + PPO) without needing the real
NeoVerse pipeline. When the real backend is ready (Milestone B), swap the
imports in the training script — SceneEnv doesn't care.

Scene layout (world x-forward, y-right, z-up):

    y=+3 ┤    trees        trees          trees         trees
         │   (class 19)    (class 19)    (class 19)   (class 19)
    y=+1 ┤   ─────────────────────────── (rocks around x=5 y=+1)
    y= 0 ┤ START → · · · · dirt · · · · · GOAL (x=8)
    y=-1 ┤   ─────────────────────────── (water at x=3 y=-1)
    y=-3 ┤    trees        trees          trees         trees
         └────────────────────────────────────────────
          x=-2                                          x=10

The robot walks along y≈0 through the dirt corridor. Steering off the middle
into grass or trees drops the reward. Hitting the rocks / water patches is a
collision.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .scene_env import WorldBackend

# Class IDs (must match config/traversability.yaml + palette.py).
CID_VOID       = 0
CID_SKY        = 1
CID_DIRT       = 2
CID_GRASS      = 4
CID_WATER      = 8
CID_ROCK       = 9
CID_TREE       = 19


class MockWorldBackend:
    """WorldBackend implementation for local dev — synthesises RGB + semantic
    from a hand-drawn top-down scene grid via ray-casting."""

    def __init__(self, H: int = 84, W: int = 84, seed: int = 0):
        self.H = H
        self.W = W
        self._rng = np.random.default_rng(seed)

        # ---- Scene grid ----
        # Resolution: 0.1m per grid cell. Extent covers 15m x 10m.
        self._grid_res = 0.1
        self._x_min, self._x_max = -2.0, 13.0     # 15m along x
        self._y_min, self._y_max = -5.0, 5.0      # 10m along y
        gx = int((self._x_max - self._x_min) / self._grid_res)
        gy = int((self._y_max - self._y_min) / self._grid_res)
        self._grid_shape = (gy, gx)
        self._scene_grid = self._build_scene_grid()   # (gy, gx) int class IDs

        # ---- Camera params ----
        self._camera_height = 0.4                 # meters
        self._camera_pitch_down = 0.05            # radians
        fx = fy = 0.9 * W                          # ~54° horizontal FOV
        cx = W / 2.0
        cy = H / 2.0
        self._K = np.array([[fx, 0.0, cx],
                            [0.0, fy, cy],
                            [0.0, 0.0, 1.0]], dtype=np.float32)

        # Static rotation for the extra pitch-down applied on top of the robot
        # rotation. Same math as load_clip._synthetic_trajectory.
        c, s = np.cos(self._camera_pitch_down), np.sin(self._camera_pitch_down)
        self._R_pitch = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)

        # Cache the latest semantic image so MockSemanticBackend can read it.
        self._last_semantic_image: Optional[np.ndarray] = None

    # ------------ WorldBackend API ------------

    def load_scene(self, scene_id: str) -> None:
        # Only one scene for now; no-op. Later: support multi-scene.
        pass

    def start_pose(self, scene_id: str) -> np.ndarray:
        # Robot at world origin, facing +x.
        return np.eye(4, dtype=np.float32)

    def goal_position(self, scene_id: str) -> np.ndarray:
        # Goal 8m ahead down the trail.
        return np.array([8.0, 0.0, 0.0], dtype=np.float32)

    def render(self, pose_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ray-cast the scene at the current pose; return RGB + K + w2c.

        Also stashes the semantic image for MockSemanticBackend to read.
        """
        # Build the actual camera pose: robot pose + height offset + pitch-down.
        robot_pos = pose_world[:3, 3]
        R_robot   = pose_world[:3, :3]
        cam_pos_world = robot_pos + np.array([0.0, 0.0, self._camera_height], dtype=np.float32)

        # cam-to-world = R_robot · standard_cam_orientation · R_pitch^T
        # standard_cam_orientation: cam_z aligns with world +x when robot yaw=0
        # We need: cam +z = robot heading (world x after robot rotation)
        #         cam +x = robot right
        #         cam +y = -world_up (down)
        R_std = np.array([[ 0.0,  1.0,  0.0],    # cam_x -> robot right = world +y (before rotation)
                          [ 0.0,  0.0, -1.0],    # cam_y -> -world_z (down)
                          [ 1.0,  0.0,  0.0]],   # cam_z -> world +x (forward)
                         dtype=np.float32)
        # First orient camera to face along robot heading (world +x rotated by R_robot),
        # then apply the fixed pitch-down.
        R_world_to_cam = self._R_pitch @ R_std @ R_robot.T
        t_world_to_cam = -R_world_to_cam @ cam_pos_world

        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R_world_to_cam
        w2c[:3, 3] = t_world_to_cam

        # Ray-cast every pixel to the ground plane and look up the class in the scene grid.
        semantic_image = self._raycast(cam_pos_world, R_world_to_cam)
        self._last_semantic_image = semantic_image

        # Colorize the semantic image as our RGB (each class -> its canonical color).
        from src.eval.palette import CLASS_COLORS_255
        rgb = CLASS_COLORS_255[np.clip(semantic_image, 0, len(CLASS_COLORS_255) - 1)]

        return rgb.astype(np.uint8), self._K.copy(), w2c

    # ------------ scene construction ------------

    def _build_scene_grid(self) -> np.ndarray:
        """Top-down class-ID grid. Origin at (self._x_min, self._y_min)."""
        gy, gx = self._grid_shape
        grid = np.full((gy, gx), CID_GRASS, dtype=np.int32)     # default = grass

        # Dirt trail down the middle (y in [-0.7, +0.7])
        y_center_lo = self._world_y_to_grid(-0.7)
        y_center_hi = self._world_y_to_grid(+0.7)
        grid[y_center_lo:y_center_hi + 1, :] = CID_DIRT

        # Trees on the far sides (|y| > 3)
        y_edge_top = self._world_y_to_grid(3.0)
        y_edge_bot = self._world_y_to_grid(-3.0)
        grid[y_edge_top:, :] = CID_TREE
        grid[:y_edge_bot + 1, :] = CID_TREE

        # Rocks patch around (x=5, y=+1)
        self._paint_disk(grid, cx=5.0, cy=1.0, r=0.4, class_id=CID_ROCK)
        # Water patch around (x=3, y=-1)
        self._paint_disk(grid, cx=3.0, cy=-1.0, r=0.5, class_id=CID_WATER)

        return grid

    def _paint_disk(self, grid: np.ndarray, cx: float, cy: float, r: float, class_id: int) -> None:
        """Paint a filled disk of `class_id` centered at world (cx, cy) with radius r."""
        gy, gx = grid.shape
        for iy in range(gy):
            world_y = self._grid_y_to_world(iy)
            if abs(world_y - cy) > r:
                continue
            for ix in range(gx):
                world_x = self._grid_x_to_world(ix)
                if (world_x - cx) ** 2 + (world_y - cy) ** 2 <= r ** 2:
                    grid[iy, ix] = class_id

    def _world_x_to_grid(self, world_x: float) -> int:
        return int((world_x - self._x_min) / self._grid_res)

    def _world_y_to_grid(self, world_y: float) -> int:
        return int((world_y - self._y_min) / self._grid_res)

    def _grid_x_to_world(self, ix: int) -> float:
        return self._x_min + ix * self._grid_res

    def _grid_y_to_world(self, iy: int) -> float:
        return self._y_min + iy * self._grid_res

    # ------------ ray-casting ------------

    def _raycast(self, cam_pos_world: np.ndarray, R_world_to_cam: np.ndarray) -> np.ndarray:
        """Cast rays from cam through every pixel; look up scene class at ground hit.

        Vectorized: no per-pixel Python loop.
        """
        H, W = self.H, self.W
        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]

        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        # Ray in camera frame (each pixel).
        rays_cam = np.stack([
            (us - cx) / fx,
            (vs - cy) / fy,
            np.ones_like(us, dtype=np.float32),
        ], axis=-1).astype(np.float32)                 # (H, W, 3)

        # Transform to world: ray_world = R_world_to_cam.T @ ray_cam
        R_cam_to_world = R_world_to_cam.T
        rays_world = rays_cam @ R_cam_to_world.T       # (H, W, 3)

        # Intersect z = 0 plane: cam_z + t * ray_z_world = 0 => t = -cam_z / ray_z_world
        ray_z_world = rays_world[..., 2]
        cam_z = cam_pos_world[2]

        # Only rays going down (ray_z_world < 0) hit the ground when camera is above.
        hits_ground = (ray_z_world < -1e-4) & (cam_z > 0)
        # For rays that DON'T hit the ground we still need a finite `t` value so the
        # downstream arithmetic doesn't emit inf/nan warnings — 0 is fine because
        # `hits_ground` masks those pixels out at the very end anyway.
        safe_denom = np.where(hits_ground, ray_z_world, -1.0)
        t = np.where(hits_ground, -cam_z / safe_denom, 0.0)

        # Ground hit points in world coords.
        hit_x = cam_pos_world[0] + t * rays_world[..., 0]
        hit_y = cam_pos_world[1] + t * rays_world[..., 1]

        # Look up class in scene grid.
        gy, gx_ = self._grid_shape
        ix = ((hit_x - self._x_min) / self._grid_res).astype(np.int32)
        iy = ((hit_y - self._y_min) / self._grid_res).astype(np.int32)
        in_bounds = (ix >= 0) & (ix < gx_) & (iy >= 0) & (iy < gy)
        ix_clip = np.clip(ix, 0, gx_ - 1)
        iy_clip = np.clip(iy, 0, gy - 1)

        semantic = np.full((H, W), CID_SKY, dtype=np.int32)     # sky where no ground hit
        ground_mask = hits_ground & in_bounds
        semantic[ground_mask] = self._scene_grid[iy_clip[ground_mask], ix_clip[ground_mask]]
        return semantic


class MockSemanticBackend:
    """SemanticBackend that reads back the label image the mock world just rendered.

    Cheats via a reference to the mock world backend — fine for local dev, and
    aligns with how real SAM3 will get labels from the diffusion output at each step.
    """

    def __init__(self, world_backend: MockWorldBackend):
        self._world = world_backend

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        if self._world._last_semantic_image is None:
            raise RuntimeError("MockSemanticBackend.segment called before world.render()")
        return self._world._last_semantic_image
