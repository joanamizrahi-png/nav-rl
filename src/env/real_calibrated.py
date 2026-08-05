"""Metric-calibrated real backend + Gaussian-label semantic backend (Milestone B/C glue).

Adds two things on top of real_backend.RealWorldBackend, using artifacts we
already produced and validated:

1. CalibratedRealWorldBackend — SceneEnv talks METERS in a gravity-aligned
   "nav frame" (x/y ground plane, z up, ground at z=0 — the same frame
   scripts/extract_poses.py writes poses.npz in). This backend converts nav-frame
   robot poses to recon-frame camera poses using the per-scene calibration
   stored in that npz (plane normal/offset + metric scale), so
   SceneEnvConfig.step_size_m, look_ahead_dist, and the GO2 body dims are all
   REAL meters. start_pose()/goal_position() come from the clip's real
   trajectory (start = frame 0, goal = a configurable frame index's position).

2. GaussianLabelBackend — SemanticBackend for the reward that NEVER looks at
   generated RGB. On each render, the world backend also rasterizes the
   `labels` feature from the SAM3-labeled Gaussians at the same pose (real
   observed geometry -> projected class map). Holey/unseen pixels come out as
   void, which the traversability table already treats as collision-worthy →
   "unknown = risky" keeps the policy inside the reconstructed volume, where
   observations are trustworthy too.

The math here inverts scripts/extract_poses.poses_from_c2w_recon exactly:
    forward:  p_nav = s * (R_up @ R_rs @ p_recon - [0, 0, ground_z])
    inverse:  p_recon = (R_up @ R_rs)^T @ (p_nav / s + [0, 0, ground_z])
Verified by round-trip test (see scripts/test_calibration.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .real_backend import RealWorldBackend, RealWorldBackendConfig, R_SCENE_TO_RECON


def _rotation_aligning_to_z(normal: np.ndarray) -> np.ndarray:
    """Same Rodrigues construction as scripts/extract_poses.py (kept in sync)."""
    n = normal / np.linalg.norm(normal)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.clip(n @ z, -1.0, 1.0))
    if c > 1.0 - 1e-8:
        return np.eye(3)
    if c < -1.0 + 1e-8:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(n, z)
    axis /= np.linalg.norm(axis)
    s = np.sqrt(1.0 - c * c)
    Kx = np.array([[0.0, -axis[2], axis[1]],
                   [axis[2], 0.0, -axis[0]],
                   [-axis[1], axis[0], 0.0]])
    return np.eye(3) + s * Kx + (1.0 - c) * (Kx @ Kx)


@dataclass
class NavCalibration:
    """Per-scene nav(metric, z-up, ground z=0) <-> recon(normalized) transform."""
    A: np.ndarray            # (3,3) = R_up @ R_rs : recon -> aligned-scene rotation
    ground_z: float          # aligned-scene z of the ground plane (pre-scale)
    scale: float             # meters per recon unit
    positions: np.ndarray    # (T,3) real robot trajectory in nav frame (meters)
    headings: np.ndarray     # (T,3)
    camera_height_m: float   # camera mount height used at extraction

    @classmethod
    def from_npz(cls, path: str | Path) -> "NavCalibration":
        d = np.load(path)
        R_rs = R_SCENE_TO_RECON.T.astype(np.float64)
        R_up = _rotation_aligning_to_z(d["plane_normal_scene"].astype(np.float64))
        return cls(
            A=R_up @ R_rs,
            ground_z=float(-d["plane_offset_scene"]),
            scale=float(d["scale_m_per_unit"]),
            positions=d["positions"].astype(np.float64),
            headings=d["headings"].astype(np.float64),
            camera_height_m=float(d["camera_height_m"]) if "camera_height_m" in d else 0.25,
        )

    # ---- point maps ----
    def nav_to_recon_point(self, p_nav: np.ndarray) -> np.ndarray:
        return self.A.T @ (p_nav / self.scale + np.array([0.0, 0.0, self.ground_z]))

    def recon_to_nav_point(self, p_recon: np.ndarray) -> np.ndarray:
        return self.scale * (self.A @ p_recon - np.array([0.0, 0.0, self.ground_z]))

    # ---- pose maps ----
    def nav_cam_to_recon_cam(self, c2w_nav: np.ndarray) -> np.ndarray:
        """(4,4) camera-to-world in nav frame -> camera-to-world in recon frame."""
        out = np.eye(4)
        out[:3, :3] = self.A.T @ c2w_nav[:3, :3]
        out[:3, 3] = self.nav_to_recon_point(c2w_nav[:3, 3])
        return out

    def nav_world_to_recon_homogeneous(self) -> np.ndarray:
        """(4,4) M with X_recon_h = M @ X_nav_h (linear part includes 1/scale).
        Composing w2c_recon @ M gives a w2c usable on nav-frame points: the
        uniform 1/scale on xyz cancels in the pinhole division, so pixel
        coordinates are exact."""
        M = np.eye(4)
        M[:3, :3] = self.A.T / self.scale
        M[:3, 3] = self.A.T @ np.array([0.0, 0.0, self.ground_z])
        return M

    def robot_pose_nav(self, frame: int) -> np.ndarray:
        """(4,4) robot-to-nav-world pose from the real trajectory (z=0, yaw from heading)."""
        fwd = self.headings[frame].copy()
        fwd[2] = 0.0
        fwd /= max(np.linalg.norm(fwd), 1e-8)
        up = np.array([0.0, 0.0, 1.0])
        pose = np.eye(4)
        # SUBTLE (debugged 2026-07-20 via black replay frames): the nav frame is
        # MIRRORED — real_backend's R_SCENE_TO_RECON has det=-1 (its "x fwd,
        # y right, z up" scene convention is left-handed), so every stored c2w
        # in the poses npz is an improper rotation. Robot poses must match that
        # handedness or the composed recon-frame camera comes out det=-1 and
        # gsplat renders garbage/black. Hence +y = fwd x up (det -1, matching
        # the frame), NOT the right-handed up x fwd. Verified against the npz's
        # own c2w: <=9 deg per-axis error (residual = real camera pitch).
        # Deployment note: left/right yaw sign must be mirrored when
        # transferring actions to a real robot.
        pose[:3, 0] = fwd
        pose[:3, 1] = np.cross(fwd, up)
        pose[:3, 2] = up
        pose[:3, 3] = self.positions[frame] * np.array([1.0, 1.0, 0.0])
        return pose.astype(np.float32)


@dataclass
class CalibratedBackendConfig(RealWorldBackendConfig):
    # scene_id -> extract_poses npz (metric calibration + real trajectory)
    scene_poses_paths: dict = field(default_factory=dict)
    # scene_id -> SAM3 label npz (labels attach to Gaussians -> reward rasterization)
    scene_labels_paths: dict = field(default_factory=dict)
    # goal = real-trajectory position at this frame index (metric nav frame)
    goal_frame: int = 45
    # spawn curriculum: cap the spawn range. None = anywhere up to goal_frame-5
    # (short-range task); small value (e.g. 3) = full traverses from the start.
    spawn_max_frame: "int | None" = None
    # rung 6: sample the goal frame per EPISODE from this inclusive range instead
    # of the fixed goal_frame. Spawns then range over the whole trajectory, so
    # the policy also sees goals BEHIND it (learns to turn around / not overshoot).
    goal_frame_range: "tuple[int, int] | None" = None
    # rung 7 (multi-scene): max scene caches resident on GPU at once; older
    # ones are evicted to CPU and lazily moved back when their scene activates.
    # Each rasterizer-only cache is well under 1 GB, so a few coexist fine —
    # the cap is a guardrail, not a bottleneck.
    max_gpu_scenes: int = 2
    # v6d: spawn-goal separation guard is now a config knob. 1.0 was v6b/v6c's
    # close-range-exposure value; 1.5 is v6's (the only config that has learned
    # this task). Suspect in the v6c post-mortem — knob added to test it.
    goal_min_sep_m: float = 1.0


class CalibratedRealWorldBackend(RealWorldBackend):
    """RealWorldBackend that speaks the metric nav frame and rasterizes labels.

    render(pose) takes a (4,4) ROBOT-to-nav-world pose in METERS. Returns
    (rgb, K, w2c_nav) where w2c_nav projects nav-frame points to pixels —
    exactly what SceneEnv's reward needs. After every render, the label map
    rasterized from the Gaussians at the same pose is stored in
    `_last_semantic_image` for GaussianLabelBackend.
    """

    def __init__(self, cfg: CalibratedBackendConfig):
        super().__init__(cfg)
        self.cfg: CalibratedBackendConfig = cfg
        self._calib: dict[str, NavCalibration] = {}
        self._last_semantic_image: Optional[np.ndarray] = None

    # ---------- scene registration ----------

    def load_scene(self, scene_id: str) -> None:
        if scene_id not in self._calib:
            poses_path = self.cfg.scene_poses_paths.get(scene_id)
            if not poses_path:
                raise ValueError(f"no poses npz configured for scene {scene_id}")
            self._calib[scene_id] = NavCalibration.from_npz(poses_path)
        # Parent calls _reconstruct_scene(video_path) without the scene id;
        # stash it so the label-attachment override can find the labels npz.
        self._current_loading_scene_id = scene_id
        super().load_scene(scene_id)
        if self.cfg.render_mode == "rasterizer_only":
            self._activate_scene_on_gpu(scene_id)

    _gpu_lru: "list | None" = None

    def _activate_scene_on_gpu(self, scene_id: str) -> None:
        """Rung 7 residency management: ensure the active scene's cache is on
        the GPU; evict least-recently-used caches to CPU past max_gpu_scenes.
        (Diffusion mode keeps its own stricter discipline — everything CPU.)"""
        import torch
        from .real_backend import _move_tree_to
        if self._reconstructor is None:
            return
        device = next(self._reconstructor.parameters()).device
        if device.type == "cpu":
            return
        scene = self._cache[scene_id]
        if torch.is_tensor(scene.get("K")) and scene["K"].device.type == "cpu":
            self._cache[scene_id] = _move_tree_to(scene, device)
        if self._gpu_lru is None:
            self._gpu_lru = []
        self._gpu_lru = [s for s in self._gpu_lru if s != scene_id and s in self._cache]
        self._gpu_lru.append(scene_id)
        while len(self._gpu_lru) > max(1, int(self.cfg.max_gpu_scenes)):
            victim = self._gpu_lru.pop(0)
            self._cache[victim] = _move_tree_to(self._cache[victim], "cpu")
            torch.cuda.empty_cache()

    def start_pose(self, scene_id: str) -> np.ndarray:
        return self._calib[scene_id].robot_pose_nav(0)

    def sample_start_pose(self, scene_id: str, rng) -> np.ndarray:
        """Spawn curriculum: a random pose along the REAL trajectory, upstream
        of the goal (so some spawns are near it -> early successes to learn from)."""
        cal = self._calib[scene_id]
        if self.cfg.goal_frame_range is not None:
            hi = len(cal.positions) - 6      # goal varies: spawn anywhere on the trail
        else:
            hi = min(self.cfg.goal_frame - 5, len(cal.positions) - 6)
        hi = max(1, hi)
        if self.cfg.spawn_max_frame is not None:
            hi = max(1, min(hi, self.cfg.spawn_max_frame))
        return cal.robot_pose_nav(int(rng.integers(0, hi)))

    def goal_position(self, scene_id: str) -> np.ndarray:
        cal = self._calib[scene_id]
        frame = min(self.cfg.goal_frame, len(cal.positions) - 1)
        goal = cal.positions[frame].copy()
        goal[2] = 0.0
        return goal.astype(np.float32)

    def sample_goal_position(self, scene_id: str, rng, spawn_xy,
                             min_sep_m: "float | None" = None) -> np.ndarray:
        if min_sep_m is None:
            min_sep_m = self.cfg.goal_min_sep_m
        """Per-episode goal (rung 6). With goal_frame_range set, draw a frame
        uniformly, rejecting draws closer than min_sep_m to the spawn (those
        episodes are pre-won and teach nothing). Range unset -> fixed goal."""
        if self.cfg.goal_frame_range is None:
            return self.goal_position(scene_id)
        cal = self._calib[scene_id]
        lo, hi = self.cfg.goal_frame_range
        hi = min(hi, len(cal.positions) - 1)
        goal = None
        for _ in range(20):
            frame = int(rng.integers(lo, hi + 1))
            goal = cal.positions[frame].copy()
            goal[2] = 0.0
            if np.linalg.norm(goal[:2] - np.asarray(spawn_xy)) >= min_sep_m:
                break
        return goal.astype(np.float32)

    # ---------- label attachment (SAM3 -> Gaussians) ----------

    def _reconstruct_scene(self, video_path: str) -> dict:
        """Same as the parent, plus: attach the scene's SAM3 labels to the views
        BEFORE reconstruction so the Gaussians carry class ids and the labels
        feature can be rasterized for the reward."""
        import torch

        # Parent builds views internally; easiest robust hook is to temporarily
        # wrap load_video is invasive — instead reimplement the small view-build
        # with labels, then call the parent's heavy path via monkey-free copy.
        # To keep this maintainable we call the parent and then re-reconstruct
        # ONLY if labels are configured (labels change the gaussians).
        scene_id = self._current_loading_scene_id
        labels_path = self.cfg.scene_labels_paths.get(scene_id)
        if labels_path is None:
            return super()._reconstruct_scene(video_path)

        from torchvision.transforms import functional as F
        from diffsynth.utils.auxiliary import load_video

        cfg = self.cfg
        reconstructor = self._reconstructor
        device = next(reconstructor.parameters()).device
        dtype = next(reconstructor.parameters()).dtype

        images = load_video(video_path, cfg.num_frames,
                            resolution=(cfg.W, cfg.H), resize_mode="center_crop",
                            static_scene=False)
        labels = np.load(labels_path)["labels"]
        n = min(len(images), len(labels))
        views = {
            "img": torch.stack([F.to_tensor(im)[None] for im in images[:n]], dim=1).to(device),
            "is_target": torch.zeros((1, n), dtype=torch.bool, device=device),
            "is_static": torch.zeros((1, n), dtype=torch.bool, device=device),
            "timestamp": torch.arange(0, n, dtype=torch.int64, device=device).unsqueeze(0),
            "labels": torch.as_tensor(labels[:n], dtype=torch.long, device=device).unsqueeze(0),
        }
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
            predictions = reconstructor(views, is_inference=True, use_motion=False)
        cache = {
            "gaussians": predictions["splats"],
            "K": predictions["rendered_intrinsics"][0],
            "cam2world": predictions["rendered_extrinsics"][0],
            "timestamps": predictions["rendered_timestamps"][0],
            "views": views,
        }
        del predictions
        # MEMORY DISCIPLINE (parity with the parent class, lost in this override
        # and cause of the probe OOM): in diffusion mode the scene — including
        # the hefty one-hot label tensors — must live on CPU so the ~30GB Wan
        # pipeline can load; renders move it back per call.
        if self.cfg.render_mode == "rasterizer_plus_diffusion":
            from .real_backend import _move_tree_to
            import gc
            cache = _move_tree_to(cache, "cpu")
            gc.collect()
        torch.cuda.empty_cache()
        return cache

    _current_loading_scene_id: Optional[str] = None

    # ---------- rendering in the nav frame ----------

    def render(self, pose_nav_robot: np.ndarray):
        import torch
        from diffsynth.utils.auxiliary import homo_matrix_inverse

        scene = self._cache.get(self._current_scene_id)
        if scene is None:
            raise RuntimeError("call load_scene() first")
        cal = self._calib[self._current_scene_id]

        # robot(nav) -> camera(nav): lift by mount height, robot->camera axes.
        c2w_nav = pose_nav_robot.astype(np.float64).copy()
        c2w_nav[:3, 3] += np.array([0.0, 0.0, cal.camera_height_m])
        # Columns = camera local axes expressed in ROBOT axes (robot: x fwd,
        # y LEFT, z up; camera: x right, y down, z fwd):
        #   cam_x (right) = -robot_y -> col0 (0,-1,0)
        #   cam_y (down)  = -robot_z -> col1 (0,0,-1)
        #   cam_z (fwd)   =  robot_x -> col2 (1,0,0)
        R_cam_local = np.array([
            [0.0,  0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ])
        c2w_nav[:3, :3] = c2w_nav[:3, :3] @ R_cam_local

        # nav camera pose -> recon camera pose.
        pose_recon = cal.nav_cam_to_recon_cam(c2w_nav)

        # TIME MATTERS (debugged 2026-07-21 via panning-then-black replay): the
        # scene is reconstructed as DYNAMIC — Gaussians are time-associated, and
        # rendering everything at timestamp 0 culls content that belongs to later
        # moments (works near frame 0's position, black further along). Render at
        # the timestamp of the NEAREST source-trajectory point instead: scene
        # content is densest at the time the camera actually stood nearby.
        t_idx = int(np.argmin(np.linalg.norm(
            cal.positions[:, :2] - pose_nav_robot[:2, 3], axis=1)))

        if self.cfg.render_mode == "rasterizer_only":
            # Single-frame rasterization (the parent broadcasts to all 81 frames
            # for the diffusion path — 81x wasted work per RL step here).
            rgb, K, w2c_recon = self._rasterize_single(scene, pose_recon, t_idx)
        else:
            rgb, K, w2c_recon = self._rasterize_and_diffuse(scene, pose_recon.astype(np.float32))

        # Label rasterization at the SAME pose + time (reward source; never generated RGB).
        self._last_semantic_image = self._rasterize_labels(scene, pose_recon, t_idx)

        # Return a w2c usable on NAV-frame (metric) points.
        w2c_nav = w2c_recon.astype(np.float64) @ cal.nav_world_to_recon_homogeneous()
        return rgb, K, w2c_nav.astype(np.float32)

    def _single_frame_inputs(self, scene: dict, pose_recon: np.ndarray, t_idx: int):
        import torch
        from diffsynth.utils.auxiliary import homo_matrix_inverse
        device = next(self._reconstructor.parameters()).device
        c2w = torch.from_numpy(pose_recon.astype(np.float32)).to(
            device=device, dtype=scene["cam2world"].dtype).unsqueeze(0)
        w2c = homo_matrix_inverse(c2w)
        t_idx = int(np.clip(t_idx, 0, len(scene["timestamps"]) - 1))
        return w2c, scene["K"][0:1], scene["timestamps"][t_idx:t_idx + 1]

    def _rasterize_single(self, scene: dict, pose_recon: np.ndarray, t_idx: int = 0):
        import torch
        w2c, K1, ts1 = self._single_frame_inputs(scene, pose_recon, t_idx)
        rgb, _, _ = self._reconstructor.gs_renderer.rasterizer.forward(
            scene["gaussians"], render_viewmats=[w2c], render_Ks=[K1],
            render_timestamps=[ts1], sh_degree=0, width=self.W, height=self.H,
        )
        rgb_np = (rgb[0, 0].detach().clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
        return rgb_np, K1[0].detach().cpu().float().numpy(), w2c[0].detach().cpu().float().numpy()

    def _rasterize_labels(self, scene: dict, pose_recon: np.ndarray, t_idx: int = 0) -> np.ndarray:
        import torch
        w2c, K1, ts1 = self._single_frame_inputs(scene, pose_recon, t_idx)
        sem_probs, _, _ = self._reconstructor.gs_renderer.rasterizer.forward(
            scene["gaussians"], render_viewmats=[w2c], render_Ks=[K1],
            render_timestamps=[ts1], sh_degree=0, width=self.W, height=self.H,
            feature="labels",
        )
        return sem_probs[0, 0].argmax(dim=-1).to(torch.int32).cpu().numpy()


class GaussianLabelBackend:
    """SemanticBackend whose labels come from the 3D Gaussians, not from RGB.

    Real observed geometry projected at the current pose. Unseen regions ->
    void -> treated as collision-worthy by the traversability table, which
    keeps the policy inside the reconstructed volume.
    """
    def __init__(self, world_backend: CalibratedRealWorldBackend):
        self._world = world_backend

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        if self._world._last_semantic_image is None:
            raise RuntimeError("GaussianLabelBackend.segment called before world.render()")
        return self._world._last_semantic_image
