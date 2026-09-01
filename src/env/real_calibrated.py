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
        # YAW-MIRROR FIX (2026-08-13): the scene convention behind the npz is
        # left-handed ("x fwd, y RIGHT, z up"), which made the nav frame
        # left-handed too — commanded turn-left rendered as a visual
        # turn-right (confirmed by pose_direction_check sweep videos on two
        # scenes; the 07-20 deployment note predicted it). Fix: flip nav y
        # ONCE here, at the calibration boundary (nav' = F·nav,
        # F = diag(1,-1,1)). The flip cancels in every nav→recon composition
        # (A'ᵀ·F·F = Aᵀ on points, and the composed camera rotation is
        # column-for-column identical — verified by hand), so rendering is
        # bit-identical for the same physical pose. But the nav frame is now
        # right-handed: +yaw about +z MEANS turn-left in the image, and
        # real-robot action transfer needs no sign mirroring.
        # POLICY CHECKPOINTS trained before this fix learned the mirrored
        # convention — retrain (v14) before evaluating them.
        F = np.diag([1.0, -1.0, 1.0])
        return cls(
            A=F @ (R_up @ R_rs),
            ground_z=float(-d["plane_offset_scene"]),
            scale=float(d["scale_m_per_unit"]),
            positions=d["positions"].astype(np.float64) @ F,
            headings=d["headings"].astype(np.float64) @ F,
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
        # Right-handed since the 2026-08-13 yaw-mirror fix (see from_npz):
        # +y = up × fwd, det = +1. The old left-handed +y = fwd × up (needed
        # while the nav frame itself was mirrored — see the 07-20 note in git
        # history) composed with the flipped A to the SAME recon camera, so
        # rendering is unchanged; only the meaning of +yaw is now physical.
        pose[:3, 0] = fwd
        pose[:3, 1] = np.cross(up, fwd)
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
    # Designed obstacle/detour tests: pin the goal to an arbitrary nav-frame
    # (x, y) instead of a trajectory frame — e.g. just past a tree, so the
    # straight line crosses it. Overrides goal_frame/goal_frame_range.
    goal_xy_override: "tuple[float, float] | None" = None
    # RW5-v2 (advisor spec): fixed spawn->goal distance. When set, the goal
    # frame is chosen (within goal_frame_range) so the straight-line distance
    # from the spawn is as close as possible to this many meters.
    goal_dist_m: "float | None" = None
    # J-spec (2026-08-31, Jing meeting): goals at a random bearing (full 360,
    # NOT on the trajectory) and a random distance drawn from goal_dist_range.
    # Goals may land on non-traversable ground BY DESIGN — with the strict
    # traversability table the optimal policy walks the sidewalk to the edge
    # nearest the goal and stops. The robot itself stays on-corridor, so the
    # world model is never asked to render where reconstruction is empty.
    goal_dir_360: bool = False
    goal_dist_range: "tuple | None" = None      # (lo_m, hi_m), e.g. (5, 10)
    # Cone constraint (2026-08-31, her spin sweep verdict: single-pass capture
    # only supports a forward viewing cone — backward views render the backs
    # of one-sided splats and the reward labels there are garbage). Goals are
    # sampled within +-goal_cone_deg/2 of the path tangent at the spawn.
    # 360 = unconstrained (the original J-spec; valid once pano scenes land).
    goal_cone_deg: float = 360.0
    # Pano side views are OPT-IN (her rule 2026-08-31: keep the dense-data
    # refinement and the pano experiment SEPARATE while pano is unproven —
    # auto-append on file presence would silently confound them).
    use_pano_views: bool = False
    # Spawn jitter (her J-v2 spec 2026-08-31): rotate/offset the spawn pose so
    # the policy can't equate "diverged from spawn axis" with "bad terrain" —
    # it must learn to stop at GRASS, not at deviation. Values certified by
    # the cold-start jitter walks (sy20/sl0.4) and the strafe cov sweep.
    spawn_yaw_jitter_deg: float = 0.0
    spawn_lat_jitter_m: float = 0.0
    # Every Nth pano side-view frame joins the reconstruction (243 full views
    # OOM the gs_head; 3 -> 81+27+27=135). Raise to 4-5 if OOM persists.
    pano_view_stride: int = 3
    # Spawn validity (her check): only spawn on frames whose ground patch is
    # labeled with one of these class ids (e.g. (6, 8) = sidewalk/pavement).
    # None = no filtering (all runs before this).
    spawn_label_classes: "tuple | None" = None
    # Keep spawns out of the weak-reconstruction zone at the clip's edges
    # (live generation confabulates there — measured 2026-08-29).
    spawn_min_frame: int = 0


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
        of the goal (so some spawns are near it -> early successes to learn from).

        spawn_min_frame (2026-08-29): the reconstruction is weakest at the
        clip's edges (few overlapping views) and live generation there
        confabulates (the girl/sheep frames) — keep spawns out of that zone."""
        cal = self._calib[scene_id]
        lo = max(0, int(self.cfg.spawn_min_frame))
        if self.cfg.goal_frame_range is not None:
            hi = len(cal.positions) - 6      # goal varies: spawn anywhere on the trail
        else:
            hi = min(self.cfg.goal_frame - 5, len(cal.positions) - 6)
        hi = max(lo + 1, hi)
        if self.cfg.spawn_max_frame is not None:
            hi = max(lo + 1, min(hi, self.cfg.spawn_max_frame))
        ok = self._spawn_ok_frames(scene_id)
        if ok is not None:
            cand = [f for f in range(lo, hi) if ok[f]]
            if cand:
                return self._jitter_spawn(
                    cal.robot_pose_nav(int(cand[int(rng.integers(0, len(cand)))])), rng)
            print(f"[spawn_label_classes] WARNING: no valid spawn frames in "
                  f"[{lo},{hi}) for {scene_id}; falling back to unfiltered")
        return self._jitter_spawn(
            cal.robot_pose_nav(int(rng.integers(lo, hi))), rng)

    def _jitter_spawn(self, pose: np.ndarray, rng) -> np.ndarray:
        """Anti-memorization spawn jitter (her spec): rotate heading by
        U(-yaw,+yaw) and slide laterally by U(-lat,+lat). Identity when both
        knobs are 0 (every run before 2026-08-31)."""
        jy = float(getattr(self.cfg, "spawn_yaw_jitter_deg", 0.0))
        jl = float(getattr(self.cfg, "spawn_lat_jitter_m", 0.0))
        if jy <= 0.0 and jl <= 0.0:
            return pose
        pose = pose.copy()
        if jy > 0.0:
            a = float(rng.uniform(-1.0, 1.0)) * float(np.deg2rad(jy))
            c, s = np.cos(a), np.sin(a)
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                          dtype=pose.dtype)
            pose[:3, :3] = Rz @ pose[:3, :3]
        if jl > 0.0:
            fwd = pose[:3, 0]
            lat = np.array([-fwd[1], fwd[0], 0.0], dtype=pose.dtype)
            n = float(np.linalg.norm(lat))
            if n > 1e-6:
                pose[:3, 3] += (float(rng.uniform(-1.0, 1.0)) * jl / n) * lat
        return pose

    _spawn_ok_cache: "dict | None" = None

    def _spawn_ok_frames(self, scene_id: str):
        """Per-frame spawn validity from the scene's label npz: the ground
        patch under the camera (bottom-center of the frame) must be one of
        cfg.spawn_label_classes. Computed once per scene. None = filter off."""
        if self.cfg.spawn_label_classes is None:
            return None
        if self._spawn_ok_cache is None:
            self._spawn_ok_cache = {}
        if scene_id in self._spawn_ok_cache:
            return self._spawn_ok_cache[scene_id]
        labels_path = self.cfg.scene_labels_paths.get(scene_id)
        if labels_path is None:
            self._spawn_ok_cache[scene_id] = None
            return None
        labels = np.load(labels_path)["labels"]          # [T, H, W]
        H, W = labels.shape[-2:]
        patch = labels[:, int(H * 0.8):, int(W * 0.35):int(W * 0.65)]
        allowed = set(int(c) for c in self.cfg.spawn_label_classes)
        ok = np.zeros(len(labels), dtype=bool)
        for i in range(len(labels)):
            vals, counts = np.unique(patch[i], return_counts=True)
            ok[i] = int(vals[np.argmax(counts)]) in allowed
        n = int(ok.sum())
        print(f"[spawn_label_classes] {scene_id}: {n}/{len(ok)} frames "
              f"spawnable on classes {sorted(allowed)}")
        self._spawn_ok_cache[scene_id] = ok
        return ok

    def goal_position(self, scene_id: str) -> np.ndarray:
        if self.cfg.goal_xy_override is not None:
            gx, gy = self.cfg.goal_xy_override
            return np.array([gx, gy, 0.0], dtype=np.float32)
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
        if self.cfg.goal_dir_360:
            lo_d, hi_d = self.cfg.goal_dist_range or (5.0, 10.0)
            d = float(rng.uniform(lo_d, hi_d))
            cone = float(self.cfg.goal_cone_deg)
            if cone >= 360.0:
                th = float(rng.uniform(0.0, 2.0 * np.pi))
            else:
                cal = self._calib[scene_id]
                pos = np.asarray(cal.positions)[:, :2]
                i = int(np.argmin(np.linalg.norm(
                    pos - np.asarray(spawn_xy), axis=1)))
                fwd = pos[min(i + 1, len(pos) - 1)] - pos[max(i - 1, 0)]
                base = float(np.arctan2(fwd[1], fwd[0]))
                half = float(np.deg2rad(cone)) / 2.0
                th = base + float(rng.uniform(-half, half))
            return np.array([spawn_xy[0] + d * np.cos(th),
                             spawn_xy[1] + d * np.sin(th), 0.0],
                            dtype=np.float32)
        if self.cfg.goal_xy_override is not None or self.cfg.goal_frame_range is None:
            return self.goal_position(scene_id)
        cal = self._calib[scene_id]
        lo, hi = self.cfg.goal_frame_range
        hi = min(hi, len(cal.positions) - 1)
        if self.cfg.goal_dist_m is not None:
            # fixed-distance mode: frame whose straight-line distance from the
            # spawn best matches goal_dist_m (small jitter keeps episodes varied)
            pos = np.asarray(cal.positions[lo:hi + 1])
            d = np.linalg.norm(pos[:, :2] - np.asarray(spawn_xy), axis=1)
            frame = lo + int(np.argmin(np.abs(d - self.cfg.goal_dist_m)))
            frame = int(np.clip(frame + rng.integers(-2, 3), lo, hi))
            goal = cal.positions[frame].copy()
            goal[2] = 0.0
            return goal.astype(np.float32)
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
        # Label npz files are stored at the labeler's resolution (336x560);
        # when the backend runs at another (e.g. the 112-multiple speed rungs)
        # resize nearest-neighbor so labels stay per-pixel aligned with views.
        if labels.shape[-2:] != (cfg.H, cfg.W):
            lt = torch.as_tensor(np.ascontiguousarray(labels))[:, None].float()
            labels = torch.nn.functional.interpolate(
                lt, size=(cfg.H, cfg.W), mode="nearest"
            )[:, 0].to(torch.int64).numpy()
        n = min(len(images), len(labels))
        views = {
            "img": torch.stack([F.to_tensor(im)[None] for im in images[:n]], dim=1).to(device),
            "is_target": torch.zeros((1, n), dtype=torch.bool, device=device),
            "is_static": torch.zeros((1, n), dtype=torch.bool, device=device),
            "timestamp": torch.arange(0, n, dtype=torch.int64, device=device).unsqueeze(0),
            "labels": torch.as_tensor(labels[:n], dtype=torch.long, device=device).unsqueeze(0),
        }
        # PANO SIDE-VIEWS (2026-08-31, the true-360 track): if virtual pinhole
        # side views carved from a 360 pano exist beside the clip
        # (<stem>_pano_yawNNN.mp4, made by prepare_rosbag_clips --pano_topic)
        # AND have v14 labels (<stem>_pano_yawNNN.npz in the labels dir),
        # append them to the reconstruction diet at the SAME timestamps —
        # real geometry at the flanks, so off-heading rendering stops
        # hallucinating. Back view (yaw180) excluded by convention: the
        # operator-followers live there. Inert when the files don't exist.
        from pathlib import Path as _P
        _vp, _lp = _P(video_path), _P(labels_path)
        _pano_on = bool(getattr(cfg, "use_pano_views", False))
        # Stride: full 81+81+81 = 243 views OOMs WorldMirror's gs_head
        # (458356 maiden flight, 2026-08-31). Side views carry flank SUPPORT
        # not detail — every 3rd frame (81+27+27=135) keeps the coverage at a
        # memory cost the reconstructor survives.
        _stride = int(getattr(cfg, "pano_view_stride", 3))
        for _yaw in (90, 270) if _pano_on else ():
            side_mp4 = _vp.with_name(f"{_vp.stem}_pano_yaw{_yaw:03d}.mp4")
            side_lab = _lp.with_name(f"{_lp.stem}_pano_yaw{_yaw:03d}.npz")
            if not (side_mp4.exists() and side_lab.exists()):
                continue
            simgs = load_video(str(side_mp4), cfg.num_frames,
                               resolution=(cfg.W, cfg.H),
                               resize_mode="center_crop", static_scene=False)
            slab = np.load(side_lab)["labels"]
            if slab.shape[-2:] != (cfg.H, cfg.W):
                lt = torch.as_tensor(np.ascontiguousarray(slab))[:, None].float()
                slab = torch.nn.functional.interpolate(
                    lt, size=(cfg.H, cfg.W), mode="nearest"
                )[:, 0].to(torch.int64).numpy()
            m = min(len(simgs), len(slab), n)
            keep = list(range(0, m, _stride))
            ts = torch.tensor(keep, dtype=torch.int64, device=device)
            views["img"] = torch.cat(
                [views["img"], torch.stack(
                    [F.to_tensor(simgs[j])[None] for j in keep],
                    dim=1).to(device)], dim=1)
            # is_static=True: side views carry static scene content, and the
            # dynamic-gaussian builder asserts non-decreasing timestamps
            # across views — appending a second 0..78 sequence after main's
            # 0..80 violates it (458625). Static views skip that machinery.
            for k2, fill in (("is_target", False), ("is_static", True)):
                views[k2] = torch.cat(
                    [views[k2], torch.full((1, len(keep)), fill,
                                           dtype=torch.bool,
                                           device=device)], dim=1)
            views["timestamp"] = torch.cat(
                [views["timestamp"], ts.unsqueeze(0)], dim=1)
            views["labels"] = torch.cat(
                [views["labels"], torch.as_tensor(
                    slab[keep], dtype=torch.long,
                    device=device).unsqueeze(0)], dim=1)
            print(f"[pano] {_vp.stem}: appended side view yaw{_yaw:03d} "
                  f"({len(keep)}/{m} frames, stride {_stride})", flush=True)
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

    def _pose_nav_to_recon(self, pose_nav_robot: np.ndarray):
        """(pose_recon, t_idx) for a nav-frame robot pose — the calibration
        math shared by render() and the batched live backend."""
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
        return pose_recon, t_idx

    def render(self, pose_nav_robot: np.ndarray):
        import torch
        from diffsynth.utils.auxiliary import homo_matrix_inverse

        scene = self._cache.get(self._current_scene_id)
        if scene is None:
            raise RuntimeError("call load_scene() first")
        cal = self._calib[self._current_scene_id]
        pose_recon, t_idx = self._pose_nav_to_recon(pose_nav_robot)

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
