"""RealWorldBackend — NeoVerse-based WorldBackend implementation for Milestone B.

Wraps NeoVerse's pipeline as a Gym-compatible world model backend. Same interface
as MockWorldBackend so SceneEnv doesn't care which backend is plugged in.

Coordinate conventions
----------------------
SceneEnv's world frame (from scene_env.py):
    x = robot forward
    y = robot right
    z = up (gravity)

Reconstructor's world frame (from pose diagnostics on 12 clips):
    x = right (ground plane)
    y = up (perpendicular to ground)
    z = forward (camera looks along +z at frame 0)

Robot's local frame (what SceneEnv._advance_pose assumes):
    local +x = forward direction the robot is facing

Camera local frame (NeoVerse convention, standard CV):
    local +x = right
    local +y = down
    local +z = forward

So `render(pose_scene)` has to convert:
    pose_scene (robot-to-world in scene frame)
    -> pose_recon (camera-to-world in reconstructor frame)

...via two fixed 4x4 rotations built into the constants below.

Scale
-----
Reconstructor units are NORMALIZED per clip, not meters. See REWARD.md and
OPTIONS.md. This backend accepts scene-frame poses in the same "recon units"
convention and does NOT apply a scale factor. Reward function and step sizes
in SceneEnvConfig should be tuned assuming the world is ~1 unit ≈ few real
meters. Deploy-time scale calibration is a separate problem.

Coverage
--------
The Gaussians only exist where the source video's camera saw them. If the RL
policy chooses a pose FAR from the source trajectory, the rasterizer will
return mostly-holey output that the diffusion tries to inpaint. Quality
degrades gracefully as the policy strays from the reconstructed volume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Coordinate conversion constants
# ---------------------------------------------------------------------------

# Scene world -> Reconstructor world (both right-handed, y-up in recon, z-up in scene).
# scene_y (right) -> recon_x (right)
# scene_z (up)    -> recon_y (up)
# scene_x (fwd)   -> recon_z (fwd)
R_SCENE_TO_RECON = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
], dtype=np.float32)

# Robot local -> Camera local (both applied on the LEFT of the local axes).
# robot_x (fwd)  -> camera_z (fwd)
# robot_y (right)-> camera_x (right)
# robot_z (up)   -> -camera_y (up = -down)
R_ROBOT_TO_CAM = np.array([
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
], dtype=np.float32)


def _pose_scene_to_recon(pose_scene: np.ndarray, camera_height_scene: float = 0.0) -> np.ndarray:
    """Convert a 4x4 robot-to-scene-world pose into a 4x4 camera-to-recon-world pose.

    Steps:
      1. Optionally lift the camera above the robot's position by `camera_height_scene`
         (in scene world coords, along the scene z-up axis).
      2. Apply the robot->camera local rotation on the RIGHT (rotates the local frame).
      3. Apply the scene->recon world rotation on the LEFT (rotates the world frame).
    """
    T_lift = np.eye(4, dtype=np.float32)
    T_lift[2, 3] = camera_height_scene    # scene z is up

    # Build the two 4x4 rotations
    T_rc = np.eye(4, dtype=np.float32)
    T_rc[:3, :3] = R_ROBOT_TO_CAM
    T_sr = np.eye(4, dtype=np.float32)
    T_sr[:3, :3] = R_SCENE_TO_RECON

    # pose_scene assumed shape (4,4) float32
    # camera_to_world_scene = T_lift @ pose_scene @ T_rc
    # camera_to_world_recon = T_sr @ camera_to_world_scene
    return T_sr @ (T_lift @ pose_scene @ T_rc)


# ---------------------------------------------------------------------------
# Backend config
# ---------------------------------------------------------------------------

@dataclass
class RealWorldBackendConfig:
    # Where to find the source video for each scene_id (scene_id -> mp4 path).
    scene_video_paths: dict = field(default_factory=dict)

    # Where to put the robot at reset — defaults to identity (origin in scene frame).
    scene_start_poses: dict = field(default_factory=dict)

    # Where the goal is in scene frame per scene_id.
    scene_goals: dict = field(default_factory=dict)

    # Camera mounted this high above the robot's body-origin in SceneEnv's world
    # frame (along scene +z, up).
    #
    # DEFAULT = 0.0: identity pose_scene puts the camera exactly where the
    # SOURCE VIDEO's frame-0 camera was. Simplest, matches what the reconstructor
    # actually gives us. Any real-robot calibration (Go2 camera is ~0.4m above
    # its feet in real world) happens at deploy time by mapping physical actions
    # to sim actions -- not by trying to compensate here.
    #
    # A non-zero value would try to invent a "body position" below the camera,
    # which never existed in the source video and can produce off-manifold renders.
    camera_height_scene: float = 0.0

    # Rendering config
    H: int = 336
    W: int = 560
    num_frames: int = 81       # frames sampled from source video for reconstruction

    # Diffusion config
    use_lora: bool = True      # 4-step distilled LoRA (fast); False = 50 steps
    cfg_scale: float = 1.0
    prompt: str = ("A smooth video with complete scene content. Inpaint any missing "
                   "regions or margins naturally to match the surrounding scene.")
    negative_prompt: str = ""

    # Rendering mode:
    #   "rasterizer_plus_diffusion" (DEFAULT) — run 4-step diffusion on top of the
    #       raw Gaussian rasterization. Clean output, ~5x slower per step. Video
    #       prior may hallucinate small dynamics (parked cars fake-driving on
    #       driving.mp4-style scenes). Less problematic on natural / static
    #       scenes like RUGD trails.
    #   "rasterizer_only" — raw rasterizer, no diffusion. Deterministic and cheap
    #       BUT holey outside the reconstructed volume — likely too bad for RL
    #       observations (holes as big as the visible area). Retained for tests.
    render_mode: str = "rasterizer_plus_diffusion"

    # Model paths
    model_path: str = "/scratch/m000204-pm06b/joana/NeoVerse/models"
    reconstructor_path: str = "/scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt"


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------

class RealWorldBackend:
    """WorldBackend implementation that renders through NeoVerse's pipeline.

    Attributes match the WorldBackend Protocol from scene_env.py:
        H, W: render resolution
        load_scene(scene_id): reconstruct + cache Gaussians for that scene
        render(pose_scene): return (rgb, K, w2c) at the requested robot pose
        start_pose(scene_id): initial robot pose (identity by default)
        goal_position(scene_id): goal in scene world coords
    """

    def __init__(self, cfg: RealWorldBackendConfig = RealWorldBackendConfig()):
        self.cfg = cfg
        self.H = cfg.H
        self.W = cfg.W
        self._pipe = None                       # lazy-loaded
        self._current_scene_id: Optional[str] = None
        self._cache = {}                         # scene_id -> per-scene state

    # ------------ WorldBackend API ------------

    def load_scene(self, scene_id: str) -> None:
        """Reconstruct the source video (once per scene) and cache Gaussians + K."""
        if scene_id in self._cache:
            self._current_scene_id = scene_id
            return
        self._ensure_pipe_loaded()

        video_path = self.cfg.scene_video_paths.get(scene_id)
        if not video_path:
            raise ValueError(f"no video path configured for scene_id={scene_id}")

        cached = self._reconstruct_scene(video_path)
        self._cache[scene_id] = cached
        self._current_scene_id = scene_id

    def start_pose(self, scene_id: str) -> np.ndarray:
        return self.cfg.scene_start_poses.get(
            scene_id, np.eye(4, dtype=np.float32)
        ).copy()

    def goal_position(self, scene_id: str) -> np.ndarray:
        # If not configured, default goal is "3 units forward in scene x" (a few meters).
        return self.cfg.scene_goals.get(
            scene_id, np.array([3.0, 0.0, 0.0], dtype=np.float32)
        ).copy()

    def render(self, pose_scene: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (rgb, K, w2c) after rasterize + diffuse at the requested pose.

        pose_scene: (4, 4) robot-to-world matrix in SceneEnv's world frame.
        """
        scene = self._cache.get(self._current_scene_id)
        if scene is None:
            raise RuntimeError("call load_scene() first")

        # Convert robot-to-scene-world -> camera-to-recon-world.
        pose_recon = _pose_scene_to_recon(
            pose_scene.astype(np.float32),
            camera_height_scene=self.cfg.camera_height_scene,
        )
        rgb, K, w2c = self._rasterize_and_diffuse(scene, pose_recon)
        return rgb, K, w2c

    # ------------ heavy lifting (deferred to NeoVerse) ------------

    def _ensure_pipe_loaded(self):
        if self._pipe is not None:
            return
        # Deferred import so this file can be imported without NeoVerse on path.
        import torch
        from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
        self._torch = torch

        lora_path = None
        if self.cfg.use_lora:
            lora_path = str(Path(self.cfg.model_path) /
                            "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors")

        print(f"[RealWorldBackend] loading NeoVerse pipeline from {self.cfg.model_path} ...", flush=True)
        self._pipe = WanVideoNeoVersePipeline.from_pretrained(
            local_model_path=self.cfg.model_path,
            reconstructor_path=self.cfg.reconstructor_path,
            lora_path=lora_path, lora_alpha=1.0,
            device="cuda", torch_dtype=torch.bfloat16,
        )

    def _reconstruct_scene(self, video_path: str) -> dict:
        """Run the reconstructor once and cache Gaussians + K + first-frame pose."""
        import torch
        from torchvision.transforms import functional as F
        from diffsynth.utils.auxiliary import load_video, homo_matrix_inverse

        cfg = self.cfg
        pipe = self._pipe

        print(f"[RealWorldBackend] loading source video: {video_path}", flush=True)
        images = load_video(
            video_path, cfg.num_frames,
            resolution=(cfg.W, cfg.H), resize_mode="center_crop", static_scene=False,
        )
        device = pipe.device
        views = {
            "img": torch.stack([F.to_tensor(im)[None] for im in images], dim=1).to(device),
            "is_target": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
            "is_static": torch.zeros((1, len(images)), dtype=torch.bool, device=device),
            "timestamp": torch.arange(0, len(images), dtype=torch.int64, device=device).unsqueeze(0),
        }
        print(f"[RealWorldBackend] running reconstructor ...", flush=True)
        with torch.amp.autocast("cuda", dtype=pipe.torch_dtype):
            predictions = pipe.reconstructor(views, is_inference=True, use_motion=False)

        return {
            "gaussians": predictions["splats"],
            "K": predictions["rendered_intrinsics"][0],                # (T, 3, 3)
            "cam2world": predictions["rendered_extrinsics"][0],        # (T, 4, 4)
            "timestamps": predictions["rendered_timestamps"][0],       # (T,)
            "views": views,
        }

    def _rasterize_and_diffuse(self, scene: dict, pose_recon: np.ndarray):
        """Rasterize the cached Gaussians at pose_recon, then run diffusion.

        pose_recon: (4, 4) camera-to-recon-world matrix (numpy).
        """
        import torch
        from diffsynth.utils.auxiliary import homo_matrix_inverse

        pipe = self._pipe
        device = pipe.device
        N = len(scene["timestamps"])

        # Broadcast our single requested pose to N frames (matches the reconstructor's timestamp layout).
        fixed_c2w = torch.from_numpy(pose_recon).to(device=device, dtype=scene["cam2world"].dtype).unsqueeze(0).repeat(N, 1, 1)
        fixed_w2c = homo_matrix_inverse(fixed_c2w)
        K_rep = scene["K"][0:1].repeat(N, 1, 1)

        target_rgb, target_depth, target_alpha = pipe.reconstructor.gs_renderer.rasterizer.forward(
            scene["gaussians"],
            render_viewmats=[fixed_w2c], render_Ks=[K_rep],
            render_timestamps=[scene["timestamps"]],
            sh_degree=0, width=self.W, height=self.H,
        )

        if self.cfg.render_mode == "rasterizer_only":
            # Fast path: raw rasterizer output, no diffusion, no video-prior
            # hallucinations. Take the first rendered frame as the observation.
            rgb_frame = (target_rgb[0, 0].detach().clamp(0, 1).float().cpu().numpy() * 255).astype(np.uint8)
            K_np = scene["K"][0].detach().cpu().float().numpy()
            w2c_np = fixed_w2c[0].detach().cpu().float().numpy()
            return rgb_frame, K_np, w2c_np

        # Slow path: full 4-step diffusion. Runs on all N frames at once and takes
        # the FIRST rendered frame as the observation. Cleaner output but has
        # video-prior hallucinations (see freeze_camera_diffuse.py test on driving.mp4).
        target_mask = (target_alpha > 1.0).float()
        wrapped_data = {
            "source_views": scene["views"],
            "target_rgb": target_rgb, "target_depth": target_depth, "target_mask": target_mask,
            "target_poses": fixed_c2w.unsqueeze(0), "target_intrs": K_rep.unsqueeze(0),
        }
        with torch.no_grad():
            generated = pipe(
                prompt=self.cfg.prompt, negative_prompt=self.cfg.negative_prompt,
                seed=0, rand_device=device,
                height=self.H, width=self.W, num_frames=N,
                cfg_scale=self.cfg.cfg_scale, num_inference_steps=4 if self.cfg.use_lora else 50,
                tiled=False, **wrapped_data,
            )
        rgb_frame = np.array(generated[0])
        K_np = scene["K"][0].detach().cpu().float().numpy()
        w2c_np = fixed_w2c[0].detach().cpu().float().numpy()
        return rgb_frame, K_np, w2c_np
