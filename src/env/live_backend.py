"""Live semantic-diffusion backend: per-action generation, no cache.

The policy queries the generative model along its own motion instead of a
pre-rendered grid (design decision 2026-08-21). Every env.step renders ONE
minimal diffusion call — a 5-frame clip whose
frames are the robot's 4 previous poses + the current pose (the video model's
4k+1 minimum, filled with true history rather than padding), last frame = the
observation. The SAME call's semantic half feeds the reward, so the ungated
recipe trains against the diffusion model's labels at the exact pose, with no
grid discretization.

Differences from the old live path (real_backend._rasterize_and_diffuse):
  - loads the SEMANTIC pipeline (v10 checkpoint: expansion v2 + LoRA + reader
    head) — the old path loaded the vanilla model and had no diffused labels;
  - renders k requested poses, not 1 pose broadcast to 81 (30 s -> target ~3 s);
  - scene stays resident on GPU (no CPU<->GPU shuttle per call);
  - nothing is written to disk.

Scene time advances with the robot (per-pose nearest source frame), so a
dynamic reconstruction (SCAND) renders moving pedestrians — something caches
structurally cannot do (they freeze time per sweep).

Speed instrumentation: self.last_timings holds per-phase seconds for the most
recent render; scripts/live_benchmark.py builds the decision table from it.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .real_calibrated import CalibratedRealWorldBackend


class LiveDiffusedBackend(CalibratedRealWorldBackend):
    """Per-action live generation with the semantic diffusion pipeline."""

    def __init__(self, cfg, checkpoint: str, live_frames: int = 5,
                 alpha_gate: bool = False, num_classes: int = 14,
                 lora_rank: int = 8,
                 lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"):
        # Any mode other than "rasterizer_plus_diffusion" keeps the scene
        # resident on GPU in _reconstruct_scene (both parent and calibrated
        # override) — exactly what we want here.
        cfg.render_mode = "live_diffusion"
        super().__init__(cfg)
        self._sem_checkpoint = checkpoint
        self.live_frames = int(live_frames)
        self._alpha_gate = bool(alpha_gate)
        self._num_classes = int(num_classes)
        self._lora_rank = int(lora_rank)
        self._lora_targets = lora_target_modules
        self._pose_hist: list = []            # recon-frame c2w, newest last
        self._pending_labels: Optional[np.ndarray] = None
        self.last_timings: dict = {}

    # ---------- semantic pipeline (loaded once, resident) ----------

    def _ensure_semantic_pipe(self):
        if getattr(self, "_pipe", None) is not None:
            return
        import gc
        import torch
        from pathlib import Path
        from diffsynth.pipelines.wan_video_neoverse import WanVideoNeoVersePipeline
        from diffsynth.utils.semantics import (
            expand_dit_for_semantics_v2, expand_control_branch_for_semantics_v2,
            set_active_palette,
        )
        from inference_semantic import (
            _inject_lora_for_finetune, _load_finetune_checkpoint,
        )
        self._torch = torch

        # Same discipline as the parent's _ensure_pipe_loaded: drop any
        # standalone reconstructor before the pipe brings its own copy.
        if self._reconstructor is not None:
            del self._reconstructor
            self._reconstructor = None
            gc.collect()
            torch.cuda.empty_cache()

        lora_path = None
        if self.cfg.use_lora:
            lora_path = str(Path(self.cfg.model_path) /
                            "NeoVerse/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors")
        print(f"[LiveDiffusedBackend] loading SEMANTIC pipeline "
              f"(ckpt={self._sem_checkpoint}) ...", flush=True)
        pipe = WanVideoNeoVersePipeline.from_pretrained(
            local_model_path=self.cfg.model_path,
            reconstructor_path=self.cfg.reconstructor_path,
            lora_path=lora_path, lora_alpha=1.0,
            device="cuda", torch_dtype=torch.bfloat16,
        )
        pipe.semantic_channels = 16
        pipe.semantic_x0_prediction = True
        set_active_palette(self._num_classes)
        expand_dit_for_semantics_v2(pipe.dit, extra=16)
        if pipe.control_branch is not None:
            expand_control_branch_for_semantics_v2(pipe.control_branch, extra=16)
        _inject_lora_for_finetune(pipe, rank=self._lora_rank,
                                  target_modules=self._lora_targets)
        _load_finetune_checkpoint(pipe, self._sem_checkpoint)
        self._pipe = pipe
        self._reconstructor = pipe.reconstructor
        free, _ = torch.cuda.mem_get_info()
        print(f"[LiveDiffusedBackend] semantic pipe ready. "
              f"VRAM free {free/1e9:.1f} GB", flush=True)

    def load_scene(self, scene_id: str) -> None:
        # Pipe (and therefore the reconstructor used for scene building) must
        # exist before reconstruction so there is exactly one copy on GPU.
        self._ensure_semantic_pipe()
        super().load_scene(scene_id)
        self._pose_hist = []

    # ---------- per-step live render ----------

    def _rasterize_and_diffuse(self, scene: dict, pose_recon: np.ndarray):
        torch = self._torch
        from diffsynth.utils.auxiliary import homo_matrix_inverse
        from inference_semantic import (
            _make_dual_decode, _sem_video_to_labels_and_colorized,
        )
        pipe = self._pipe
        device = next(pipe.reconstructor.parameters()).device
        t = {}
        t0 = time.perf_counter()

        # Rolling pose history. A jump (reset/teleport) restarts the clip so
        # the conditioning never spans a discontinuity.
        pose_recon = pose_recon.astype(np.float32)
        if self._pose_hist:
            jump = np.linalg.norm(self._pose_hist[-1][:3, 3] - pose_recon[:3, 3])
            if jump > 0.5:
                self._pose_hist = []
        if not self._pose_hist:
            self._pose_hist = [pose_recon] * self.live_frames
        else:
            self._pose_hist.append(pose_recon)
            self._pose_hist = self._pose_hist[-self.live_frames:]
        k = len(self._pose_hist)

        c2w = torch.from_numpy(np.stack(self._pose_hist)).to(
            device=device, dtype=scene["cam2world"].dtype)          # [k,4,4]
        w2c = homo_matrix_inverse(c2w)
        K_rep = scene["K"][0:1].repeat(k, 1, 1)

        # Scene time per pose: nearest source-camera position — the clock
        # advances as the robot moves (dynamic scenes render motion).
        src_pos = scene["cam2world"][:, :3, 3]                      # [N,3]
        d = torch.cdist(c2w[:, :3, 3].float(), src_pos.float())
        t_idx = d.argmin(dim=1)
        target_ts = scene["timestamps"][t_idx]

        raster = pipe.reconstructor.gs_renderer.rasterizer
        target_rgb, target_depth, target_alpha = raster.forward(
            scene["gaussians"], render_viewmats=[w2c], render_Ks=[K_rep],
            render_timestamps=[target_ts], sh_degree=0,
            width=self.W, height=self.H)
        sem_probs, _, _ = raster.forward(
            scene["gaussians"], render_viewmats=[w2c], render_Ks=[K_rep],
            render_timestamps=[target_ts], sh_degree=0,
            width=self.W, height=self.H, feature="labels")
        target_semantic = sem_probs.argmax(dim=-1).to(torch.long)
        target_mask = (target_alpha > 1.0).float()   # conditioning threshold (cache-gen parity)
        t["raster"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        sink: dict = {}
        orig_decode = pipe.vae.decode
        pipe.vae.decode = _make_dual_decode(orig_decode, sink)
        try:
            with torch.no_grad():
                generated = pipe(
                    prompt=self.cfg.prompt,
                    negative_prompt=self.cfg.negative_prompt,
                    seed=0, rand_device=device,
                    height=self.H, width=self.W, num_frames=k,
                    cfg_scale=self.cfg.cfg_scale,
                    num_inference_steps=4 if self.cfg.use_lora else 50,
                    tiled=False,
                    source_views=scene["views"],
                    target_rgb=target_rgb, target_depth=target_depth,
                    target_mask=target_mask,
                    target_poses=c2w.unsqueeze(0),
                    target_intrs=K_rep.unsqueeze(0),
                    target_semantic=target_semantic,
                )
        finally:
            pipe.vae.decode = orig_decode
        t["diffusion"] = time.perf_counter() - t1

        t2 = time.perf_counter()
        head = getattr(pipe, "semantic_class_head", None)
        if "sem_video" in sink:
            labels, _ = _sem_video_to_labels_and_colorized(sink["sem_video"], head=head)
            lab_last = labels[-1].astype(np.int8)
        else:  # checkpoint without semantic output — degrade to raster hint
            lab_last = target_semantic[0, -1].detach().cpu().numpy().astype(np.int8)
        alpha_last = (target_alpha[0, -1].detach().float().cpu().numpy() > 0.5).squeeze(-1)
        if self._alpha_gate:
            gated = lab_last.copy()
            gated[~alpha_last] = 0
            self._pending_labels = gated
        else:
            self._pending_labels = lab_last
        self._last_semantic_raw = lab_last
        self._live_alpha = alpha_last
        t["decode"] = time.perf_counter() - t2
        t["total"] = time.perf_counter() - t0
        self.last_timings = t

        rgb_frame = np.array(generated[-1])
        K_np = scene["K"][0].detach().cpu().float().numpy()
        w2c_np = w2c[-1].detach().cpu().float().numpy()
        return rgb_frame, K_np, w2c_np

    def _rasterize_labels(self, scene, pose_recon, t_idx: int = 0):
        # Calibrated.render() stores this as _last_semantic_image right after
        # the RGB dispatch; serve the labels DECODED FROM THE SAME DIFFUSION
        # CALL instead of a raster pass, so the reward reads what the policy saw.
        if self._pending_labels is not None:
            return self._pending_labels
        return super()._rasterize_labels(scene, pose_recon, t_idx)
