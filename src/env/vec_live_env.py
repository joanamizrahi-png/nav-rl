"""Batched live-diffusion training: N robots, one pipe, one generation per step.

Why: live training at batch 1 costs ~1.4 s/step — 48 h buys ~120k steps, the
budget pure-live runs kept dying at. The GPU is mostly idle at batch 1, so N
independent robots batched into ONE diffusion call amortize the cost to a
fraction per robot. SB3's own vectorization cannot do this: SubprocVecEnv
duplicates the 40 GB pipe per process, DummyVecEnv steps envs sequentially
(N separate batch-1 calls). This module centralizes the render:

    LiveVecEnv.step_wait():
        1. every SceneEnv advances its sim state CHEAPLY (defer_render=True)
        2. all robots' new poses -> BatchedLiveDiffusedBackend.render_batch()
           -> ONE pipe call with batch dim B
        3. frames + generated labels injected back into each env
        4. done envs get a correct terminal observation, then reset (their
           spawn views render in a second, usually tiny, batched call)

All robots must live in the SAME scene (one gaussian field on the GPU); the
scene rotates per training run, not per episode.
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .live_backend import LiveDiffusedBackend


class InjectedLabelBackend:
    """SemanticBackend shim for batched-live envs: labels arrive via
    SceneEnv.inject_render() instead of a backend-side render."""

    def __init__(self, env):
        self._env = env

    def segment(self, rgb: np.ndarray) -> np.ndarray:
        labels = getattr(self._env, "_injected_labels", None)
        if labels is None:
            raise RuntimeError("InjectedLabelBackend: no labels injected yet")
        return labels


class BatchedLiveDiffusedBackend(LiveDiffusedBackend):
    """LiveDiffusedBackend with per-robot pose histories and a batched render.

    render() (the single-robot path) still works — evals and smokes reuse it.
    render_batch() serves the vectorized env.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._hists: dict[int, list] = {}

    def _robot_hist(self, robot_id: int, pose_recon: np.ndarray) -> list:
        from .live_backend import cold_history
        hist = self._hists.get(robot_id, [])
        if hist:
            jump = np.linalg.norm(hist[-1][:3, 3] - pose_recon[:3, 3])
            if jump > 0.5:
                hist = []
        if not hist:
            scene = self._cache.get(self._current_scene_id)
            hist = cold_history(pose_recon, scene, self.live_frames)
        else:
            hist = (hist + [pose_recon])[-self.live_frames:]
        self._hists[robot_id] = hist
        return hist

    def render_batch(self, items: "list[tuple[int, np.ndarray]]"):
        """items: [(robot_id, pose_nav_robot 4x4), ...] -> per item
        (rgb HxWx3 uint8, K 3x3, w2c_nav 4x4, labels HxW int8)."""
        import torch
        from diffsynth.utils.auxiliary import homo_matrix_inverse
        from inference_semantic import (
            _decoded_video_to_uint8, _make_dual_decode,
            _sem_video_to_labels_and_colorized,
        )

        scene = self._cache.get(self._current_scene_id)
        if scene is None:
            raise RuntimeError("call load_scene() first")
        cal = self._calib[self._current_scene_id]
        pipe = self._pipe
        device = next(pipe.reconstructor.parameters()).device
        # 2026-08-30: the startup pre-reconstruction pass parks scene caches on
        # CPU; whatever path re-activates the resident scene can miss the
        # upload (cdist device crash, G12 456804). Guard here — the render is
        # the one place that MUST have the scene on the GPU.
        if torch.is_tensor(scene.get("K")) and scene["K"].device.type == "cpu":
            from .real_backend import _move_tree_to
            scene = _move_tree_to(scene, device)
            self._cache[self._current_scene_id] = scene
        B, k = len(items), self.live_frames
        t0 = time.perf_counter()

        hists = []
        for robot_id, pose_nav in items:
            pose_recon, _ = self._pose_nav_to_recon(pose_nav)
            hists.append(self._robot_hist(robot_id, pose_recon.astype(np.float32)))

        c2w = torch.from_numpy(np.stack([np.stack(h) for h in hists])).to(
            device=device, dtype=scene["cam2world"].dtype)        # [B,k,4,4]
        flat = c2w.reshape(B * k, 4, 4)
        w2c = homo_matrix_inverse(flat)
        K_rep = scene["K"][0:1].repeat(B * k, 1, 1)
        src_pos = scene["cam2world"][:, :3, 3]
        t_idx = torch.cdist(flat[:, :3, 3].float(), src_pos.float()).argmin(dim=1)
        target_ts = scene["timestamps"][t_idx]

        raster = pipe.reconstructor.gs_renderer.rasterizer
        rgb_t, depth_t, alpha_t = raster.forward(
            scene["gaussians"], render_viewmats=[w2c], render_Ks=[K_rep],
            render_timestamps=[target_ts], sh_degree=0,
            width=self.W, height=self.H)
        sem_t, _, _ = raster.forward(
            scene["gaussians"], render_viewmats=[w2c], render_Ks=[K_rep],
            render_timestamps=[target_ts], sh_degree=0,
            width=self.W, height=self.H, feature="labels")

        def bshape(x):
            return x.reshape(B, k, *x.shape[2:]) if x.shape[0] == 1 else x

        tgt_rgb, tgt_depth = bshape(rgb_t), bshape(depth_t)
        tgt_alpha = bshape(alpha_t)
        tgt_mask = (tgt_alpha > 1.0).float()
        tgt_sem = bshape(sem_t.argmax(dim=-1).to(torch.long))

        sink: dict = {}
        orig_decode = pipe.vae.decode
        pipe.vae.decode = _make_dual_decode(orig_decode, sink)
        try:
            with torch.no_grad():
                generated = pipe(
                    prompt=[self.cfg.prompt] * B,
                    negative_prompt=[self.cfg.negative_prompt] * B,
                    seed=0, rand_device=device,
                    height=self.H, width=self.W, num_frames=k,
                    cfg_scale=self.cfg.cfg_scale,
                    num_inference_steps=getattr(
                        self, "num_inference_steps",
                        4 if self.cfg.use_lora else 50),
                    tiled=False,
                    source_views=scene["views"],
                    target_rgb=tgt_rgb, target_depth=tgt_depth,
                    target_mask=tgt_mask,
                    target_poses=c2w,
                    target_intrs=K_rep.reshape(B, k, 3, 3),
                    target_semantic=tgt_sem,
                )
        finally:
            pipe.vae.decode = orig_decode

        # ---- unpack last frames per robot ----
        # The pipe's own return is USELESS for B>1: vae_output_to_video does
        # reduce("B C T H W -> T H W C", "mean") — it AVERAGES the robots into
        # one ghost video. The dual-decode sink captures the raw VAE decode
        # BEFORE that reduce, batch dim intact — per-robot frames come from it.
        def last_rgb_frames() -> list:
            rv = sink.get("rgb_video")
            if rv is not None and getattr(rv, "ndim", 0) == 5 and rv.shape[0] == B:
                return [_decoded_video_to_uint8(rv[i:i + 1])[-1] for i in range(B)]
            if B == 1:
                return [np.array(generated[-1])]
            raise RuntimeError(
                f"render_batch: no batched rgb_video in dual-decode sink for "
                f"B={B} (keys={list(sink)}); the pipe return is batch-averaged "
                "and cannot be unpacked per robot")

        rgbs = last_rgb_frames()

        head = getattr(pipe, "semantic_class_head", None)
        labels_per_robot: list = []
        sem_video = sink.get("sem_video")
        if sem_video is not None:
            sv = sem_video
            if B == 1:
                lab, _ = _sem_video_to_labels_and_colorized(sv, head=head)
                labels_per_robot = [lab[-1].astype(np.int8)]
            elif hasattr(sv, "shape") and sv.shape[0] == B:
                for i in range(B):
                    lab, _ = _sem_video_to_labels_and_colorized(sv[i], head=head)
                    labels_per_robot.append(lab[-1].astype(np.int8))
        if not labels_per_robot:                       # degrade to raster hint
            labels_per_robot = [
                tgt_sem[i, -1].detach().cpu().numpy().astype(np.int8)
                for i in range(B)]

        alpha_last = (tgt_alpha[:, -1].detach().float().cpu().numpy() > 0.5)
        if alpha_last.ndim == 4:
            alpha_last = alpha_last.squeeze(-1)

        K_np = scene["K"][0].detach().cpu().float().numpy()
        nav2recon = cal.nav_world_to_recon_homogeneous()
        out = []
        for i in range(B):
            lab = labels_per_robot[i]
            if self._alpha_gate:
                lab = lab.copy()
                lab[~alpha_last[i]] = 0
            w2c_recon = homo_matrix_inverse(
                c2w[i, -1:].detach()).squeeze(0).cpu().float().numpy()
            w2c_nav = (w2c_recon.astype(np.float64) @ nav2recon).astype(np.float32)
            out.append((rgbs[i], K_np, w2c_nav, lab))
        self.last_timings = {"total": time.perf_counter() - t0, "batch": B}
        return out


try:
    from stable_baselines3.common.vec_env import VecEnv
except Exception:                                     # import-time safety only
    VecEnv = object


class LiveVecEnv(VecEnv):
    """N SceneEnvs (defer_render) sharing one BatchedLiveDiffusedBackend.

    Multi-scene (2026-08-29, Joana's call: "the policy must learn what
    obstacles look like — one world can't teach that"): pass `scenes` and
    `rotate_every` and the env rotates the resident world on a step budget —
    all episodes force-truncate cleanly, the next scene's Gaussians load,
    pose histories clear (cold-start walk-ins rebuild), robots respawn there.
    """

    def __init__(self, envs: list, backend: BatchedLiveDiffusedBackend,
                 scenes: "list[str] | None" = None, rotate_every: int = 0):
        self.envs = envs
        self.backend = backend
        self.scenes = list(scenes) if scenes else []
        self.rotate_every = int(rotate_every)
        self._scene_i = 0
        self._steps_since_rot = 0
        self._actions: Optional[np.ndarray] = None
        self.render_mode = None
        super().__init__(len(envs), envs[0].observation_space, envs[0].action_space)

    # ---------- helpers ----------

    def _render_and_inject(self, idxs: list) -> None:
        if not idxs:
            return
        items = [(i, self.envs[i]._robot_pose_world) for i in idxs]
        results = self.backend.render_batch(items)
        for i, (rgb, K, w2c, lab) in zip(idxs, results):
            self.envs[i].inject_render(rgb, K, w2c, labels=lab)

    def _stack_obs(self, obs_list: list) -> dict:
        return {key: np.stack([o[key] for o in obs_list])
                for key in obs_list[0]}

    # ---------- VecEnv API ----------

    def reset(self):
        for e in self.envs:
            e.reset()
        self._render_and_inject(list(range(self.num_envs)))
        return self._stack_obs([e._obs() for e in self.envs])

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = actions

    def step_wait(self):
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=bool)
        infos: list = [None] * self.num_envs
        for i, e in enumerate(self.envs):
            _, r, term, trunc, info = e.step(self._actions[i])
            rewards[i] = r
            dones[i] = bool(term or trunc)
            info = dict(info)
            info["TimeLimit.truncated"] = bool(trunc and not term)
            infos[i] = info
        self._steps_since_rot += self.num_envs
        rotate = (self.rotate_every > 0 and len(self.scenes) > 1
                  and self._steps_since_rot >= self.rotate_every)
        if rotate:
            for i in range(self.num_envs):     # force-truncate survivors
                if not dones[i]:
                    dones[i] = True
                    infos[i]["TimeLimit.truncated"] = True
        # one batched render at the post-step poses (correct terminal obs too,
        # still in the OLD scene when rotating)
        self._render_and_inject(list(range(self.num_envs)))
        reset_idx = [i for i in range(self.num_envs) if dones[i]]
        for i in reset_idx:
            infos[i]["terminal_observation"] = self.envs[i]._obs()
        if rotate:
            self._scene_i = (self._scene_i + 1) % len(self.scenes)
            new_scene = self.scenes[self._scene_i]
            self.backend._hists.clear()        # cross-scene coords may alias;
            for e in self.envs:                # walk-ins rebuild on reset
                e.scene_ids = [new_scene]
            self._steps_since_rot = 0
        for i in reset_idx:
            self.envs[i].reset()               # load_scene(new) happens here
        self._render_and_inject(reset_idx)     # spawn views, usually tiny
        obs = self._stack_obs([e._obs() for e in self.envs])
        return obs, rewards, dones, infos

    def close(self) -> None:
        for e in self.envs:
            e.close()

    def get_attr(self, attr_name, indices=None):
        idx = range(self.num_envs) if indices is None else np.atleast_1d(indices)
        return [getattr(self.envs[i], attr_name) for i in idx]

    def set_attr(self, attr_name, value, indices=None):
        idx = range(self.num_envs) if indices is None else np.atleast_1d(indices)
        for i in idx:
            setattr(self.envs[i], attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        idx = range(self.num_envs) if indices is None else np.atleast_1d(indices)
        return [getattr(self.envs[i], method_name)(*args, **kwargs) for i in idx]

    def env_is_wrapped(self, wrapper_class, indices=None):
        idx = range(self.num_envs) if indices is None else np.atleast_1d(indices)
        return [False for _ in idx]

    def seed(self, seed=None):
        return [None for _ in range(self.num_envs)]
