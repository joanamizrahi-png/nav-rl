"""K live vec-envs, one per GPU, presented to PPO as ONE VecEnv.

Why: live training is bounded by the diffusion call, and one GPU holds one
pipe and one resident scene. Environment steps are independent, so K GPUs
running K batched live envs give ~K times the steps per hour, and PPO does
not care where the rollouts came from. SB3's SubprocVecEnv cannot do this:
it wraps single gym.Envs, not VecEnvs, so it would lose the batched render
that makes one GPU worthwhile. This class is the same worker/pipe pattern,
one process per GPU, each process owning a whole LiveVecEnv.

    env = MultiGPUVecEnv(factory, n_workers=4)
        factory(worker_idx) -> a VecEnv (e.g. VecMonitor(LiveVecEnv(...)))
        worker i sees ONLY the i-th GPU of the job's CUDA_VISIBLE_DEVICES.

Global env index g maps to (worker g // B, local g % B) when every worker
has B envs; uneven workers are handled by the offsets table. Observations
(dict of arrays), rewards, dones and infos are concatenated in worker order,
so PPO sees num_envs = sum of the workers' num_envs.

The GPU pin happens in the child BEFORE the factory runs and before any CUDA
call: CUDA_VISIBLE_DEVICES is rewritten to the single id the worker owns,
taken from the job's list (Slurm hands the job e.g. "2,5", and "1" would be a
card the job does not have). torch reads the variable lazily at CUDA init,
so importing torch earlier is harmless; touching cuda earlier is not.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from typing import Any, Callable, Optional

import numpy as np
from stable_baselines3.common.vec_env import CloudpickleWrapper, VecEnv


def pin_gpu(worker_idx: int) -> str:
    """Restrict this process to the worker_idx-th GPU of the job's visible
    set. Returns the physical id chosen (for the banner)."""
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = [v.strip() for v in vis.split(",") if v.strip()]
    if ids:
        if worker_idx >= len(ids):
            raise RuntimeError(
                f"worker {worker_idx} asked for GPU #{worker_idx} but the job "
                f"only has {len(ids)} visible (CUDA_VISIBLE_DEVICES={vis!r}). "
                f"Submit with --gres=gpu:<n_workers>.")
        chosen = ids[worker_idx]
    else:
        chosen = str(worker_idx)
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen
    return chosen


def _worker(remote, parent_remote, factory_wrapper: CloudpickleWrapper,
            worker_idx: int) -> None:
    parent_remote.close()
    try:
        gpu = pin_gpu(worker_idx)
        os.environ["LIVE_WORKER_IDX"] = str(worker_idx)
        env: VecEnv = factory_wrapper.var(worker_idx)
        remote.send(("ready", dict(num_envs=env.num_envs,
                                   observation_space=env.observation_space,
                                   action_space=env.action_space,
                                   gpu=gpu)))
    except Exception:
        remote.send(("error", traceback.format_exc()))
        return
    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break
        try:
            if cmd == "step":
                env.step_async(data)
                remote.send(env.step_wait())
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "get_attr":
                name, idx = data
                remote.send(env.get_attr(name, idx))
            elif cmd == "set_attr":
                name, value, idx = data
                remote.send(env.set_attr(name, value, idx))
            elif cmd == "env_method":
                name, idx, a, kw = data
                remote.send(env.env_method(name, *a, indices=idx, **kw))
            elif cmd == "env_is_wrapped":
                wrapper_class, idx = data
                remote.send(env.env_is_wrapped(wrapper_class, idx))
            elif cmd == "seed":
                remote.send(env.seed(data))
            elif cmd == "close":
                env.close()
                remote.send(("closed", None))
                break
            else:
                raise NotImplementedError(f"unknown command {cmd!r}")
        except Exception:
            remote.send(("error", traceback.format_exc()))
    remote.close()


class MultiGPUVecEnv(VecEnv):
    def __init__(self, factory: Callable[[int], VecEnv], n_workers: int,
                 start_method: str = "spawn"):
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1")
        ctx = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(n_workers)])
        self.processes = []
        for i, (work_remote, remote) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(target=_worker,
                            args=(work_remote, remote, CloudpickleWrapper(factory), i),
                            daemon=True, name=f"live-gpu-worker-{i}")
            p.start()
            self.processes.append(p)
            work_remote.close()

        self.worker_num_envs: list[int] = []
        self.worker_gpu: list[str] = []
        obs_space = act_space = None
        for i, remote in enumerate(self.remotes):
            tag, payload = remote.recv()
            if tag != "ready":
                self.close()
                raise RuntimeError(f"worker {i} failed to build its env:\n{payload}")
            self.worker_num_envs.append(int(payload["num_envs"]))
            self.worker_gpu.append(str(payload["gpu"]))
            if obs_space is None:
                obs_space, act_space = payload["observation_space"], payload["action_space"]
        self.offsets = np.concatenate([[0], np.cumsum(self.worker_num_envs)]).astype(int)
        self._waiting = False
        self.render_mode = None
        super().__init__(int(self.offsets[-1]), obs_space, act_space)
        print(f"[MultiGPUVecEnv] {n_workers} workers, envs per worker "
              f"{self.worker_num_envs}, total {self.num_envs}, GPUs {self.worker_gpu}",
              flush=True)

    # ---------- plumbing ----------

    def _recv(self, i: int):
        r = self.remotes[i].recv()
        if isinstance(r, tuple) and len(r) == 2 and r[0] == "error":
            raise RuntimeError(f"worker {i} raised:\n{r[1]}")
        return r

    def _split(self, indices) -> list[Optional[list]]:
        """Global indices -> per-worker local index lists (None = all)."""
        if indices is None:
            return [None] * len(self.remotes)
        idx = np.atleast_1d(np.asarray(indices, dtype=int))
        out: list[Optional[list]] = []
        for w in range(len(self.remotes)):
            lo, hi = self.offsets[w], self.offsets[w + 1]
            loc = [int(g - lo) for g in idx if lo <= g < hi]
            out.append(loc)
        return out

    def _fan(self, cmd: str, build_data: Callable[[int, Optional[list]], Any],
             indices=None) -> list:
        per = self._split(indices)
        active = [w for w, loc in enumerate(per) if loc is None or len(loc) > 0]
        for w in active:
            self.remotes[w].send((cmd, build_data(w, per[w])))
        out: list = []
        for w in active:
            out.extend(self._recv(w))
        return out

    @staticmethod
    def _cat_obs(parts: list):
        if isinstance(parts[0], dict):
            return {k: np.concatenate([p[k] for p in parts], axis=0) for k in parts[0]}
        return np.concatenate(parts, axis=0)

    # ---------- VecEnv API ----------

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        parts = [self._recv(i) for i in range(len(self.remotes))]
        return self._cat_obs(parts)

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions)
        for w, remote in enumerate(self.remotes):
            remote.send(("step", actions[self.offsets[w]:self.offsets[w + 1]]))
        self._waiting = True

    def step_wait(self):
        obs_p, rew_p, done_p, info_p = [], [], [], []
        for i in range(len(self.remotes)):
            obs, rew, done, infos = self._recv(i)
            obs_p.append(obs)
            rew_p.append(np.asarray(rew))
            done_p.append(np.asarray(done))
            info_p.extend(list(infos))
        self._waiting = False
        return (self._cat_obs(obs_p), np.concatenate(rew_p),
                np.concatenate(done_p), info_p)

    def close(self) -> None:
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for remote in self.remotes:
            try:
                remote.recv()
            except (EOFError, OSError):
                pass
        for p in self.processes:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    def get_attr(self, attr_name: str, indices=None):
        return self._fan("get_attr", lambda w, loc: (attr_name, loc), indices)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        self._fan("set_attr", lambda w, loc: (attr_name, value, loc), indices)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        return self._fan("env_method",
                         lambda w, loc: (method_name, loc, method_args, method_kwargs),
                         indices)

    def env_is_wrapped(self, wrapper_class, indices=None):
        return self._fan("env_is_wrapped", lambda w, loc: (wrapper_class, loc), indices)

    def seed(self, seed: Optional[int] = None):
        return self._fan("seed", lambda w, loc: seed)

    def get_images(self):
        raise NotImplementedError("MultiGPUVecEnv does not render")
