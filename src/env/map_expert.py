"""Scripted, privileged driver for recording demonstrations (2026-09-06).

The expert reads what the policy never sees: the scene's label grid, the
recorded walk and the goal's ground class. It walks along the recorded walk
toward the goal, then straight to it, and STOPS where the reward says a
correct stop is: inside the goal radius for a walkable goal, within the verge
radius of the refusal point for a lawn goal. It never emits a still step by
accident, because a still step is a halt to the env.

Behavior cloning on the (frame, goal vector, action) triples it records is a
supervised probe of "can the diffused frame locate the map's verge at all",
with RL taken out of the loop. Joana, 2026-09-06: "train a policy to firstly
learn the difference between grass and sidewalk".

Used through eval_policy.py --expert map (same env build, adoption, videos and
outcome tables as a policy eval; the expert's outcomes verify the reward
machinery end to end: an expert that stops at the verge must be scored as
HALTED with halt_at_verge).
"""
from __future__ import annotations

import numpy as np


def _wrap(a: float) -> float:
    return float(np.arctan2(np.sin(a), np.cos(a)))


class MapExpert:
    def __init__(self, env, lookahead_m: float = 1.0, verge_margin_m: float = 0.4,
                 rejoin_m: float = 2.0, turn_throttle: float = 0.4):
        self.u = env.unwrapped
        self.lookahead = float(lookahead_m)
        self.verge_margin = float(verge_margin_m)
        self.rejoin = float(rejoin_m)
        self.turn_throttle = float(turn_throttle)
        self.n_stop = 0

    # SB3 signature so eval_policy and save_rollout_video need no change.
    def predict(self, obs, deterministic: bool = True):
        return self.act(), None

    def _stop(self) -> np.ndarray:
        self.n_stop += 1
        if self.u.cfg.stop_action:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([0.0, 0.0], dtype=np.float32)

    def target(self):
        """(target_xy, stop_radius, is_lawn) for the current episode."""
        u = self.u
        cfg = u.cfg
        goal = np.asarray(u._goal_world, dtype=float)[:2]
        gt = float(getattr(u, "_goal_traversable", float("nan")))
        lawn = (gt == 0.0)
        if not lawn:
            return goal, float(cfg.goal_radius) * 0.7, False
        vp = u._refusal_point(u._goal_world)
        if vp is None:
            return goal, max(float(cfg.refusal_dist_m) - self.verge_margin, 0.5), True
        return np.asarray(vp, dtype=float)[:2], max(float(cfg.refusal_verge_m) - self.verge_margin, 0.5), True

    def act(self) -> np.ndarray:
        u = self.u
        cfg = u.cfg
        pose = u._robot_pose_world
        p = np.asarray(pose[:2, 3], dtype=float)
        yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
        target, stop_r, lawn = self.target()
        d_t = float(np.linalg.norm(target - p))
        if d_t <= stop_r:
            return self._stop()
        # Waypoint: along the recorded walk toward the target while far from
        # it, straight at it once close. Rejoin the walk first if off it.
        wp = target
        walk = getattr(u, "_walk_xy", {}).get(u._scene_id)
        if walk is not None and len(walk) > 1 and d_t > self.lookahead:
            walk = np.asarray(walk, dtype=float)[:, :2]
            i_r = int(np.argmin(np.linalg.norm(walk - p[None, :], axis=1)))
            i_t = int(np.argmin(np.linalg.norm(walk - target[None, :], axis=1)))
            if float(np.linalg.norm(walk[i_r] - p)) > self.rejoin:
                wp = walk[i_r]
            elif i_t != i_r:
                step = 1 if i_t > i_r else -1
                i, acc = i_r, 0.0
                while i != i_t:
                    acc += float(np.linalg.norm(walk[i + step] - walk[i]))
                    i += step
                    if acc >= self.lookahead:
                        break
                wp = walk[i]
        dyaw = _wrap(float(np.arctan2(wp[1] - p[1], wp[0] - p[0])) - yaw)
        a_yaw = float(np.clip(dyaw / float(cfg.yaw_step_rad), -1.0, 1.0))
        if abs(dyaw) < 0.35:
            thr = 1.0
        elif abs(dyaw) < 0.9:
            thr = 0.6
        else:
            thr = self.turn_throttle
        # do not overshoot a close target, but never a still step: the env
        # reads throttle < halt_throttle_eps as a halt
        floor = float(cfg.halt_throttle_eps) + 0.1
        thr = min(thr, max(d_t / float(cfg.step_size_m), floor))
        thr = max(thr, floor)
        a = [thr, a_yaw] + ([-1.0] if cfg.stop_action else [])
        return np.asarray(a, dtype=np.float32)
