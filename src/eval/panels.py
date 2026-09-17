"""Drawings shared by the offline survey (scripts/check_rewards.py) and the
online training panel (WandbImagePanel in scripts/train_ppo_real.py).

They were written twice, which meant a picture verified in the survey was not
the picture training produced (Joana, 2026-09-16: "if it isn't the same then
the checks don't have any impact"). One implementation, two call sites.

Everything here is pure numpy + cv2 and takes explicit arguments, so it can be
unit-tested without an environment.
"""
from __future__ import annotations

import numpy as np

from .reward_2d import (_project_points, _footprint_corners_world, _fill_polygon,
                        GO2_BODY_LENGTH, GO2_BODY_WIDTH)

PLAN_COLOR = (255, 160, 0)      # the decision the policy committed to
BOX_IN = (255, 0, 255)          # a stored frame that contains the near box
BOX_OUT = (120, 120, 120)       # ... and one that does not


def integrate_plan(pose: np.ndarray, action: np.ndarray, step_size_m: float,
                   yaw_step_rad: float) -> np.ndarray:
    """Poses a decision leads to: (x, y, yaw) per sub-action, starting AFTER the
    first one. `action` is flat (v, w, v, w, ...); one pair for a per-step
    policy, k pairs for a chunked one. Same integration as SceneEnv.step."""
    a = np.asarray(action, dtype=np.float64).ravel()
    x, y = float(pose[0, 3]), float(pose[1, 3])
    yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
    out = []
    for k in range(a.size // 2):
        yaw += float(a[2 * k + 1]) * float(yaw_step_rad)
        v = float(a[2 * k]) * float(step_size_m)
        x += v * np.cos(yaw); y += v * np.sin(yaw)
        out.append((x, y, yaw))
    return np.asarray(out, dtype=np.float64)


def draw_plan(img: np.ndarray, pose: np.ndarray, plan: np.ndarray, K, w2c,
              look_ahead_dist: float = 0.0, color=PLAN_COLOR, label: bool = True) -> np.ndarray:
    """Draw a committed decision on `img`: the path through the planned poses,
    and the reward's footprint rectangle at each of them when look_ahead_dist>0
    (what the reward would read there, not just where the robot goes)."""
    import cv2
    if plan is None or len(plan) == 0:
        return img
    pts3 = np.concatenate([np.array([[pose[0, 3], pose[1, 3], 0.0]]),
                           np.c_[plan[:, 0], plan[:, 1], np.zeros(len(plan))]], axis=0)
    uv, front = _project_points(pts3, np.asarray(K), np.asarray(w2c))
    px = [tuple(int(v) for v in np.round(uv[i])) for i in range(len(pts3)) if front[i]]
    for a, b in zip(px[:-1], px[1:]):
        cv2.line(img, a, b, color, 2, cv2.LINE_AA)
    for q in px[1:]:
        cv2.circle(img, q, 3, color, -1, cv2.LINE_AA)
    if look_ahead_dist > 0:
        for (x, y, yaw) in plan:
            cw = _footprint_corners_world(np.array([x, y, 0.0]),
                                          np.array([np.cos(yaw), np.sin(yaw), 0.0]),
                                          look_ahead_dist=look_ahead_dist,
                                          length=GO2_BODY_LENGTH, width=GO2_BODY_WIDTH)
            u, f = _project_points(cw, np.asarray(K), np.asarray(w2c))
            if f.all():
                cv2.polylines(img, [np.round(u).astype(np.int32).reshape(-1, 1, 2)], True, color, 1, cv2.LINE_AA)
    if label:
        n = len(plan)
        cv2.putText(img, f"orange = this decision ({n} step{'s' if n > 1 else ''})", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def box_in_frame(corners_world: np.ndarray, K, w2c, sem: np.ndarray,
                 non_traversable_mask: np.ndarray, min_visible: float = 0.5):
    """(uv, inside, non_walkable_share) of a world box in one stored frame.
    `inside` uses the SAME rule as compute_reward's memory branch."""
    uv, front = _project_points(corners_world, np.asarray(K), np.asarray(w2c))
    if not front.all():
        return uv, False, float("nan")
    H, W = np.asarray(sem).shape[:2]
    mask = _fill_polygon(H, W, uv)
    x, y = uv[:, 0], uv[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
    inside = area > 0 and int(mask.sum()) >= min_visible * area
    share = float("nan")
    if int(mask.sum()) > 0:
        cl = np.asarray(sem)[mask]
        share = float((non_traversable_mask[np.clip(cl, 0, len(non_traversable_mask) - 1)] & (cl != 0)).mean())
    return uv, inside, share


def memory_strip(frame_memory, corners_world: np.ndarray, palette: np.ndarray,
                 non_traversable_mask: np.ndarray, robot_xy: np.ndarray,
                 out_h: int, out_w: int, horizontal: bool = False,
                 min_visible: float = 0.5) -> "np.ndarray | None":
    """The stored frames with the near box projected into each: magenta where
    the box is inside (that frame can answer), grey and dimmed where it is not.
    Each tile is labelled with its age, whether the box is in, the share of
    non-walkable pixels inside it, and the distance from the robot to the box --
    which must change by one step per tile if the projection is right.

    frame_memory: [(semantic_image, K, w2c)], OLDEST first (as the env stores it).
    """
    import cv2
    fm = list(frame_memory or [])
    if not fm:
        return None
    box_xy = np.asarray(corners_world)[:, :2].mean(0)
    dist = float(np.linalg.norm(box_xy - np.asarray(robot_xy)[:2]))
    tiles = []
    n = len(fm)
    th = out_h if horizontal else max(out_h // n, 24)
    tw = max(out_w // n, 64) if horizontal else out_w
    for age, (sem, K, w2c) in enumerate(reversed(fm), start=1):   # newest first
        t = palette[np.clip(np.asarray(sem).astype(np.int64), 0, len(palette) - 1)].copy()
        uv, inside, share = box_in_frame(corners_world, K, w2c, sem, non_traversable_mask, min_visible)
        cv2.polylines(t, [np.round(uv).astype(np.int32).reshape(-1, 1, 2)], True,
                      BOX_IN if inside else BOX_OUT, 2, cv2.LINE_AA)
        if not inside:
            t = (t * 0.45).astype(np.uint8)
        t = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
        txt = f"t-{age} {'IN' if inside else 'out'} {dist:.2f}m" + (f" nw {share:.2f}" if share == share else "")
        cv2.putText(t, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(t, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    strip = np.concatenate(tiles, axis=1 if horizontal else 0)
    if not horizontal and strip.shape[0] < out_h:
        strip = np.concatenate([strip, np.zeros((out_h - strip.shape[0], out_w, 3), np.uint8)], axis=0)
    return strip[:out_h, :out_w] if not horizontal else strip
