"""Offline test of the CORNER EPISODE rule through the real sampler, no cluster
(2026-09-21, Joana: "the robot can't just go in a straight line and has to detour").

Stub L-shaped walk (15 m along +x, 90 deg bend, 15 m along +y) with a walkable band
0.8 m to each side and an obstacle block filling the inside of the corner.
Checks, with GOALTURN=45 GOALTURNMIX=1.0:
  1. every episode spawns BEFORE the bend and its goal lies AFTER it;
  2. the corner flag survives goal REDRAWS inside one episode (goal-support / goal-case
     retries): all 24 redraws still land beyond the bend;
  3. the goal-case classifier (GOALCASE=corner:1.0 uses it) calls the straight line
     blocked with a detour >= 1.15 for those pairs -> the "must detour" guarantee.
Run:  python tests/test_corner_rule.py
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.real_calibrated import CalibratedRealWorldBackend, NavCalibration   # noqa: E402
from src.eval.goal_cases import classify_pair, scene_case_maps                    # noqa: E402
from src.eval.reward_map import VOID, LabelGrid                                   # noqa: E402

STEP, LEG = 0.5, 15.0
N_LEG = int(LEG / STEP)              # 30 frames per leg; bend at frame 30
WALK_HALF_W = 0.8                    # walkable band half-width (m)
BAND = (4.0, 10.0)                   # goal distance band (curriculum level)


def l_walk():
    a = np.stack([np.arange(N_LEG) * STEP, np.zeros(N_LEG), np.zeros(N_LEG)], 1)
    b = np.stack([np.full(N_LEG, LEG), np.arange(1, N_LEG + 1) * STEP, np.zeros(N_LEG)], 1)
    return np.concatenate([a, b]).astype(np.float64)


def make_backend(turn_deg: float, mix: float):
    pos = l_walk()
    head = np.diff(pos, axis=0, append=pos[-1:] + (pos[-1:] - pos[-2:-1]))
    head /= np.linalg.norm(head, axis=1, keepdims=True)
    cal = object.__new__(NavCalibration)
    cal.positions, cal.headings = pos, head
    be = object.__new__(CalibratedRealWorldBackend)
    be._calib = {"L": cal}
    be.cfg = SimpleNamespace(
        spawn_min_frame=0, spawn_max_frame=None, goal_frame=30, goal_frame_range=(0, len(pos) - 1),
        goal_xy_override=None, goal_dir_360=False, goal_dist_m=7.0, goal_dist_window_m=1.0,
        goal_dist_range=BAND, goal_min_sep_m=1.0, spawn_heading_from_walk=True,
        spawn_yaw_jitter_deg=0.0, spawn_lat_jitter_m=0.0,
        spawn_frames_by_scene="L:" + ",".join(str(f) for f in range(2, len(pos) - 4)),
        goal_turn_deg=turn_deg, goal_turn_mix=mix, goal_turn_beyond_m=(2.0 if turn_deg > 0 else 0.0))
    return be, pos


def make_grid(pos):
    res = 0.1
    x0, y0 = -3.0, -3.0
    W = H = int((LEG + 6.0) / res)
    xs = x0 + (np.arange(W) + 0.5) * res
    ys = y0 + (np.arange(H) + 0.5) * res
    X, Y = np.meshgrid(xs, ys)
    # distance to the walk polyline (two axis-aligned legs)
    d1 = np.hypot(np.clip(X, 0.0, LEG) - X, Y)                       # leg 1: y = 0, x in [0, LEG]
    d2 = np.hypot(X - LEG, np.clip(Y, 0.0, LEG) - Y)                 # leg 2: x = LEG, y in [0, LEG]
    d = np.minimum(d1, d2)
    labels = np.full((H, W), 4, np.int16)                            # 4 = obstacle (non-walkable)
    labels[d <= WALK_HALF_W] = 2                                      # 2 = walkable pavement
    labels[d > 4.0] = VOID
    return LabelGrid(x0=x0, y0=y0, res=res, labels=labels, n_points=np.ones((H, W), np.int32))


def nearest_frame(pos, xy):
    return int(np.argmin(np.linalg.norm(pos[:, :2] - np.asarray(xy)[:2], axis=1)))


def run(turn_deg, mix, n_ep=300, redraws=24):
    be, pos = make_backend(turn_deg, mix)
    grid = make_grid(pos)
    non_trav = np.zeros(14, bool); non_trav[4] = True
    maps = scene_case_maps(grid, non_trav)
    st = {"ep": 0, "spawn_pre": 0, "goal_post": 0, "both": 0, "behind": 0,
          "redraw_post": 0, "redraw_n": 0, "cls": {}, "detours": [], "case_ok_after_redraw": 0}
    for i in range(n_ep):
        rng = np.random.default_rng(1000 + i)
        sp = be.sample_start_pose("L", rng)
        yaw = be.last_spawn_base_yaw()
        sxy = sp[:2, 3]
        goals = [be.sample_goal_position("L", rng, sxy, cone_yaw=yaw) for _ in range(redraws)]
        g = goals[0]
        fs, fg = nearest_frame(pos, sxy), nearest_frame(pos, g)
        st["ep"] += 1
        st["spawn_pre"] += int(fs < N_LEG); st["goal_post"] += int(fg > N_LEG)
        st["both"] += int(fs < N_LEG and fg > N_LEG)
        fwd = np.array([np.cos(yaw), np.sin(yaw)])
        st["behind"] += int(float((g[:2] - sxy) @ fwd) <= 0.0)
        for gg in goals[1:]:
            st["redraw_n"] += 1; st["redraw_post"] += int(nearest_frame(pos, gg) > N_LEG)
        r = classify_pair(grid, maps, sxy, g[:2])
        st["cls"][r["cls"]] = st["cls"].get(r["cls"], 0) + 1
        if np.isfinite(r.get("detour", np.inf)):
            st["detours"].append(float(r["detour"]))
        # what GOALCASE=corner:1.0 does: keep redrawing until the classifier says corner
        for gg in goals:
            rr = classify_pair(grid, maps, sxy, gg[:2])
            if rr["cls"] in ("corner", "corner+narrow"):
                st["case_ok_after_redraw"] += 1; break
    return st


def main():
    for turn, mix, name in ((45.0, 1.0, "RULE ON  (GOALTURN=45 GOALTURNMIX=1.0)"), (0.0, 0.0, "RULE OFF")):
        st = run(turn, mix)
        n = st["ep"]
        print(f"== {name}: {n} episodes on the stub L-walk (bend at frame {N_LEG}, band {BAND})")
        print(f"   spawn before bend {st['spawn_pre']}/{n} | goal after bend {st['goal_post']}/{n} | both {st['both']}/{n} | goals behind robot {st['behind']}")
        print(f"   redraws inside the episode still beyond the bend: {st['redraw_post']}/{st['redraw_n']}")
        d = np.asarray(st["detours"])
        print(f"   goal-case classifier on the first draw: {st['cls']} | detour ratio median {np.median(d):.2f} (min {d.min():.2f}) " if len(d) else "   no finite detours")
        print(f"   GOALCASE=corner:1.0 would end with a blocked-straight-line goal in {st['case_ok_after_redraw']}/{n} episodes")
    st = run(45.0, 1.0)
    assert st["both"] >= 0.95 * st["ep"], "corner rule: spawn-before / goal-after share too low"
    assert st["behind"] == 0, "a goal landed behind the robot"
    assert st["redraw_post"] == st["redraw_n"], "corner flag was lost on a goal redraw"
    n_corner = st["cls"].get("corner", 0) + st["cls"].get("corner+narrow", 0)
    assert n_corner >= 0.9 * st["ep"], "straight spawn->goal line is walkable in too many corner episodes"
    assert st["case_ok_after_redraw"] == st["ep"], "GOALCASE=corner:1.0 could not find a detour goal in every episode"
    print("OK")


if __name__ == "__main__":
    main()
