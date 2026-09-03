"""Is the goal ever BEHIND the robot? Direct test of the real goal sampler.

Joana found it in the ancestor eval: episodes 14 and 17 put the goal behind the
spawn point, and a forward-only robot can only time out there. The cause was
that the spawn HEADING and the goal CONE were derived from two different path
frames -- the heading from the spawn frame f, the cone from the frame nearest
the laterally-jittered spawn, which can be f+-1. Where the walk turns, adjacent
tangents disagree by up to 150 deg (gnd_AUw360 frames 58-60 and 68-71), so the
two pointed opposite ways.

Smoothing the tangent does NOT fix this and the first version of this test is
what proved it: at a genuine reversal a +-3 frame span flips just as hard as a
+-1 span (100.4 deg either way on the hairpin below). The fix is to centre the
cone on the robot's own heading, so the two cannot disagree by construction.

This calls the REAL `CalibratedRealWorldBackend.sample_goal_position` -- not a copy --
so it also catches the case where the fix is in the sampler but the caller
never passes the heading. `scripts/goal_audit.py` reports the same statistic on
real scene paths; this one owns the adversarial geometry.

    python scripts/test_goal_cone.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env.real_calibrated import CalibratedRealWorldBackend

CONE_DEG = 50.0
N = 20000


def hairpin(n=40, step=0.36, offset=0.3):
    """An out-and-back walk. The turnaround is the adversarial case: the two
    legs run antiparallel and are only `offset` apart, so a spawn on one leg
    can be nearest a frame on the OTHER leg."""
    out = [(i * step, 0.0) for i in range(n)]
    back = [((n - 1) * step - i * step, offset) for i in range(1, n)]
    return np.array([(x, y, 0.0) for x, y in out + back], dtype=np.float64)


def make_world(path):
    w = object.__new__(CalibratedRealWorldBackend)
    w.cfg = SimpleNamespace(goal_dir_360=True, goal_dist_range=(2.0, 8.0),
                            goal_cone_deg=CONE_DEG, goal_min_sep_m=0.0,
                            goal_xy_override=None, goal_frame_range=None)
    w._calib = {"hairpin": SimpleNamespace(positions=path)}
    return w


def tangent(path, i):
    fw = path[min(i + 1, len(path) - 1), :2] - path[max(i - 1, 0), :2]
    return float(np.arctan2(fw[1], fw[0]))


def wrap_deg(a):
    return np.degrees((a + np.pi) % (2 * np.pi) - np.pi)


def run(path, world, pass_yaw, lat_jitter=0.25, yaw_jitter_deg=20.0, seed=0):
    """Replicate a reset: spawn at frame f with lateral+yaw jitter, then sample
    a goal. Report |bearing to goal - robot heading|."""
    rng = np.random.default_rng(seed)
    err = np.empty(N)
    for k in range(N):
        f = int(rng.integers(1, len(path) - 1))
        base = tangent(path, f)
        spawn = path[f, :2] + rng.uniform(-1, 1) * lat_jitter * np.array(
            [-np.sin(base), np.cos(base)])
        yaw = base + np.deg2rad(float(rng.uniform(-1, 1)) * yaw_jitter_deg)
        g = world.sample_goal_position(
            "hairpin", rng, spawn,
            spawn_yaw=(yaw if pass_yaw else None))
        err[k] = abs(wrap_deg(np.arctan2(*(g[:2] - spawn)[::-1]) - yaw))
    return err


def callers_pass_the_heading():
    """The sampler can be correct and the fix still dead if a caller never
    passes `spawn_yaw` -- the same silent-failure shape as the LoRA inject list
    and the RW5 no-op. Check every call site in the env, statically.

    Only the ENV is required to pass it: offline tools (goal_audit,
    drive_preview) legitimately audit the un-jittered path."""
    import ast
    ok = True
    for rel in ("src/env/scene_env.py",):
        src = (Path(__file__).resolve().parents[1] / rel).read_text()
        tree = ast.parse(src)
        sites = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "sample_goal_position"]
        for c in sites:
            if not any(k.arg == "spawn_yaw" for k in c.keywords):
                print(f"FAIL: {rel}:{c.lineno} calls sample_goal_position "
                      f"without spawn_yaw -- the cone falls back to the path "
                      f"tangent and the fix is inert here.")
                ok = False
        print(f"{rel}: {len(sites)} call site(s), "
              f"{'all pass spawn_yaw' if ok else 'SOME MISSING'}")
    return ok


def main():
    path = hairpin()
    world = make_world(path)

    print(f"hairpin walk, {len(path)} frames, cone {CONE_DEG:.0f}deg "
          f"(so |bearing - heading| must stay under {CONE_DEG / 2:.0f}deg), "
          f"{N} resets each\n")
    print(f"{'cone centred on':<28}{'mean':>7}{'p95':>7}{'max':>8}"
          f"{'>90deg BEHIND':>15}{'>cone/2':>10}")
    print("-" * 75)

    ok = True
    for label, pass_yaw in (("path tangent (OLD)", False),
                            ("robot heading (NEW)", True)):
        e = run(path, world, pass_yaw)
        behind = 100.0 * (e > 90).mean()
        over = 100.0 * (e > CONE_DEG / 2 + 1e-6).mean()
        print(f"{label:<28}{e.mean():7.1f}{np.percentile(e, 95):7.1f}"
              f"{e.max():8.1f}{behind:14.2f}%{over:9.2f}%")
        if pass_yaw:
            # The bound is by construction, so anything above it means the
            # heading is not reaching the sampler -- a real failure, not noise.
            if e.max() > CONE_DEG / 2 + 1e-6:
                print(f"\nFAIL: with the heading passed, a goal landed "
                      f"{e.max():.1f}deg off it -- above the {CONE_DEG / 2:.0f}deg "
                      f"cone half-width. The heading is not being used.")
                ok = False
            elif behind > 0:
                print("\nFAIL: a goal is still behind the robot.")
                ok = False

    print()
    ok = callers_pass_the_heading() and ok
    print()
    if ok:
        print(f"PASS: every goal within {CONE_DEG / 2:.0f}deg of the heading; "
              f"none behind the robot.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
