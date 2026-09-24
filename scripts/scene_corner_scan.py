#!/usr/bin/env python3
"""Is this scene an OBSTACLE corner, and is the spawn window on walkable ground?

scene_bend.py answers neither question: it reads only the camera poses. So it
called gnd_AUw360 a corner because the WALK turns 151 deg, when nothing forces
the ROBOT to turn, and it passed gnd_AUw170's spawn window 20-25, which sits on
grass -- every episode of every arm crashed at step 1 (2026-09-24, Joana: "it
starts on grass", and "that other gnd scene is not an obstacle corner so it
doesn't really show anything").

This reads the same label grid the REWARD scores against and calls the same
classify_pair the env calls to label a goal, so "corner" here means what it
means in training: the straight line from spawn to goal is BLOCKED and the
robot has to travel detour_min x further around something.

Frame positions come from the cloud's own traj_positions (with the env's y-flip),
not from the poses file, so frame i is in the same frame as the grid by
construction.

    python scripts/scene_corner_scan.py gnd_AUw170 gnd_AUw360 gnd_AU_180
    python scripts/scene_corner_scan.py quad2_00 --goal 55 --spawn 5 12
    python scripts/scene_corner_scan.py gnd_AUw170 --band 5 8 --top 8
"""
import argparse, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CLOUDS = "/scratch/m000204-pm06b/joana/outputs/scene_clouds/clouds"
TRAV = "config/traversability_v14.yaml"


def load_scene(scene, clouds_dir, trav_path, collision_threshold=0.1):
    from src.eval.traversability import load_traversability
    from src.eval.reward_map import build_label_grid
    from src.eval.goal_cases import scene_case_maps

    p = Path(clouds_dir) / f"{scene}_cloud.npz"
    if not p.exists():
        raise SystemExit(f"no cloud at {p}")
    d = np.load(p)
    pts, labs = d["points"], d["labels"].astype(int)
    scores = load_traversability(Path(trav_path) if trav_path else None)
    non_trav = scores <= collision_threshold
    if "traj_positions" not in d:
        raise SystemExit(f"{scene}: cloud has no traj_positions, cannot place frames")
    walk = (np.asarray(d["traj_positions"], float) * np.array([1.0, -1.0, 1.0]))[:, :2]
    # exactly the call SceneEnv makes (scene_env.py:1438), with its config defaults
    grid = build_label_grid(pts, labs, non_trav, res=0.1, inflate_m=0.1, fill_m=0.3,
                            fill_max_area_m2=10.0, ignore_classes=(),
                            walk_xy=walk, walk_halfwidth_m=0.4, inflate_classes=())
    return grid, scene_case_maps(grid, non_trav), walk


def frame_ok(maps, grid, xy):
    """Is the robot body clear at this frame? Same body_ok mask the reward uses."""
    i = int((xy[1] - grid.y0) / grid.res); j = int((xy[0] - grid.x0) / grid.res)
    if not (0 <= i < grid.labels.shape[0] and 0 <= j < grid.labels.shape[1]):
        return False, 0.0
    return bool(maps["body_ok"][i, j]), float(maps["dist"][i, j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--clouds_dir", default=CLOUDS)
    ap.add_argument("--trav", default=str(REPO / TRAV))
    ap.add_argument("--goal", type=int, help="score this goal frame only")
    ap.add_argument("--spawn", type=int, nargs=2, metavar=("LO", "HI"))
    ap.add_argument("--band", type=float, nargs=2, default=(5.0, 8.0),
                    metavar=("LO", "HI"), help="target spawn-goal distance, m")
    ap.add_argument("--detour_min", type=float, default=1.15,
                    help="the env's goal_case_detour_min: above this the straight line is blocked")
    ap.add_argument("--top", type=int, default=6, help="how many candidate windows to print")
    a = ap.parse_args()

    from src.eval.goal_cases import classify_pair

    for scene in a.scenes:
        print(f"\n{'='*78}\n{scene}")
        try:
            grid, maps, walk = load_scene(scene, a.clouds_dir, a.trav)
        except SystemExit as e:
            print(f"  SKIP: {e}"); continue
        N = len(walk)

        ok = np.array([frame_ok(maps, grid, walk[i])[0] for i in range(N)])
        clr = np.array([frame_ok(maps, grid, walk[i])[1] for i in range(N)])
        bad = np.where(~ok)[0]
        print(f"  {N} frames, {ok.sum()} spawnable ({100*ok.mean():.0f}%)")
        if len(bad):
            runs, s = [], bad[0]
            for k in range(1, len(bad) + 1):
                if k == len(bad) or bad[k] != bad[k-1] + 1:
                    runs.append((s, bad[k-1])); s = bad[k] if k < len(bad) else None
            print(f"  NOT spawnable: " + ", ".join(f"{x}-{y}" if x != y else str(x) for x, y in runs))

        if a.goal is not None and a.spawn:
            lo, hi = a.spawn; g = min(a.goal, N - 1)
            print(f"\n  goal {g}, spawn {lo}-{hi}:")
            print(f"  {'spawn':>6}{'ok':>5}{'clear':>8}{'straight':>10}{'detour':>9}{'width':>8}  case")
            for s in range(lo, min(hi, N - 1) + 1):
                o, c = frame_ok(maps, grid, walk[s])
                r = classify_pair(grid, maps, walk[s], walk[g], detour_min=a.detour_min)
                print(f"  {s:>6}{'y' if o else 'NO':>5}{c:>8.2f}{r.get('straight_m',0):>10.2f}"
                      f"{r.get('detour',float('nan')):>9.2f}{r.get('min_width_m',0):>8.2f}  {r['cls']}")
            continue

        # sweep: every (spawn, goal) pair inside the distance band, ranked by detour
        cands = []
        for s in range(N - 1):
            if not ok[s]:
                continue
            for g in range(s + 3, N):
                d = float(np.linalg.norm(walk[g] - walk[s]))
                if not (a.band[0] <= d <= a.band[1]):
                    continue
                r = classify_pair(grid, maps, walk[s], walk[g], detour_min=a.detour_min)
                if r["cls"] in ("offmap", "blocked"):
                    continue
                cands.append((s, g, d, r))
        if not cands:
            print(f"  no spawn/goal pair between {a.band[0]}-{a.band[1]} m"); continue

        corners = [c for c in cands if c[3]["detour"] >= a.detour_min]
        print(f"  {len(cands)} pairs in band, {len(corners)} are OBSTACLE corners "
              f"(detour >= {a.detour_min})")
        if not corners:
            best = max(cands, key=lambda c: c[3]["detour"])
            print(f"  VERDICT: no obstacle corner here. Best detour is {best[3]['detour']:.2f} "
                  f"(spawn {best[0]} -> goal {best[1]}), i.e. the straight line is walkable.")
            print(f"  This scene tests open navigation, not turning around something.")
            continue

        # group by goal frame, keep the widest contiguous spawn window per goal
        bygoal = {}
        for s, g, d, r in corners:
            bygoal.setdefault(g, []).append((s, d, r))
        rows = []
        for g, lst in bygoal.items():
            ss = sorted(x[0] for x in lst)
            run, best = [ss[0]], (ss[0], ss[0])
            for k in range(1, len(ss)):
                if ss[k] == ss[k-1] + 1:
                    run.append(ss[k])
                else:
                    if len(run) > best[1] - best[0] + 1: best = (run[0], run[-1])
                    run = [ss[k]]
            if len(run) > best[1] - best[0] + 1: best = (run[0], run[-1])
            sel = [x for x in lst if best[0] <= x[0] <= best[1]]
            rows.append((g, best, np.mean([x[2]["detour"] for x in sel]),
                         min(x[2]["min_width_m"] for x in sel),
                         min(x[1] for x in sel), max(x[1] for x in sel), len(sel)))
        rows.sort(key=lambda r: (-r[6], -r[2]))
        print(f"\n  {'goal':>5}{'spawn':>10}{'n':>4}{'detour':>9}{'min width':>11}{'distance':>16}")
        for g, (lo, hi), det, w, dmin, dmax, n in rows[:a.top]:
            print(f"  {g:>5}{f'{lo}-{hi}':>10}{n:>4}{det:>9.2f}{w:>11.2f}{f'{dmin:.1f}-{dmax:.1f} m':>16}")
        g, (lo, hi), det, w, dmin, dmax, n = rows[0]
        # TESTDIST must be the distance the robot DRIVES, i.e. the path around the
        # bend, not the straight line: quad2_00_far is 8.1 m straight and 13.6 m
        # along the walkway, and the straight figure under-budgeted it by a third
        # (2026-09-24, Joana: "did we add more steps than the training allows?").
        path_m = dmax * det
        print(f"\n  eval_scenes.env line   (TESTDIST is the PATH, {dmax:.1f} m straight x {det:.2f} detour):")
        print(f"  {scene:<12} GOAL_FRAME={g:<4} SPAWN_MIN={lo},SPAWN_MAX={hi} TESTDIST={path_m:.1f}  "
              f"# {dmin:.1f}-{dmax:.1f} m straight, detour {det:.2f}x, corridor {w:.1f} m: obstacle corner")


if __name__ == "__main__":
    main()
