#!/usr/bin/env python3
"""Which SCENE produced that dip? Bin the training curves by the resident scene.

The world rotates every --scene_rotate steps (3000) and one logged point is one
PPO rollout (~2048), so a single point is essentially one scene and an unsmoothed
curve is mostly a tour of scene difficulty. `diag/scene_idx` records which scene
was resident, so we can group the metrics by it instead of guessing.

    python scripts/scene_difficulty.py <run dir> [<run dir> ...] [--keys ...]

Points where scene_idx is not close to an integer are dropped: those rollouts
straddle a rotation and belong to two scenes at once.
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from collections import defaultdict

DEFAULT_KEYS = ["rollout/ep_rew_mean", "reward/crash", "rollout/ep_len_mean",
                "reward/goal_bonus", "diag/throttle"]


def load_scalars(run_dir, keys):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:
        sys.exit(f"tensorboard not importable ({e}); use the neoverse env")
    want = set(keys) | {"diag/scene_idx"}
    series = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(run_dir, "tensorboard", "**", "events.out.tfevents.*"),
                              recursive=True)):
        acc = EventAccumulator(f, size_guidance={"scalars": 0}); acc.Reload()
        for k in acc.Tags().get("scalars", []):
            if k in want:
                for ev in acc.Scalars(k):
                    series[k][ev.step] = ev.value
    return series


def scenes_for(run_dir, override):
    if override:
        return override.split(",")
    p = os.path.join(run_dir, "env_config.json")
    if os.path.isfile(p):
        try:
            v = json.load(open(p)).get("scenes")
            if isinstance(v, str):
                return [s for s in v.replace(",", " ").split() if s]
            if isinstance(v, list):
                return list(v)
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    ap.add_argument("--scenes", default=None, help="comma list, if env_config.json has none")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="how close scene_idx must be to an integer to count")
    args = ap.parse_args()

    for run in args.runs:
        s = load_scalars(run, args.keys)
        idx = s.get("diag/scene_idx")
        print(f"\n== {os.path.basename(run)[-46:]}")
        if not idx:
            print("   no diag/scene_idx logged (older run?) -- skipped"); continue
        names = scenes_for(run, args.scenes)
        rows = defaultdict(lambda: defaultdict(list))
        dropped = 0
        for step, v in sorted(idx.items()):
            i = round(v)
            if abs(v - i) > args.tol:
                dropped += 1; continue
            for k in args.keys:
                if step in s.get(k, {}):
                    rows[i][k].append(s[k][step])
        if not rows:
            print("   every point straddled a rotation -- nothing to report"); continue
        hdr = f"{'scene':<16}{'pts':>5}" + "".join(f"{k.split('/')[-1][:13]:>15}" for k in args.keys)
        print("   " + hdr)
        order = sorted(rows, key=lambda i: -(sum(rows[i].get('reward/crash', [0])) /
                                             max(1, len(rows[i].get('reward/crash', [1])))))
        for i in order:
            nm = names[i] if names and i < len(names) else f"idx {i}"
            n = max(len(v) for v in rows[i].values())
            line = f"{nm:<16}{n:>5}"
            for k in args.keys:
                vals = rows[i].get(k, [])
                line += f"{(sum(vals)/len(vals)):>15.2f}" if vals else f"{'--':>15}"
            print("   " + line)
        print(f"   ({dropped} points dropped for straddling a rotation; "
              f"scenes ordered worst crash last)")


if __name__ == "__main__":
    sys.exit(main())
