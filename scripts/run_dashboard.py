"""Print the training curves of policy runs from their tensorboard event files,
no wandb, no screenshots (2026-09-19, Joana: "is there a way to show you these
results without screenshots").

For each run directory (the ppo_live_* output folder) it reads
<run>/tensorboard/*/events.out.tfevents.* and prints, for the chosen keys,
the latest value and a short history at evenly spaced steps.

    python scripts/run_dashboard.py /scratch/.../outputs/ppo_live_* \
        [--keys curriculum/goal_dist reward/crash reward/goal_bonus ...] [--points 6]
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

DEFAULT_KEYS = ["curriculum/goal_dist", "reward/goal_bonus", "reward/crash", "reward/coherence_crash",
                "reward/end_goal_dist_frac", "rollout/ep_len_mean", "rollout/ep_rew_mean", "diag/throttle"]


def load_scalars(run_dir: str):
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception as e:  # pragma: no cover
        sys.exit(f"tensorboard not importable ({e}); run with the neoverse env on PATH")
    series = {}
    for f in sorted(glob.glob(os.path.join(run_dir, "tensorboard", "**", "events.out.tfevents.*"), recursive=True)):
        acc = EventAccumulator(f, size_guidance={"scalars": 0}); acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            pts = [(ev.step, ev.value) for ev in acc.Scalars(tag)]
            series.setdefault(tag, []).extend(pts)
    for tag in series:
        series[tag].sort()
    return series


def _parent_of(run_dir: str):
    """The run directory a continued run was warm-started from, or None.
    Uses env_config['warmstart'] when present (runs after 2026-09-22); otherwise resolves the
    checkpoint path recorded in the directory name is not possible, so the chain simply stops."""
    import json
    p = os.path.join(run_dir, "env_config.json")
    if not os.path.exists(p):
        return None
    try:
        ws = json.load(open(p)).get("warmstart", "")
    except Exception:
        return None
    if not ws:
        return None
    ws = os.path.realpath(ws)                       # .../<run>/checkpoints/LATEST -> resolve the symlink
    d = os.path.dirname(os.path.dirname(ws))        # strip checkpoints/<file>
    return d if os.path.isdir(d) else None


def chain_of(run_dir: str, limit: int = 8):
    """[oldest ancestor, ..., run_dir] following warm starts backwards."""
    seen, chain, cur = set(), [], os.path.realpath(run_dir)
    while cur and cur not in seen and len(chain) < limit:
        seen.add(cur); chain.append(cur); cur = _parent_of(cur)
    return list(reversed(chain))


def merge_scalars(dirs):
    """Concatenate several runs' scalars by env step. PPO keeps its step counter across a warm
    start (reset_num_timesteps=False), so the steps already line up end to end."""
    out = {}
    for d in dirs:
        for tag, pts in load_scalars(d).items():
            out.setdefault(tag, []).extend(pts)
    for tag in out:
        out[tag].sort()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--chain", action="store_true",
                    help="follow each run's warm-start ancestry and plot the whole history as one curve")
    ap.add_argument("--keys", nargs="+", default=DEFAULT_KEYS)
    ap.add_argument("--points", type=int, default=6, help="history samples per key")
    ap.add_argument("--smooth", type=int, default=0,
                    help="average each key over a window of this many env steps before sampling. "
                         "The resident scene rotates every --scene_rotate steps (3000 in the campus "
                         "arms) and one logged point is one rollout (~2048 steps), so a single point "
                         "is essentially one scene. Use a full rotation cycle (scenes x rotate, "
                         "39000 for 13 scenes) to compare learning rather than scene difficulty.")
    args = ap.parse_args()
    for run in args.runs:
        dirs = chain_of(run) if args.chain else [run]
        series = merge_scalars(dirs) if len(dirs) > 1 else load_scalars(run)
        name = os.path.basename(run.rstrip("/"))
        if len(dirs) > 1:
            print(f"\n   chained {len(dirs)} runs: " + " -> ".join(os.path.basename(d)[-14:] for d in dirs))
        tag = ("memory" if "bm5" in name else "chunk" if "chunk" in name else
               "A-map" if ("hyb" in name or "map_then" in name) else "A (4 GPU)" if "_g4_" in name else "A-like (2 GPU)")
        steps = max((s[-1][0] for s in series.values() if s), default=0)
        print(f"\n== {tag}: ...{name[-48:]}  ({steps} env steps logged{', smoothed over ' + str(args.smooth) + ' steps' if args.smooth else ''})")
        if not series:
            print("   no tensorboard events found"); continue
        for k in args.keys:
            s = series.get(k)
            if not s:
                print(f"   {k:<28} (not logged)"); continue
            if args.smooth > 0:
                sm = []
                for i, (st, _) in enumerate(s):
                    lo = st - args.smooth
                    win = [v for stp, v in s[max(0, i - 400):i + 1] if stp >= lo]
                    if win: sm.append((st, sum(win) / len(win)))
                s = sm or s
            n = len(s); idx = sorted(set(int(round(i * (n - 1) / max(1, args.points - 1))) for i in range(args.points)))
            hist = "  ".join(f"{s[i][0] // 1000}k:{s[i][1]:.3g}" for i in idx)
            print(f"   {k:<28} latest {s[-1][1]:9.3f}   | {hist}")


if __name__ == "__main__":
    main()
