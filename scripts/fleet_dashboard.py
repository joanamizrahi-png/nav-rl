"""Every running arm on one page: which one is learning the behaviour we want?

Reads the SB3 tables straight out of each arm's slurm .out -- no wandb API, no
GPU, seconds on the login node -- and overlays all arms on a small grid of the
signals that actually answer the question:

  goal_dist          the competence meter. It only advances when the policy
                     wins >= 50% of the last 100 episodes, so it is immune to
                     the reward's shifting scale AND to scene rotation.
  how episodes end   derived from the per-step reward means: goal_bonus,
                     crash and coherence_crash are one-shot terminals, so
                     (mean per step) * ep_len / penalty = fraction of episodes
                     ending that way. The remainder is timeout (or halted).
  ep_len_mean        surviving longer is the first thing that has to happen.
  throttle           creeping is freezing's cousin: ppo_240704 ran at 28%.
  halted             the new terminal -- the first direct metric for
                     "walked to the boundary and stopped".

Arms are named by what DISTINGUISHES them: the ledger label with every token
shared by all arms stripped, so "base" comes out as "s1" and the halt arm as
"halt5-s1" instead of eleven identical 80-character strings.

    python scripts/fleet_dashboard.py 463164 463165 ... --out fleet.png
    python scripts/fleet_dashboard.py --since "2026-09-03 00:00" --out fleet.png
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np

LOGDIR = Path("/scratch/m000204-pm06b/joana")
LEDGER = LOGDIR / "launch_ledger.log"

SECTION = re.compile(r"^\|\s+(\w+)/\s+\|\s+\|\s*$")
KV = re.compile(r"^\|\s+(\S+)\s+\|\s+([-+]?[\d.]+(?:e[-+]?\d+)?|nan|inf)\s+\|\s*$")
TABLE_END = re.compile(r"^-{20,}\s*$")


def parse_tables(path: Path) -> dict:
    """{key: [values...]} with one entry per SB3 dump, keys like 'rollout/ep_len_mean'."""
    series: dict[str, list] = {}
    cur: dict[str, float] = {}
    section = ""
    for ln in path.read_text(errors="ignore").splitlines():
        m = SECTION.match(ln)
        if m:
            section = m.group(1) + "/"
            continue
        m = KV.match(ln)
        if m:
            try:
                cur[section + m.group(1)] = float(m.group(2))
            except ValueError:
                pass
            continue
        if TABLE_END.match(ln) and cur:
            # a dump is complete when we hit the closing rule with content
            if "time/total_timesteps" in cur:
                for k, v in cur.items():
                    series.setdefault(k, []).append(v)
                # keys absent from this dump get NaN so lengths stay aligned
                n = len(series["time/total_timesteps"])
                for k in series:
                    if len(series[k]) < n:
                        series[k].append(float("nan"))
            cur = {}
    return series


def ledger_labels() -> dict[int, str]:
    out: dict[int, str] = {}
    if not LEDGER.exists():
        return out
    job = None
    for ln in LEDGER.read_text(errors="ignore").splitlines():
        m = re.match(r"^=== job (\d+)", ln)
        if m:
            job = int(m.group(1))
        m = re.match(r"^\s+label\s+(\S+)", ln)
        if m and job is not None:
            out[job] = m.group(1)
    return out


def short_names(labels: dict[int, str]) -> dict[int, str]:
    """Strip every token shared by ALL arms; what is left is what differs."""
    toks = {j: l.split("-") for j, l in labels.items()}
    if not toks:
        return {}
    common = set.intersection(*(set(t) for t in toks.values()))
    out = {}
    for j, t in toks.items():
        kept = [x for x in t if x not in common]
        out[j] = "-".join(kept) if kept else "base"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs", nargs="*", type=int)
    ap.add_argument("--since", default="",
                    help="if no jobs given: every ppo-real log modified after this")
    ap.add_argument("--out", default="fleet.png")
    ap.add_argument("--scenes", default="gnd_AU_180,gnd_AUd210,gnd_AUw210,gnd_AUw360,gnd_AUw330",
                    help="scene names in diag/scene_idx order -- the PRUNED list the "
                         "env actually loaded (the banner prints it), not the one "
                         "requested")
    ap.add_argument("--ceilings", default="36.8,55.8,68.1,86.1,89.8",
                    help="REACHABLE %% per scene from goal_audit (2-8 m, new cone, "
                         "2026-09-03). A goal rate is only readable against this: "
                         "on gnd_AU_180 nearly half the goals sit on grass.")
    ap.add_argument("--recent", type=int, default=6,
                    help="per-scene panels use the last N dumps on each scene")
    args = ap.parse_args()
    scene_names = args.scenes.split(",")
    ceilings = [float(v) for v in args.ceilings.split(",")]

    jobs = list(args.jobs)
    if not jobs:
        cmd = ["find", str(LOGDIR), "-maxdepth", "1", "-name", "slurm-ppo-real-*.out"]
        if args.since:
            cmd += ["-newermt", args.since]
        for f in subprocess.run(cmd, capture_output=True, text=True).stdout.split():
            m = re.search(r"slurm-ppo-real-(\d+)\.out$", f)
            if m:
                jobs.append(int(m.group(1)))
        jobs.sort()
    if not jobs:
        raise SystemExit("no jobs given and none found")

    labels = ledger_labels()
    names = short_names({j: labels.get(j, str(j)) for j in jobs})

    data = {}
    for j in jobs:
        p = LOGDIR / f"slurm-ppo-real-{j}.out"
        if not p.exists():
            print(f"[skip] {j}: no log"); continue
        s = parse_tables(p)
        if "time/total_timesteps" not in s:
            print(f"[skip] {j}: no SB3 tables yet (still loading?)"); continue
        data[j] = s

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("tab20")
    colors = {j: cmap(i % 20) for i, j in enumerate(data)}

    def get(s, k):
        return np.array(s.get(k, [float("nan")] * len(s["time/total_timesteps"])), dtype=float)

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    axes = axes.ravel()
    panels = [
        ("curriculum/goal_dist", "goal distance curriculum (m)\nadvances only on >=50% wins", None),
        ("END", "how episodes end (fraction)\nsolid=goal  dashed=crash  dotted=incoherent", None),
        ("rollout/ep_len_mean", "episode length (steps)", None),
        ("curriculum/recent_success", "recent success (100-ep window)", (0, 1)),
        ("diag/throttle", "mean commanded throttle\n(ppo_240704 crept at 0.28)", (0, 1)),
        ("diag/halted", "halted-safely rate (per step)", None),
    ]
    for ax, (key, title, ylim) in zip(axes, panels):
        for j, s in data.items():
            x = get(s, "time/total_timesteps")
            x = x - x[0]                       # steps THIS run, warm arms included
            c, nm = colors[j], names.get(j, str(j))
            if key == "END":
                L = get(s, "rollout/ep_len_mean")
                goal = get(s, "reward/goal_bonus") * L / 1000.0
                crash = -get(s, "reward/crash") * L / 1000.0
                coh = -get(s, "reward/coherence_crash") * L / 100.0
                ax.plot(x, goal, "-", c=c, lw=1.6, label=nm)
                ax.plot(x, crash, "--", c=c, lw=1.0)
                ax.plot(x, coh, ":", c=c, lw=1.0)
            else:
                y = get(s, key)
                if np.all(np.isnan(y)):
                    continue
                ax.plot(x, y, "-", c=c, lw=1.6, label=nm)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("env steps this run")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper left", ncol=2)

    # ---- BY SCENE. Behaviour depends heavily on which scene is resident:
    # gnd_AU_180 has 47.7% of goals on grass and a 36.8% reachability ceiling,
    # gnd_AUw330 has 1% and 89.8%. A fleet-wide goal rate blends them and
    # arms that started an hour apart are on different scenes at the same
    # step. All four envs share ONE scene per rollout (vec_live_env rotates
    # it), so diag/scene_idx is a clean integer per dump and this is valid.
    def by_scene(s, key_fn):
        sc = np.rint(get(s, "diag/scene_idx")).astype(int)
        vals = key_fn(s)
        out = []
        for k in range(len(scene_names)):
            m = np.where(sc == k)[0][-args.recent:]
            out.append(float(np.nanmean(vals[m])) if len(m) else float("nan"))
        return np.array(out)

    goal_fn = lambda s: get(s, "reward/goal_bonus") * get(s, "rollout/ep_len_mean") / 10.0
    crash_fn = lambda s: -get(s, "reward/crash") * get(s, "rollout/ep_len_mean") / 10.0
    xs = np.arange(len(scene_names))
    narm = max(len(data), 1)
    for ax, fn, title in ((axes[6], goal_fn, "goal rate BY SCENE (%%), last %d dumps\nblack tick = reachability ceiling" % args.recent),
                          (axes[7], crash_fn, "crash rate BY SCENE (%%), last %d dumps" % args.recent)):
        for i, (j, s) in enumerate(data.items()):
            y = by_scene(s, fn)
            off = (i - (narm - 1) / 2) * (0.7 / narm)
            ax.plot(xs + off, y, "o", ms=5, c=colors[j], mec="k", mew=0.3,
                    label=names.get(j, str(j)))
        if fn is goal_fn:
            for k, c in enumerate(ceilings[:len(scene_names)]):
                ax.plot([k - 0.4, k + 0.4], [c, c], "-", c="k", lw=1.6)
        ax.set_xticks(xs)
        ax.set_xticklabels([n.replace("gnd_", "") for n in scene_names], fontsize=8)
        ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle(f"fleet of {len(data)} arms — {Path(args.out).stem}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"==> {args.out}")

    # ---- the latest row per arm, so the picture does not have to be squinted at
    print(f"\n{'job':>7} {'arm':<28} {'steps':>7} {'gdist':>6} {'succ':>5} "
          f"{'eplen':>6} {'goal%':>6} {'crash%':>7} {'incoh%':>7} {'thr':>5} {'halt':>6}")
    for j, s in data.items():
        L = get(s, "rollout/ep_len_mean")[-1]
        gb, cr, ch = (get(s, "reward/goal_bonus")[-1], get(s, "reward/crash")[-1],
                      get(s, "reward/coherence_crash")[-1])
        x = get(s, "time/total_timesteps"); x = x - x[0]
        print(f"{j:>7} {names.get(j, str(j))[:28]:<28} {int(x[-1]):>7} "
              f"{get(s, 'curriculum/goal_dist')[-1]:>6.1f} "
              f"{get(s, 'curriculum/recent_success')[-1]:>5.2f} {L:>6.1f} "
              f"{100 * gb * L / 1000:>6.1f} {100 * -cr * L / 1000:>7.1f} "
              f"{100 * -ch * L / 100:>7.1f} {get(s, 'diag/throttle')[-1]:>5.2f} "
              f"{get(s, 'diag/halted')[-1]:>6.3f}")


if __name__ == "__main__":
    main()
