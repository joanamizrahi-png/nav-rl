"""Which scenes can the reward actually SEE? Read the geometry blocks and rank.

On 2026-09-02 three of five training scenes turned out to be scoring terrain
the camera could not see: the reward reads only the pixels inside the robot's
footprint, projected 1.5 m ahead, and on those scenes that rectangle landed
below the bottom edge of the frame. `reward/semantic` was a square wave between
the blind constant and real terrain, following the scene rotation, and `crash`
was zero because crashing was impossible where nothing could be scored.

`check_rewards.py` prints a CAMERA GEOMETRY block per scene. This collects them
and applies the two tests that matter:

  ABSOLUTE  blind zone = fy*h/(H-1-cy) must be under the look-ahead, with
            margin. This is the test that decides usability.

  RELATIVE  fy must sit near the MEDIAN OF ITS OWN DATASET. gnd_AUw240 came
            back at 532 while its five siblings were 316-338 -- same camera,
            same walk, 60% higher focal. It passes any absolute threshold you
            would have picked; it only looks wrong beside its siblings. That is
            a per-scene reconstruction failure and it is worth re-running,
            whereas the four sitex scenes at 584-608 are consistent to 4% with
            each other and are simply a narrower camera -- nothing to fix.

Usage:
    python scripts/certify_scenes.py /scratch/.../slurm-check-rew-4625*.out
    python scripts/certify_scenes.py --look_ahead 1.5 --list <logs>
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

SCENE = re.compile(r"(gnd_AU[A-Za-z0-9_]*|sitex_[A-Za-z0-9_]*|gtown\d*c\d*_[A-Za-z0-9_]*"
                   r"|go2w_[A-Za-z0-9_]*|rugd_[A-Za-z0-9_]*)")
FY = re.compile(r"fy\s+([\d.]+)\s+cy\s+([\d.]+)")
VFOV = re.compile(r"implied vertical FOV\s+([\d.]+)")
BLIND = re.compile(r"ground closer than\s+([\d.]+)\s*m")


def dataset_of(scene: str) -> str:
    """Group scenes by the rig that recorded them: focal is a property of the
    camera, so siblings are only comparable within a capture campaign."""
    for pre in ("gnd_AU", "sitex", "gtown", "go2w", "rugd"):
        if scene.startswith(pre):
            return pre
    return scene.split("_")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="slurm-check-rew-*.out files")
    ap.add_argument("--look_ahead", type=float, default=1.5,
                    help="the shaping footprint distance the run will use")
    ap.add_argument("--margin", type=float, default=0.15,
                    help="metres of clearance required beyond the blind zone")
    ap.add_argument("--sibling_tol", type=float, default=0.20,
                    help="flag fy more than this fraction from its dataset median")
    ap.add_argument("--list", action="store_true",
                    help="print only the passing scene names, space separated, "
                         "ready to paste into SCENES=")
    args = ap.parse_args()

    rows = {}
    for p in args.logs:
        try:
            txt = Path(p).read_text(errors="ignore")
        except OSError:
            continue
        if "CAMERA GEOMETRY" not in txt:
            continue
        m_s, m_f, m_b = SCENE.search(txt), FY.search(txt), BLIND.search(txt)
        if not (m_s and m_f and m_b):
            continue
        m_v = VFOV.search(txt)
        rows[m_s.group(1)] = dict(
            scene=m_s.group(1), fy=float(m_f.group(1)), cy=float(m_f.group(2)),
            vfov=float(m_v.group(1)) if m_v else float("nan"),
            blind=float(m_b.group(1)), log=Path(p).name)
    if not rows:
        raise SystemExit("no CAMERA GEOMETRY blocks found — run check_rewards "
                         "with the geometry print first")

    med = {}
    for ds in {dataset_of(s) for s in rows}:
        fys = [r["fy"] for r in rows.values() if dataset_of(r["scene"]) == ds]
        med[ds] = statistics.median(fys)

    passing = []
    for r in rows.values():
        ds = dataset_of(r["scene"])
        r["dataset"] = ds
        r["dev"] = (r["fy"] - med[ds]) / med[ds] if med[ds] else 0.0
        r["clears"] = r["blind"] + args.margin <= args.look_ahead
        r["sibling_ok"] = abs(r["dev"]) <= args.sibling_tol
        if r["clears"]:
            passing.append(r["scene"])

    if args.list:
        print(",".join(sorted(passing)))
        return

    print(f"look_ahead {args.look_ahead} m, margin {args.margin} m, "
          f"sibling tolerance {100 * args.sibling_tol:.0f}%")
    print(f"dataset focal medians: "
          + "  ".join(f"{k} {v:.0f}" for k, v in sorted(med.items())))
    hdr = (f"\n{'scene':<18}{'fy':>7}{'vFOV':>8}{'blind':>8}{'vs sibs':>9}"
           f"   verdict")
    print(hdr)
    print("-" * (len(hdr) + 14))
    for r in sorted(rows.values(), key=lambda r: (r["dataset"], r["blind"])):
        if not r["clears"] and not r["sibling_ok"]:
            verdict = "RE-RECONSTRUCT (focal off vs siblings)"
        elif not r["clears"]:
            verdict = f"too narrow — needs look_ahead >= {r['blind'] + args.margin:.1f} m"
        elif not r["sibling_ok"]:
            verdict = "usable, but focal disagrees with its siblings"
        else:
            verdict = f"OK ({args.look_ahead - r['blind']:.2f} m margin)"
        print(f"{r['scene']:<18}{r['fy']:7.0f}{r['vfov']:8.1f}{r['blind']:8.2f}"
              f"{100 * r['dev']:+8.0f}%   {verdict}")

    print(f"\n{len(passing)}/{len(rows)} scenes clear {args.look_ahead} m:")
    print(f"  SCENES={','.join(sorted(passing))}")
    bad_sib = [r['scene'] for r in rows.values() if not r['sibling_ok']]
    if bad_sib:
        print(f"\n  Focal disagrees with dataset siblings: {', '.join(bad_sib)}")
        print("  A whole dataset sitting at a different focal is a different "
              "CAMERA (fine).\n  One scene off its own siblings is a "
              "RECONSTRUCTION failure (re-run it).")


if __name__ == "__main__":
    main()
