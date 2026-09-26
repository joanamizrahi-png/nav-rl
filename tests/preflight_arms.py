#!/usr/bin/env python3
"""Run every arm's REAL launch line through the REAL argparse before submitting.

For each arm file: build the exact argument list the launcher would pass (via
PREFLIGHT=1, which reuses submit_arm's knob export and the launcher's knob->flag
logic), then run train_ppo_real.py --parse_only on it. Catches invalid choices,
unrecognized flags, missing values, bad types -- everything argparse rejects with
exit 2 -- plus a few things it does not: files named by TRAV/LIVECKPT that do not
exist, and knobs in the arm file that reach no flag at all.

    /users/jmizrahi/.conda/envs/neoverse/bin/python tests/preflight_arms.py memory_cnn_slowcur ...
    /users/jmizrahi/.conda/envs/neoverse/bin/python tests/preflight_arms.py --family
"""
import os, re, shlex, subprocess, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
FAMILY = ["memory_cnn_slowcur", "corner_ramped_cnn_slowcur", "memory_dino_slowcur",
          "trajectory_chunk5_mem_slowcur", "trajectory_chunk5_stack3_mem_slowcur",
          "memory_cnn_stack3s3_slowcur", "multiple_images_stack3_mem_slowcur",
          "memory_cnn_uniform", "memory_cnn_fixed10", "memory_cnn_slowcur_ent0"]


def launch_line(arm):
    """The launcher's argument line for this arm, produced the way submit_arm produces it."""
    env = dict(os.environ, PREFLIGHT="1", GPUS="4")
    # submit_arm.sh exports the knobs then calls sbatch; run the launcher body directly
    # with the same exports so the knob->flag logic is the real one.
    cmd = f'''set -a; . configs/arms/_base_campus.env; . configs/arms/{arm}.env; set +a;
              SLURM_JOB_ID=preflight bash scripts/slurm/train_ppo_real.sh'''
    r = subprocess.run(["bash", "-c", cmd], cwd=ROOT, env=env, capture_output=True, text=True)
    for line in r.stdout.splitlines():
        if line.startswith("PREFLIGHT_ARGS "):
            return line[len("PREFLIGHT_ARGS "):], r
    return None, r


def main():
    arms = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--family" in sys.argv or not arms:
        arms = FAMILY
    py = sys.executable
    bad = 0
    for arm in arms:
        if not (ROOT / "configs" / "arms" / f"{arm}.env").exists():
            print(f"FAIL  {arm}: no such arm file"); bad += 1; continue
        line, r = launch_line(arm)
        if line is None:
            print(f"FAIL  {arm}: launcher produced no argument line\n      {r.stderr.strip()[-400:]}"); bad += 1; continue
        argv = shlex.split(line)
        # files the job will open
        for flag in ("--trav_path", "--live_ckpt", "--clouds_dir", "--poses_dir", "--labels_dir"):
            if flag in argv:
                f = argv[argv.index(flag) + 1]
                if not os.path.exists(f) and not os.path.exists(ROOT / f):
                    print(f"FAIL  {arm}: {flag} {f} does not exist"); bad += 1
        # the real parser
        p = subprocess.run([py, "scripts/train_ppo_real.py", "--parse_only"] + argv,
                           cwd=ROOT, capture_output=True, text=True)
        if p.returncode != 0 or "PARSE_OK" not in p.stdout:
            err = [l for l in p.stderr.splitlines() if "error:" in l or "Traceback" in l or "Error" in l]
            print(f"FAIL  {arm}: {' | '.join(err[-2:]) or p.stderr.strip()[-300:]}"); bad += 1; continue
        # knobs set in the arm file that never became a flag (a typo the base still defines)
        arm_knobs = {l.split("=")[0] for l in (ROOT / "configs" / "arms" / f"{arm}.env").read_text().splitlines()
                     if "=" in l and not l.startswith("#") and l.split("=")[1].strip()}
        sh = (ROOT / "scripts" / "slurm" / "train_ppo_real.sh").read_text()
        unused = sorted(k for k in arm_knobs if k not in ("GPUS", "ENCODER", "IMGFIX") and f"${{{k}" not in sh and f"${k}" not in sh)
        if unused:
            print(f"WARN  {arm}: knobs the launcher never reads: {unused}")
        print(f"ok    {arm}  ({len(argv)} args)")
    print("\n" + ("PREFLIGHT FAILED" if bad else "PREFLIGHT OK: every arm parses and every named file exists"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
