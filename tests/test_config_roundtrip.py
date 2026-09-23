"""The training/eval seam, checked without a GPU (2026-09-22, Joana: "why do we get so many of
these bugs that are things that have worked previously?").

Nearly every silent-wrong-number bug this month lived in the same place: train_ppo_real writes
env_config.json, eval_policy reads it back and rebuilds the environment, and nothing checked that
the two agree. Instances: collision_look_ahead_m adopted onto an attribute build_env never read
(09-04, far-box evals); trav_path recorded but never adopted (09-20, every eval scored grass as
walkable); --goal_frame overridden by the adopted frame range (09-21, every corner test used random
short goals); goal_dist_range recorded as the STRING "2,4" and joined character-by-character into
"2,,,4" (09-22); goal_dist never recorded at all, so eval fell into a different sampling branch
than training used (09-22, the 13.8 m goals).

This test reads both scripts as text and compares three sets:
  1. keys eval ADOPTS from env_config  vs  keys training WRITES        -> adopted-but-never-written
  2. keys training writes              vs  keys eval adopts            -> written-but-ignored (informational)
  3. every adopted key is also passed into build_env's cfg             -> adopted-into-nothing
It needs no torch, no SB3 and no cluster.

    python tests/test_config_roundtrip.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = open(os.path.join(ROOT, "scripts", "train_ppo_real.py")).read()
EVAL = open(os.path.join(ROOT, "scripts", "eval_policy.py")).read()

# Keys eval deliberately does NOT take from training, each with the reason it is excluded.
INTENTIONAL = {
    "mirror_prob": "evaluated in the real world unless passed explicitly",
    "scenes": "the eval names its own scene",
    "scene": "the eval names its own scene",
    "warmstart": "provenance, not behaviour",
    "obs_frame_stack": "adopted; listed here only if the writer lags the reader",
}


def written_keys() -> set:
    """Keys train_ppo_real puts into env_config.json: the dict literal passed to json.dumps
    right after the env_config.json path is built."""
    i = TRAIN.index('"env_config.json"')
    j = TRAIN.index("write_text(_json.dumps({", i)
    depth, k = 0, j + len("write_text(_json.dumps(")
    for k in range(k, len(TRAIN)):
        if TRAIN[k] == "{": depth += 1
        elif TRAIN[k] == "}":
            depth -= 1
            if depth == 0: break
    return set(re.findall(r'^\s*"([a-z0-9_]+)":', TRAIN[j:k], re.M))


def adopted_keys() -> set:
    """Keys eval_policy copies out of env_config (the tuple it iterates over)."""
    m = re.search(r'for _k in \((.*?)\):', EVAL, re.S)
    assert m, "could not find the adoption list in eval_policy.py"
    keys = set(re.findall(r'"([a-z0-9_]+)"', m.group(1)))
    # several keys are adopted in their own block instead of the loop (trav_path, the goal
    # sampling fields): count any read of the training dict as an adoption
    keys |= set(re.findall(r'_tr\.get\("([a-z0-9_]+)"', EVAL))
    keys |= set(re.findall(r'_tr\["([a-z0-9_]+)"\]', EVAL))
    return keys


# Keys that MUST survive the round trip because they change what the policy does or what the
# reward pays. Each was added after a bug where it did not. Listed by name so a new knob that
# matters has to be added here deliberately rather than forgotten.
MUST_ROUNDTRIP = [
    ("step_size_m",           "kinematics: eval ran 0.25/0.3 against training 0.30/0.50 (09-01)"),
    ("yaw_step_rad",          "kinematics"),
    ("forward_only",          "reverse re-enabled in eval hid a backing gait (09-01)"),
    ("trav_path",             "eval scored grass as walkable for a whole day (09-20)"),
    ("reward_source",         "generated labels vs the fused map decide every crash"),
    ("collision_threshold",   "what counts as a collision"),
    ("collision_terminate_frac", "when an episode ends"),
    ("goal_radius",           "what counts as arrival"),
    ("goal_dist",             "the target distance; without it eval draws goals a different way (09-22)"),
    ("goal_dist_range",       "the band the target is drawn from"),
    ("render_window",         "fusion window: a different window is a different world"),
    ("coverage_window",       "the coverage the gate reads"),
    ("sem_palette",           "label ids must mean the same thing"),
    ("obs_frame_stack",       "how many views the policy is handed"),
    ("collision_box_memory",  "the memory arm's whole mechanism"),
    ("terrain_speed_scaled",  "whether the terrain cost is scaled by speed"),
    ("timeout_distance_scaled", "how a timeout is priced"),
]


def build_env_reads() -> set:
    """Keys build_env actually threads into the backend/env config.

    2026-09-23: also follows helper functions build_env CALLS. max_steps moved
    behind _eval_step_budget(args) and the flat text scan reported it as going
    nowhere, which would have masked a real break next time.
    """
    b = EVAL[EVAL.index("def build_env"):EVAL.index("def main")]
    for helper in re.findall(r"\b(_[a-z][a-z0-9_]*)\(args", b):
        m = re.search(rf"^def {helper}\(.*?(?=\n(?:def |class )|\Z)", EVAL, re.S | re.M)
        if m:
            b += m.group(0)
    used = set(re.findall(r'getattr\(args, "([a-z0-9_]+)"', b)) | set(re.findall(r'\bargs\.([a-z0-9_]+)', b))
    # eval renames a few on the way in (_dest): the value lands on a different attribute name
    m = re.search(r'_dest = \{(.*?)\}', EVAL, re.S)
    if m:
        for src, dst in re.findall(r'"([a-z0-9_]+)":\s*"([a-z0-9_]+)"', m.group(1)):
            if dst in used: used.add(src)
    return used


def main() -> int:
    written, adopted, used = written_keys(), adopted_keys(), build_env_reads()
    assert len(written) > 30, f"only found {len(written)} written keys; the parser needs updating"
    assert len(adopted) > 30, f"only found {len(adopted)} adopted keys; the parser needs updating"
    bad = sorted(adopted - written - set(INTENTIONAL))
    dead = sorted(k for k in adopted if k not in used and k not in INTENTIONAL)
    ignored = sorted(written - adopted - set(INTENTIONAL))
    print(f"== config round-trip: {len(written)} keys written, {len(adopted)} adopted, {len(used)} read by build_env")
    if bad:
        print("\n   ADOPTED BUT NEVER WRITTEN -- eval silently keeps its own default:")
        for k in bad: print(f"     {k}")
    if dead:
        print("\n   ADOPTED BUT NOT READ BY build_env -- the value goes nowhere:")
        for k in dead: print(f"     {k}")
    if ignored:
        print(f"\n   written but not adopted ({len(ignored)}, informational -- check any that change behaviour):")
        print("     " + ", ".join(ignored))
    missing = [(k, why) for k, why in MUST_ROUNDTRIP if k not in written or k not in adopted]
    if missing:
        print("\n   BEHAVIOUR-CRITICAL KEYS THAT DO NOT ROUND-TRIP:")
        for k, why in missing:
            where = []
            if k not in written: where.append("not written by training")
            if k not in adopted: where.append("not adopted by eval")
            print(f"     {k:<26} {' and '.join(where)}\n       -> {why}")
    if bad or dead or missing:
        print("\nFAIL")
        return 1
    print("\nOK: every adopted key is written by training, reaches the built env, "
          "and every behaviour-critical key round-trips")
    return 0


if __name__ == "__main__":
    sys.exit(main())
