#!/usr/bin/env python3
"""Every reward_source the env implements must be an accepted --reward_source choice.
2026-09-26: REWSRC=either was added to scene_env.py but not to the argparse choices,
and all ten fixed-family arms died in 6 s with exit 2 on a saturated cluster night."""
import ast, pathlib, re, sys
ROOT = pathlib.Path(__file__).resolve().parents[1]
env = (ROOT / "src" / "env" / "scene_env.py").read_text()
tr = (ROOT / "scripts" / "train_ppo_real.py").read_text()
implemented = set(re.findall(r'self\.cfg\.reward_source == "([a-z_]+)"', env)) | {"generated"}
m = re.search(r'add_argument\("--reward_source".*?choices=[\[(]([^\])]*)[\])]', tr, re.S)
choices = set(re.findall(r'"([a-z_]+)"', m.group(1))) | set(re.findall(r"'([a-z_]+)'", m.group(1)))
missing = sorted(implemented - choices)
print(f"env implements: {sorted(implemented)}\nargparse accepts: {sorted(choices)}")
if missing:
    print(f"FAIL: implemented but not accepted by --reward_source: {missing}"); sys.exit(1)
print("OK")
