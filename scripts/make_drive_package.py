"""Build the Drive upload package: a curated COPY of the results.

Why a copy instead of reorganizing outputs/: generated paths are hardcoded in
a dozen scripts and mirrored on the cluster — renaming them risks the exact
path-mismatch failures that have cost us the most this month. This script maps
messy-but-stable generated paths to the clean Drive layout in RESULTS_INDEX.md.
Rerun any time (idempotent; overwrites); missing artifacts are reported, not
fatal, so it works at any stage of the project.

Usage (Mac): python scripts/make_drive_package.py
Output:      "World Model/Drive_package/"  -> drag into Drive.
"""
from __future__ import annotations

import shutil
from glob import glob
from pathlib import Path

NAV = Path(__file__).resolve().parents[1]          # nav-rl/
# Since the 2026-08 repo move, nav-rl lives in ~/code, not inside World Model/
# — so WM can't be derived from the script location anymore. Fixed path (the
# results tree stayed in iCloud on purpose; only the git repos moved).
WM = Path.home() / ("Documents/Education/Stanford/Quarter 3/"
                    "CS399 Research/World Model")
DEST = WM / "Drive_package"

# (destination subdir, source glob relative to World Model/)
MAPPING = [
    ("01_world_model_validation", "inference_runs/replays/replay_v3_vs_original.mp4"),
    ("01_world_model_validation", "inference_runs/replays/EXPERT_REPLAY_reference.mp4"),
    ("01_world_model_validation", "nav-rl/outputs/projection_test/*.mp4"),
    ("01_world_model_validation/ground", "nav-rl/outputs/scene_clouds/ground/*"),
    ("02_geometric_boundary", "nav-rl/outputs/probe/probe_curves.png"),
    ("02_geometric_boundary/strips", "nav-rl/outputs/probe/*_strip.png"),
    ("02_geometric_boundary/diffusion_pairs", "nav-rl/outputs/probe/*_diffusion_pairs.png"),
    ("02_geometric_boundary/motion_videos", "nav-rl/outputs/probe/*_pair.mp4"),
    ("03_goal_setting", "nav-rl/outputs/goal_maps/goals.json"),
    ("03_goal_setting/photos", "nav-rl/outputs/goal_maps/*_goalphoto.png"),
    ("03_goal_setting", "nav-rl/outputs/goal_maps/*_frame_index_sheet.png"),
    ("04_rl_policy/eval_v4_success_videos", "inference_runs/replays/v4_success_ep*.mp4"),
    ("04_rl_policy/eval_v5_traverse", "nav-rl/outputs/eval_ppo_v5_traverse_trail00_ppo_200000_steps/*"),
    ("04_rl_policy/eval_v5_goal60_FAIL", "nav-rl/outputs/eval_ppo_v5_traverse_trail00_ppo_200000_steps_goal60/*"),
    ("04_rl_policy/eval_v6_peak_goal30", "nav-rl/outputs/eval_ppo_v6_randomgoal_trail00_ppo_306000_steps_goal30/*"),
    ("04_rl_policy/eval_v6_peak_goal60", "nav-rl/outputs/eval_ppo_v6_randomgoal_trail00_ppo_306000_steps_goal60/*"),
    ("04_rl_policy/reward_audit", "nav-rl/outputs/audit_*/*"),
    # 2026-08-03: v6d = 100% at all nine goal distances; zero-shot transfer
    ("04_rl_policy/v6d_evals", "nav-rl/outputs/eval_ppo_v6d_*"),
    ("05_semantics", "inference_runs/compare_v6_v7_semantic.mp4"),
    ("05_semantics", "inference_runs/v8_val5_rugdtrail/compare_v8val5_2x2.mp4"),
    ("05_semantics/sam2_segment_examples", "NeoVerse/outputs/sam2_segments/*/seg_overlay_000.png"),
    # 2026-08-03: v8 staged ladder at 5 epochs + palette legend
    ("05_semantics", "inference_runs/compare_v8_stages_1_2_3.mp4"),
    ("05_semantics", "inference_runs/v8_stage2_rugdtrail/compare_stage2_2x2.mp4"),
    ("05_semantics", "inference_runs/v8_stage3_rugdtrail/compare_stage3_2x2.mp4"),
    ("05_semantics", "inference_runs/class_palette_legend.png"),
    # 2026-08-05: first HONEST evaluation — v9 (14 classes) on held-out scenes
    ("05_semantics/heldout_v9", "inference_runs/HELDOUT_v9/*"),
    # 2026-08-13: the v10/v11 week — judgment videos (one-pass, reader,
    # ablations, anchoring), overlay rollouts, pose-direction probes
    ("05_semantics/v10_v11_verdicts", "inference_runs/V10_verdict/*"),
    ("04_rl_policy/rollouts_with_heading_overlay", "inference_runs/rl_rollouts/*"),
    ("06_diagnostics", "inference_runs/diagnostics/*"),
    # Root of the package: a plain README + the architecture/loss doc.
    ("", "nav-rl/diagrams/DRIVE_README.md"),
    ("", "nav-rl/ARCHITECTURE.md"),
    ("", "nav-rl/diagrams/Diffusion internals.png"),
    ("", "nav-rl/diagrams/RL gym.png"),
]



def _copy_with_retry(src, dst, tries=4):
    """iCloud times out mid-read on evicted files; each attempt forces more of
    the file warm, so retrying converges. Falls back to chunked read/write
    (survives where the fcopyfile fast path dies)."""
    import time
    for i in range(tries):
        try:
            with open(src, "rb") as f, open(dst, "wb") as g:
                shutil.copyfileobj(f, g, 1024 * 1024)
            shutil.copystat(src, dst)
            return
        except (TimeoutError, OSError) as e:
            if i == tries - 1:
                raise
            print(f"    retry {i+1} after {type(e).__name__}: {Path(src).name}")
            time.sleep(3)

def main():
    copied, missing = 0, []
    for sub, pattern in MAPPING:
        matches = sorted(glob(str(WM / pattern)))
        if not matches:
            missing.append(pattern)
            continue
        out = DEST / sub
        out.mkdir(parents=True, exist_ok=True)
        for m in matches:
            src = Path(m)
            if src.is_dir():
                # delete-then-copy: macOS compressed-flagged files refuse
                # in-place overwrite (EPERM on rebuild)
                if (out / src.name).exists():
                    shutil.rmtree(out / src.name)
                shutil.copytree(src, out / src.name, copy_function=_copy_with_retry)
            else:
                # sam2 overlays are all named seg_overlay_000 -> prefix by clip
                name = (f"{src.parent.name}_{src.name}"
                        if src.name == "seg_overlay_000.png" else src.name)
                if name == "DRIVE_README.md":
                    name = "README.md"
                (out / name).unlink(missing_ok=True)
                _copy_with_retry(src, out / name)
            copied += 1
    print(f"copied {copied} artifacts -> {DEST}")
    if missing:
        print(f"\nnot found yet ({len(missing)}) — fine mid-project, rerun later:")
        for p in missing:
            print(f"  {p}")


if __name__ == "__main__":
    main()
