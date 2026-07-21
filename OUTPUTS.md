# OUTPUTS.md — every artifact, what it shows, how to make it, how to view it

Written 2026-07-21. Viewing basics on the Mac:
- **.mp4 / .png** → double-click in Finder, or select + press SPACE (QuickLook), or `open <file>` in terminal
- **.html** → double-click (opens in browser); 3D views: drag = rotate, scroll = zoom
- **.npz / .csv** → data, not directly viewable; each has a script that renders it (listed below)
- Marlowe → Mac transfer is always: `scp 'jmizrahi@login.marlowe.stanford.edu:<remote path>' <local path>`

---

## 1. Expert replay — `replay_trail00.mp4`

**What it is:** the RL environment driven along the REAL robot trajectory — no policy.
The world model re-renders the original walk from the Gaussians; the banner shows the
reward the env would grant at each step; the corner map shows the path (white) on the
reference (gray) with the goal (green).

**What good looks like:** approximately the ORIGINAL clip (`data/rugd_trail_00.mp4`),
re-shot — same trail sliding toward you, speckly/holey but recognizable, reward mostly
positive. Judge it by playing it side-by-side with the original.

**Why it exists:** it's the env's lie detector. It caught the handedness bug (black
frames) and the timestamp bug (pan-then-black). If replay looks right, the whole
render/calibration/reward stack is verified.

**Make it (Marlowe):** `sbatch scripts/slurm/replay_trail00.sh` (~23 s)
**Fetch:** `scp '...:/scratch/m000204-pm06b/joana/outputs/replay_trail00.mp4' .`

**Local canonical copies** (`World Model/inference_runs/replays/`):
- `replay_v1_prefix_BROKEN.mp4` / `replay_v2_handedness_BROKEN.mp4` — kept as the
  debugging story (black frames; handedness bug, then timestamp bug)
- `replay_v3_timestamps.mp4` — the PASSING replay (2026-07-21)
- `replay_v3_vs_original.mp4` — side-by-side vs the original clip. THE demo video.

**Naming convention from now on:** downloads keep a version/date suffix
(`_v3`, `_200k`, `_<jobid>`), never a bare name that silently overwrites; broken
artifacts get `_BROKEN` instead of deletion (they document caught bugs).

## 2. Policy rollout — `outputs/ppo_real_trail00/rollout.mp4` (Marlowe)

**What it is:** same HUD/map format as the replay, but driven by the TRAINED policy.
**Judge it against the replay** (that's why they share a format): does the white line
track the gray? Does the view stay on the trail? Black regions = policy wandered off
the reconstructed volume (the reward is teaching it not to).

**Make it:** produced automatically at the end of `sbatch scripts/slurm/train_ppo_real.sh`
(~1.5 h for 200k steps). Training curves: wandb project `nav-rl` → `rollout/ep_rew_mean`
(should climb) and `ep_len_mean` (drops below 60 only if it ever reaches the goal).

## 3. Projection test (Test B) — `outputs/projection_test/<scene>_projection.mp4`

**What it is:** the next-step projection test. Each frame shows the robot's ACTUAL next position
(~2 m ahead, known because the robot drove it) projected into the current image as a
footprint: RED = flat-ground assumption (what the reward uses), YELLOW = true ground
height (from the trajectory). The banner prints the pixel offset between them.

**What it means:** rectangles landing on the path = "the next step projects well".
RED-vs-YELLOW gap = the error the flat-ground assumption causes (measured: median
7-11 px vs a ~100 px footprint on the three stage-1 scenes).

**Make it (Mac, no GPU):** `python3 scripts/test_projection.py [scene ...]`
(needs `data/<scene>.mp4`, `data/<scene>.npz`, `data/poses/<scene>_poses.npz` locally)

## 4. Gaussian cloud 3D — `outputs/scene_clouds/<scene>_cloud3d.html`

**What it is:** the world model itself — 120k of the scene's Gaussians in true color,
with the real trajectory as a white line. This is what the policy's observations are
rendered FROM and what the reward reads terrain FROM.

**Look for:** the trail as a coherent colored surface; trees as vertical structures;
the white line lying ON the surface (not floating/buried).

**Make it:** (a) Marlowe: `sbatch scripts/slurm/dump_clouds.sh` → npz per scene;
(b) scp npz to `outputs/scene_clouds/`; (c) Mac:
`python3 scripts/viz_scene_cloud.py outputs/scene_clouds/<scene>_cloud.npz`
(edit the scene list inside the slurm script to dump other scenes)

## 5. Ground-shape check (Test A) — `outputs/scene_clouds/<scene>_ground_testA.png`

**What it is:** made by the same viz script. Ground height along the real trajectory:
gray dashed = flat assumption; YELLOW = true profile (camera height minus 0.6 m —
ground truth along the driven line); BLUE = ground estimated from the Gaussians
(5th-percentile splat height within 0.75 m).

**What it means:** blue tracking yellow = the Gaussians know the ground shape →
we can query ground height ANYWHERE (not just on the driven line), which is what
the reward needs off-path. Known quirks: blue sits slightly LOW (percentile bias —
calibratable constant) and may lag; results so far: trail_00 median 8 cm,
park-2_00 (hilly) 18.5 cm — both beat the flat assumption.

## 6. Reward eval on real paths — `outputs/reward_eval/<scene>_realtraj/`

**What it is:** `reward_curves.png` (reward-vs-time, 4 components), `reward.csv`
(numbers), `reward_overlay.mp4` (RGB | labels+footprint | labels). Offline validation
of the reward on real trajectories with real poses. Dips on a path the robot actually
drove = SAM3 label errors or reward-design issues, localized per frame.

**Make it (Mac):** `python3 scripts/validate_reward.py --video data/<scene>.mp4
--labels data/<scene>.npz --pose_source npz --poses_npz data/poses/<scene>_poses.npz
--look_ahead_dist 2.0 --output_dir outputs/reward_eval/<scene>_realtraj --overlay`

## 7. Track A (semantic diffuser) artifacts — `World Model/inference_runs/`

- `compare_v6_v7_semantic.mp4` — 3x3 grid: rows v6-e20 / v7-e15 / v7-e30,
  columns hint | raw decode | snapped. The progression story.
- `hint_vs_gt_vs_v7.mp4` — SAM3 hint | RUGD ground truth | v7 output. Shows v7
  matches GROUND TRUTH, not the hint's errors (e.g., trees green, not "building" red).
- `outputs/legend.png` (nav-rl) — class id ↔ color ↔ traversability score. THE
  reference when any semantic color looks wrong.
- Per-run folders (`v7_e30_rugdtrail/` etc.): `rgb.mp4`, `holey_semantic.mp4` (hint),
  `semantic_raw.mp4` (pre-snap decode — judge quality here), `semantic.mp4` (snapped).

## 8. Diagnostics that already earned their keep

- `outputs/ground_flatness.png` — camera-height profiles for 3 scenes; how flat each
  scene really is (justifies flat-ground on trails, flags hilly parks).
- `outputs/pose_overlay/`, `outputs/pose_diag/` — earlier pose sanity artifacts.
- Marlowe logs: `/scratch/m000204-pm06b/joana/slurm-<jobname>-<jobid>.out|.err`;
  job status: `squeue -u jmizrahi`; job history: `sacct -u jmizrahi --starttime today`.
