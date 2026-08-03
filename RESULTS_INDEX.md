# Results index — Drive package

Suggested Drive layout below; each entry says what the artifact shows, how it
was produced (run + loss where relevant), and how to judge it. Local sources:
`nav-rl/outputs/` and `World Model/inference_runs/`.

```
Drive/
├── 01_world_model_validation/
├── 02_geometric_boundary/
├── 03_goal_setting/
├── 04_rl_policy/
├── 05_semantics/
└── 06_docs/
```

## 01_world_model_validation — "is the simulator real?"

| Artifact | What it shows / how to judge |
|---|---|
| `replay_v3_vs_original.mp4` (inference_runs/replays/) | The environment driven along the real robot's recorded path, side-by-side with the original video. Near-identical = the world model + calibration + rendering chain is faithful. Caught two real geometry bugs (handedness mirror, timestamp pinning) before passing. |
| `EXPERT_REPLAY_reference.mp4` | Same replay with the reward HUD: reward stays positive on the real path — the reward function endorses ground-truth behavior. |
| `outputs/projection_test/*.mp4` | "Can the next footstep be projected into the current image?" The robot's actual future position lands on the path it then walked (3+ scenes). |
| `outputs/scene_clouds/ground/` | Local ground = 40th percentile of Gaussian-splat heights within 0.75 m; chosen by a 31-scene sweep; ~2.7 cm mean error vs trajectory-derived truth. |
| `outputs/scene_clouds/html/`, `splats/` | Interactive 3D point clouds; `.splat` files open in https://superspl.at/editor. |

Metric note for everything: scale anchored to the RUGD platform camera height
(0.25 m per the paper); honest uncertainty ±20%.

## 02_geometric_boundary — "where can we trust it?" (full writeup: WORLD_MODEL_LIMITS.md)

| Artifact | What it shows |
|---|---|
| `outputs/probe/probe_curves.png` | Coverage (fraction of real pixels) vs rotation angle and vs lateral offset, all scenes. The forward-corridor result: front 81–100%, sides/back ~0%; lateral 100/80/67/46% at 0/0.5/1/2 m. |
| `outputs/probe/<scene>_spin_strip.png`, `_offset_strip.png` | The raster views behind those numbers, coverage printed per tile. |
| `outputs/probe/<scene>_diffusion_pairs.png` (4 scenes) | Top = rasterizer (holes visible), bottom = diffusion. Inside the corridor: cleanup. Outside: plausible-looking invention — why the reward never reads diffused content. |
| `outputs/probe/<scene>_{spin,slide,walk1m}_pair.mp4` | Moving-camera version (left raster, right diffused), one 81-frame diffusion pass each: shows the invention is temporally stable. slide = invention taking over as coverage drains; walk1m = deployment-relevant 1 m-offset traverse; spin = pure invention behind the robot. |

## 03_goal_setting — "where do goals go?"

| Artifact | What it shows |
|---|---|
| `outputs/goal_maps/goals.json` | Per-scene goal (frame, position) scored by walkability + splat density; bad-label scenes auto-flagged. |
| `outputs/goal_maps/*_goalphoto.png` | Approach-view photo with a true-scale 0.3 m ground ring projected at the goal + at-goal view. The ring doubles as a per-scene metric-scale eyeball check. |
| `outputs/goal_maps/*_frame_index_sheet.png` | Which trajectory frame is which (goal frames index this). |

## 04_rl_policy — the ladder (one change per rung)

Reward (v4 onward): terrain as COST (score−1 ≤ 0 per step, nothing to farm)
+ goal-progress shaping (w=1.5) − collision (1.0) − void (0.3) − step (0.05)
− spin (0.05) + one-time goal bonus +50. PPO (SB3), BC warm-start from real
trajectories. Full architecture: ARCHITECTURE.md.

| Artifact | What it shows |
|---|---|
| `outputs/eval_*/metrics.json` + `episode_*.mp4` | Standard exam per checkpoint: 20 episodes, success/steps/collisions + first-person videos with minimap HUD (white = path walked, green = goal). |
| v4 eval | 100% (20/20, 4.6 steps): fixed goal, near spawns. The reward-farming fix validated. |
| v5 eval | 100% (7.4 steps, ~0 collisions): full traverses from the trail start. |
| v5 @ goal60 | 0% — walks past the unseen-distance goal forever: memorized distance, not goal-following. Motivated random goals. |
| v6 sweep (9 evals) | Success vs goal distance for the random-goal policy: 0% at 1.5 m → 100% from ~4 m. Far-goal failure cured; close-range gap mapped; also PPO's late-training self-destruction documented (3 runs) → checkpoint-eval workflow. |
| `outputs/audit_*/reward_audit.png` | Per-step contribution of every reward term + share chart. Headline: terminal bonus ≈70% of all reward moved (by design); per-step economy is terrain-dominated (≈2:1 over goal shaping). Baseline for the conservative/aggressive knob. |

## 05_semantics — the diffusion co-generates labels

Training target = RUGD dense human GT (hint = SAM3 labels rasterized from the
Gaussians; non-RUGD clips fall back to SAM3 targets). Losses per version in
ARCHITECTURE.md §3.

| Artifact | What it shows |
|---|---|
| `inference_runs/compare_v6_v7_semantic.mp4` | v6 vs v7: dense GT targets cleaned regions and corrected SAM3's tree-as-building error; residual boundary speckle. |
| `inference_runs/inference_v8_val5_rugdtrail/compare_v8val5_2x2.mp4` | v8 stage 1 (x0-prediction), 5 epochs: RGB / holey hint / raw semantic / snapped. Speckle mechanism gone (smooth regions), hint's "building" error already overruled; remaining defect = low confidence (muddy raw → wrong snaps), which stages 2–3 target. RGB pane is single-pass diagnostic output — deployment uses the two-pass scheme. |
| `NeoVerse/outputs/sam2_segments/<clip>/seg_overlay_*.png` | SAM2 class-agnostic segments (homogeneity-loss preprocessing): segments should hug object boundaries. |

## 06_docs

`WORLD_MODEL_LIMITS.md` (boundary numbers, trust-signal design, caveats),
`ARCHITECTURE.md` (both systems + losses), `SEMANTIC_V8_RESEARCH.md` +
`SEMANTIC_ARCHITECTURE_DECISION.md` (literature + v8 design), diagrams
(`diagrams/architecture.drawio`, editable at app.diagrams.net).
