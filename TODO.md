# Running TODO — Track B + Track A follow-ups

Snapshot as of 2026-07-14. Sorted by blocker + priority.
See `DECISIONS.md` for what's already decided and `OPTIONS.md` for known alternatives.

---

## 🕒 Blocked on Marlowe queue (jobs PD)

- [ ] **Job 380006** — v4 epoch-5 inference on rugd_trail_00. Verdict: does v4 produce structured semantic output? Blocks decision on whether v5 is needed.
- [ ] **Job 380016** — re-label 4 test clips with SAM3 priority-order fix. Blocks re-validation of the reward function on cleaner labels.

Both fire-and-forget. Check `squeue -u jmizrahi` occasionally. No tmux needed.

---

## ✅ Immediate next (do now, Mac-side, no cluster needed)

- [ ] **Reward-component ablation** — add weight args to `validate_reward.py`, run 4 configs (all-on, semantic-only, semantic+goal, semantic+collision), overlay curves on one plot. This is Jing's explicit "values of different components" ask.
- [ ] **Once 380016 completes**: SCP re-labeled npz files to Mac, re-run validation on the 4 clips, compare RGB+labels overlays before-vs-after to confirm the SAM3 fix worked.
- [ ] **Once 380006 completes**: SCP v4 inference outputs to Mac, eyeball semantic.mp4 — does the fix actually produce structured output (vs v3's palette noise)?

---

## Milestone A remaining (offline reward validation)

- [ ] Ablation runs (see above)
- [ ] Compare RUGD vs Cityscapes reward curves (different scene classes, does reward generalize?)
- [ ] Extract REAL trajectories from NeoVerse reconstructor per clip (write `extract_poses.py` + `slurm/extract_poses.sh`, sbatch, save poses.npz per clip)
- [ ] Re-run validation with `--pose_source npz` using real poses (removes the "robot walks forward in a straight line" assumption)
- [ ] Real camera intrinsics from reconstructor output (`rendered_intrinsics`) — bundled with poses.npz
- [ ] Test on more clips beyond the 4 (need to SCP + re-label; batch this)
- [ ] Write a short "reward validation report" for Jing meeting — plots + interpretation, points at what SAM3 gets wrong and how the reward tracks it

---

## Track A — deferred until v4 verdict (from 380006)

Two paths depending on v4 output quality:

**If v4 is good enough** (semantic MP4 shows recognizable structure):
- [ ] Skip v5, move on to Milestone B (world-model simulator)
- [ ] Log Track A as "done for now" in DECISIONS.md

**If v4 needs improvement**:
- [ ] Re-label ALL 46 training clips with SAM3 priority fix (bigger sbatch, ~50 min compute)
- [ ] Design v5 config — reuse v4 approach + cleaner labels + maybe more epochs / SEM_WEIGHT tweak / more data
- [ ] Launch v5 training (~10h wall clock)
- [ ] Test v5 inference, iterate

**Independent of v4 outcome** (should do eventually):
- [ ] Re-label all 46 clips with priority fix — needed for any v5+ training
- [ ] Consider hybrid: use RUGD's ground-truth 24-class labels for RUGD clips + SAM3 for Cityscapes. See OPTIONS.md D4 alternatives.

---

## Milestone B — world model as simulator (~weeks 5-7)

- [ ] Design `WorldBackend` interface: `step(pose, action) → (obs, next_gaussians)`
- [ ] Cache Gaussians once per episode, use rasterizer + diffusion at each step
- [ ] Wire NeoVerse's rasterizer + Track A's finetuned diffusion into the API
- [ ] Timestep / rollout length decisions (see OPTIONS.md #9)
- [ ] Test rollout on one scene end-to-end before RL wiring

---

## Milestone C — RL policy training (~weeks 8-9)

- [ ] Choose base pretrained Go2 walker (Unitree public? Lab-trained?)
- [ ] Wire SB3 PPO into `SceneEnv` (Gymnasium env in `src/env/scene_env.py`)
- [ ] Reward: switch from 2D-image (Milestone A code) to Gaussian-based (Milestone C code — the `reward_gaussian.py` stack I already wrote lives on ice for this)
- [ ] Rollout collection + training loop
- [ ] Wandb integration (probably already have via NeoVerse's wandb setup)
- [ ] First few training runs, tuning

---

## Milestone D — real Go2 deployment (~weeks 10-11)

- [ ] Pick platform (lab Go2W / Gitamini — depends on availability)
- [ ] On-robot inference pipeline (RGB camera → policy → gait controller)
- [ ] Safety fallback (stop on uncertain semantic)
- [ ] Real-world nav test in a controlled outdoor scene
- [ ] Failure analysis if it doesn't transfer

---

## Long-term / architectural

- [ ] **Full taxonomy fix for v5+**: reorder CLASSES so class IDs match priority order (would invalidate v3/v4 checkpoints but cleaner going forward). See OPTIONS.md #10.
- [ ] SAM3 prompt tuning — try tighter prompts like "shrubs and small bushes" instead of "vegetation" to reduce runaway. Complementary to priority fix.
- [ ] Consider RUGD 24-class labels as ground truth for RUGD-native clips (hybrid pipeline). See OPTIONS.md D4 / Track A hybrid decision.
- [ ] Ground plane detection fallback: verify whether NeoVerse's reconstructor outputs gravity-aligned frames (would simplify OPTIONS.md #5)
- [ ] Larger ablation with more reward components (clearance, stability) once we have real trajectories
- [ ] Ultimate Jing deliverable: paper draft

---

## Housekeeping / small cleanup

- [ ] Legend PNG is only 30 classes; add "collision threshold: 0.1" line to make the flag rule obvious
- [ ] Update OPTIONS.md when we resolve unknowns (e.g. once we confirm reconstructor coord frame convention)
- [ ] Delete or archive the legacy 2D reward in `src/env/reward.py` after Milestone C reward gets committed (or keep for ablation)
