# Working plan (from 2026-07-27) — guidelines + specifics

Ordered by dependency, not by day. Each item says WHY, WHAT exactly, and DONE-WHEN.

---

## A. Verify the training foundation (before any new training)

**Why:** run 390195 proved the risk — it executed a stale script and we nearly
credited a reward design that never ran. Verification is cheaper than repair.

- **A1. Confirm cluster poses are corrected-scale.**
  `python3 -c "import numpy as np; print(np.load('/scratch/m000204-pm06b/joana/outputs/poses/rugd_trail_00_poses.npz')['camera_height_m'])"`
  → must print 0.25. If 0.6: scp the Mac's `data/poses/*.npz` up first.
  *Done when: prints 0.25.*
- **A2. Adopt run bookkeeping.** Every sbatch: `git log --oneline -1` matches the
  intended commit; output dir contains a version tag; job name changed per
  experiment. (390195's output overwrote run 1's rollout — never again.)
- **A3. Re-launch the shaped config FOR REAL** (it has never run). This becomes
  the true "run 2". Read its curves before stacking more changes.
  *Done when: wandb run named ppo_shaped_v2_* exists with goal_frame=30 in its config.*

## B. Policy results (target: Thursday afternoon)

**Why:** first presentable RL results promised.

- **B1. Goal placement, verified visually.** Goals must be: on walkable terrain,
  in well-reconstructed space (Gaussian coverage), at controlled short distances
  (2-4 m first), not "the video's endpoint". Specifics: per-scene top-down maps
  showing trajectory + current goal + coverage-scored candidates; eyeball all
  scenes; encode chosen rule (e.g., "farthest trajectory point within 3 m whose
  footprint coverage > 90% and p40-ground class is traversable").
  *Done when: a goals.json (scene -> goal xy + spawn range) exists and the maps
  look right.*
- **B2. Demonstrations pipeline** (built, unlaunched): make_demos job → check
  action-saturation stats in its log → BC pretrain inside training script.
- **B3. Ladder of runs, one change at a time now:**
  shaped-v2 (A3) → v3 = v2 + terminal goal bonus + lower LR → v4 = v3 + BC init.
  Attribution preserved; Thursday shows the ladder.
- **B4. Evaluation protocol** (what "results" means): N=20 eval episodes on
  random spawn/goal pairs; report success rate, avg steps-to-goal, collisions;
  HUD rollout videos vs the expert replay. Same protocol for every rung.

## C. Geometric accuracy program (explicitly insisted on)

**Why:** the world model's trust boundary must be measured, not assumed.

- **C1. Spin + distance probe:** at trajectory points, render 360° sweeps AND
  lateral offsets (0.5/1/2 m off-path); for each view record raster void-%
  (= how much the diffusion must invent) and render the diffused pair.
  Deliverable: void-% vs angle/offset curves + side-by-side strips. This
  operationalizes "how far can we go from the source views".
- **C2. Runtime confidence signal:** per-view coverage (1 - void%) exposed in
  env info at every step → later usable to threshold actions (the "robot detects
  inaccuracy" idea) and to gate reward trust.
- **C3. (later) Held-out view synthesis:** reconstruct from half the frames,
  render at held-out poses, PSNR/SSIM vs real frames; raster vs diffused.

## D. Diffusion into the loop (the "full world model" question)

**Why:** observations currently come from the rasterizer only; the generative
half is not in the loop. Policy obs should eventually be diffused (deploy realism).

- D1. Mechanism decision (pose cache vs short-batch vs every-N-th) — a sit-down
  discussion, then build (the week after Thursday).
- D2. Raster-vs-diffused obs ablation once built (already a planned paper table).
- D3. Reward may need re-tuning under diffused obs — expect it, don't panic.

## E. Track A v8 (background; costs cluster-time, not Joana-time)

Per docs/SEMANTIC_V8_RESEARCH.md: v8a = x0-prediction + decoded-space CE
(fixes the diagnosed speckle mechanism), v8d = data expansion (more RUGD +
RELLIS-3D), later v8b analog bits, v8c SegSplat-style reward source.

## F. Communication

- One-liner to advisor re "offline RL data": demonstrations-warm-start vs full
  offline RL — which did he mean? (BC proceeds either way.)
- Thursday package = B4 results ladder + C1 probe figures + honest notes.
