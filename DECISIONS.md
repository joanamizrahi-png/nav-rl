# Track B — Design decisions log

Running log of decisions made during Milestone A (offline reward validation).
Each entry: what was decided, why, and what alternatives were rejected.

---

## Overall structure

### D1 — Build new `eval/` module instead of refactoring existing `src/env/reward.py`
- Existing `reward.py` is 2D-image-based (projects next foot to pixel, looks up SAM3 class at that pixel).
- Milestone A's design is 3D-Gaussian-based (query gaussians in a footprint volume, no camera projection).
- New `eval/` module keeps the two approaches independent. `reward.py` stays as legacy — potentially useful for later ablation ("2D-pixel reward vs 3D-Gaussian reward").
- Rejected: rewriting `reward.py` in place. Would lose the 2D baseline for comparison.

### D2 — Prototype on Mac with mock gaussians before running on real reconstructed scenes
- Real gaussians come from NeoVerse's reconstructor, which needs a GPU (Marlowe).
- To iterate fast on the reward function itself (which is pure geometry + label lookups), we don't need real gaussians — a hand-made synthetic scene works.
- Mock scene: a flat grass ground plane + a "tree" column at some offset + a "road" strip. Robot walks along the road; reward should be high there and low near/on the tree.
- Once the code works on mock data, we port to Marlowe and run on actual RUGD clip gaussians.
- Rejected: running from the start on Marlowe. Iteration cycle would be scp + srun + wait, slow.

---

## Reward function design

### D3 — Reward has 4 components (semantic + goal + collision + clearance). Skip stability and energy for now.
- Semantic traversability: dominant class in the footprint volume → traversability score.
- Goal progress: dot product of motion vector and direction toward goal.
- Collision: binary — any gaussian intersecting the robot's body volume.
- Clearance: max-z gaussian in the column above the robot's next position vs robot height.
- Stability (terrain roughness) skipped for now — largely captured by semantic (grass vs rocks).
- Energy skipped — overlaps with stability + goal progress, muddies the picture at this stage.

### D4 — Class-to-traversability lives in a yaml config, not hard-coded
- Stored at `config/traversability.yaml`.
- Class ID → score in [0, 1] (continuous, not binary — reflects Go2 preference).
- Each entry has a `note` field so future-you can see the reasoning.
- Rejected: hard-coded Python dict (harder to review + change; less clear to Jing).

### D5 — Footprint is a rectangle (robot body), not a small cylinder (single foot)
- Per Joana's ask: "not feet, just where the robot lands in general."
- Rectangle: 0.3m × 0.6m (approximate Go2 body footprint), aligned to robot's heading.
- Positioned centered under the robot's next position at the ground plane height.
- Rejected: circular footprint (simpler math but ignores that robot is rectangular). Rejected: single-foot cylinder (over-narrow query region).

### D6 — Ground plane detected via RANSAC fit on gaussian positions
- Fit a plane to the lowest 30% of gaussians by z-coordinate (rough heuristic for "these are ground").
- RANSAC handles outliers (e.g. gaussians on the underside of overhangs).
- Alternative: assume NeoVerse's reconstructor outputs gravity-aligned frames (z-axis = up). Need to verify — if true, we can skip RANSAC and just use z = min z of local neighborhood.
- For now: implement RANSAC as fallback; will simplify later if we confirm z-alignment.

---

## Dataset

### D7 — Start on RUGD + Cityscapes (already SAM3-labeled) instead of NCLT/GND
- NCLT and GND are Jing's suggestions and worth doing later, but each needs ~1-2 days of labeling prep.
- RUGD + Cityscapes are ready today; code we write is dataset-agnostic.
- Show Jing preliminary results on RUGD next week; port to NCLT the week after if there's time.
- Caveat: need to verify RUGD ships ground-truth trajectory data. If not, Cityscapes ego-motion via SLAM as fallback, or defer to NCLT.

---

## Coding decisions (accumulating as we build)

### D8 — Milestone A uses 2D-image reward, NOT Gaussian reward (corrected)
- Initial today: I started coding a Gaussian-based reward.
- Joana caught: "why Gaussians and not just 2D image? and it's not blocking because we can always run SAM3 on RGB."
- Correct answer: for Milestone A (offline validation on recorded clips), 2D is simpler because:
  - RUGD + Cityscapes SAM3 labels are already available on Marlowe
  - No need to run the reconstructor (Marlowe GPU) to get Gaussians
  - Faster iteration + faster Jing deliverable
- Long-term (Milestone C — RL training with world model): switch to Gaussian reward for privileged 3D info
- Gaussian code written today is NOT deleted, just unused for Milestone A. Lives in `src/eval/reward_gaussian.py`, `ground_plane.py`, `footprint.py`, `mock_data.py`. Seed for Milestone C.

### D9 — Policy observation stays RGB + goal vector (no semantic in observation)
- Reconfirms the original 2026-07-03 decision.
- Semantic is used only in REWARD (training-only). Real robot at deploy sees RGB + goal, same as training.
- Track A's semantic finetune contributes via reward path, not observation path.

### D10 — Reward has 3 components for Milestone A: semantic + goal + collision
- Semantic: mean traversability score over projected footprint pixels
- Goal: signed distance closed to goal since previous step
- Collision: fraction of footprint pixels on non-traversable classes (score <= 0.1). Weighted 5x.
- Deferred: clearance (needs depth or Gaussians), stability, energy.

### D11 — Footprint = 0.3m × 0.6m rectangle, projected via camera geometry
- Robot's next body position = 0.5m ahead of current position along heading
- Rectangle in world coords, 4 corners projected onto semantic image via K + w2c
- Pixels inside the projected polygon = the region we score
- Uses PIL for polygon rasterization (no scipy dep)

### D12 — Trajectory data for Milestone A: synthetic linear forward motion for now
- RUGD doesn't ship ground-truth robot trajectories.
- Options: extract via COLMAP (slow), get from NeoVerse's reconstructor (needs Marlowe GPU run per clip), synthesize (fast, unrealistic).
- For a first pass: synthetic linear forward motion (0.3m/frame along +x). Gives us a reward-vs-time plot to eyeball the pipeline.
- **Next step after first plot**: extract real trajectories from NeoVerse's `rendered_extrinsics` (one-time Marlowe run per clip, save poses.npz, scp to Mac).
- `load_clip()` already supports both via `pose_source="synthetic"` or `pose_source="npz"`.

### D13 — Class-to-traversability lives in yaml, not hard-coded (already committed above at D4)

### D14 — OPTIONS.md is our forward-looking catalog for alternatives
- 9 design axes with alternatives + cost-to-switch documented
- Rule: when we hit a wall, look here first; when we make a new decision, update both DECISIONS.md and OPTIONS.md
