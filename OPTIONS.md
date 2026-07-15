# Track B — Design options catalog

Design axes with their current pick + alternatives to consider if we want to
iterate. Complement to `DECISIONS.md` (which records what we chose today);
this file records **what we could switch to later** and how much work it'd be.

---

## 1. Reward representation

**Current pick**: 2D-image-based reward (Option A) using SAM3 labels for Milestone A.
Long-term (Milestone C): switch to Gaussian-based reward (Option C).

| Option | What it does | Pros | Cons | Cost to switch |
|---|---|---|---|---|
| **A. 2D image** (current) | Project robot's next step onto 2D semantic image, look up class | Fast, uses SAM3 labels directly, no reconstruction needed | Depends on Track A's semantic quality; ignores 3D info | — |
| **B. 2.5D from depth** | Unproject 2D semantic + 2D depth to a BEV grid | Adds 3D info without needing Gaussians | Only as good as diffusion depth; only sees current view | ~1 week — needs depth-to-BEV unprojection code |
| **C. 3D Gaussians directly** | Query gaussians in footprint volume; take dominant class | Geometrically accurate; independent of Track A quality | Needs reconstructor to have run; only works in sim (privileged) | Partially built — see `src/eval/reward_gaussian.py` etc. ~1 day to plug in |
| **D. BEV of Gaussians** | Top-down rasterize the labeled Gaussians into a 2D grid | Best of C + BEV convenience for path planning | Extra rasterization pass; only in sim | ~2 days once Gaussians are on hand |

**When to switch:**
- To B: if we ever need depth-aware reward AND Track A's depth output is reliable AND we don't want to run the reconstructor at reward-eval time
- To C: for RL training loop when we control the sim (Milestone C)
- To D: if we also want to give the policy an occupancy map or path plan on top of a grid

---

## 2. Observation representation

**Current pick**: RGB + goal vector only. Same at train and deploy.

| Option | Content | Pros | Cons | Cost to switch |
|---|---|---|---|---|
| **RGB + goal** (current) | Raw camera image + goal direction/distance | Zero sim-to-real gap in observation format | Policy must learn to interpret RGB from scratch | — |
| **RGB + semantic + goal** | Add per-pixel class map | Richer input, may learn faster | Semantic must be produced at deploy (SAM3 slow, or distilled model needed) | ~few days — add semantic to obs space, ensure deploy inference pipeline can produce it |
| **RGB + depth + goal** | Add per-pixel depth | Better geometry for the policy | Depth must be produced at deploy (monodepth model, or Track A depth) | ~1 week — same story as semantic |
| **RGB + semantic + depth + goal** | Everything | Maximum info | Most complex to align train ↔ deploy | ~2 weeks |
| **BEV occupancy grid + goal** | Top-down grid of what's around the robot | Simpler for RL, standard in AV | Needs BEV construction at deploy (either from LiDAR or unproject-from-camera) | ~2 weeks |

**When to switch:**
- Add semantic to obs: if the policy struggles to learn from RGB alone
- Add depth: same reason, plus better collision avoidance
- BEV obs: if we go multi-step planning or lidar-based nav

---

## 3. Reward decomposition (which components)

**Current pick**: 3 components for Milestone A: **semantic traversability + goal progress + collision**. Add clearance for Milestone C.

| Component | What it captures | Currently included? |
|---|---|---|
| Semantic traversability | Is the class of terrain walkable? (grass yes, water no) | ✓ |
| Goal progress | Did the robot get closer to the goal? | ✓ |
| Collision | Did the robot hit something solid? | ✓ (2D approximation via non-traversable classes in footprint) |
| Clearance | Is there vertical headroom (no branches)? | Deferred (needs depth/Gaussians) |
| Stability | Is the terrain flat enough to walk on stably? | Deferred |
| Energy | Is this an efficient motion? | Deferred |
| Time penalty | Small per-step cost to encourage speed | Optional; add if policy dawdles |
| Smoothness | Penalize jerky actions | Optional; add if policy is jittery |
| Exploration | Encourage novel positions early in training | Optional; not usually needed for nav |

**When to add:**
- Clearance: as soon as we have depth or Gaussians (Milestone C)
- Stability: if the policy learns "walk fast on rocky terrain and fall down"
- Time penalty: if trajectories converge to correct-but-slow

---

## 4. Class-to-traversability mapping

**Current pick**: continuous scores in [0, 1] per class (see `config/traversability.yaml`).

| Option | How it works | Pros | Cons |
|---|---|---|---|
| **Continuous scores** (current) | Per-class float in [0, 1]; reward is weighted mean | Nuanced; can express "mud is walkable but bad" | Hand-tuned; scores are guesses |
| **Binary traversable/not** | Per-class bool | Simplest | Ignores that some classes are conditionally walkable |
| **Learned from robot data** | Fit per-class score from where recorded robots actually walked | Data-driven | Needs a lot of recorded expert trajectories with GT labels |
| **Per-robot** | Different scores for Go2 vs Jackal vs Husky | Handles multi-robot benchmarks (GND) | Multiplies config surface area |

**When to switch:**
- Binary: if hand-tuning continuous scores turns into bikeshedding
- Learned: if we ever have enough data to fit them properly
- Per-robot: if we run on GND multi-robot benchmark

---

## 5. Ground plane detection

**Current pick (for Gaussian reward)**: RANSAC fit against the lowest 30% of gaussians.
For 2D reward: not needed at all.

| Option | How | Pros | Cons |
|---|---|---|---|
| **RANSAC on low-z gaussians** (current) | Fit plane to lowest 30% | Robust to outliers | Slow (~100ms), needs threshold tuning |
| **Assume z-axis is up** | If NeoVerse's reconstructor uses gravity-aligned frame, just take z-min | Instant | Only works if the coord frame is gravity-aligned (unverified) |
| **Hough transform / accumulator** | Vote for plane parameters in a grid | Robust in cluttered scenes | Overkill for outdoor |
| **Assume horizontal** | Just use z=0 in world | Simplest | Wrong for tilted scenes |

**When to switch:**
- Assume z-axis is up: verify NeoVerse's coord convention; if yes, delete RANSAC code
- Assume horizontal: only if we ever restrict to perfectly-level scenes

---

## 6. Footprint shape

**Current pick**: 0.3m × 0.6m rectangle (approximate Go2 body).

| Option | Shape | Pros | Cons |
|---|---|---|---|
| **Rectangle** (current) | Aligned to robot heading | Matches robot body | Slightly more math than circle |
| **Circle** | Radius ~0.15m centered under robot | Simplest math (no heading needed) | Ignores that robot is longer than wide |
| **Foot placements** | 4 small circles where feet will land | Most physically accurate for a quadruped | Requires knowing gait phase |

**When to switch:**
- Circle: if heading estimation is noisy/expensive
- Foot placements: if we want fine-grained gait-level reward

---

## 7. Dataset for reward validation

**Current pick**: RUGD + Cityscapes (SAM3-labeled, already on hand).

| Dataset | Robot | Environment | Labels | Trajectory available? |
|---|---|---|---|---|
| **RUGD** (current) | Jackal | Off-road trails | SAM3 done ✓ | Need to verify |
| **Cityscapes** (current) | Car (dashcam) | Urban | SAM3 done ✓ | Vehicle odometry available |
| **NCLT** | Segway | Campus, seasonal | Need SAM3 | ✓ high-quality GPS+IMU |
| **GND** | Multi (Jackal, Husky, Spot) | Outdoor | Has semantic labels | ✓ |
| **RELLIS-3D** | Warthog | Off-road | Has semantic labels | ✓ |

**When to switch:**
- NCLT: to test cross-season robustness or use pre-existing high-quality trajectory data
- GND: for multi-robot / cross-embodiment story that Jing hinted at
- RELLIS-3D: for larger off-road dataset if RUGD is too small

---

## 8. Action space

**Current pick**: continuous `(v_forward, omega_yaw)` in Box(-1, 1, (2,)).

| Option | Content | Pros | Cons |
|---|---|---|---|
| **v_forward + omega_yaw** (current) | 2D continuous | Simple; matches high-level nav | Ignores lateral motion (strafe) |
| **v_forward + v_lateral + omega_yaw** | 3D continuous | Full holonomic control | Larger action space for policy to explore |
| **Waypoint output** | Policy outputs next (x, y) target | Higher-level, more sample-efficient | Requires downstream controller |
| **Discrete grid** | {forward, left, right, stop} | Trivially explorable | Coarse; not useful for real robot |
| **Joint torques** | Direct low-level control | Total control | Way outside our scope; walker handles this |

**When to switch:**
- Add lateral: if the robot needs to sidestep obstacles frequently
- Waypoint: if per-step control is too fine-grained for our sim step size

---

## 9. World model rollout format (Milestone B)

**Current pick**: reconstruct Gaussians once at episode start; use rasterizer + diffusion to render at each step.

| Option | How | Pros | Cons |
|---|---|---|---|
| **Reconstruct once, rasterize+diffuse per step** (current) | Fast rasterization, one expensive reconstruction | Simple; sim is deterministic | Static world; can't handle dynamic obstacles |
| **Reconstruct per step** | Rebuild Gaussians every step | Handles dynamic scenes | 60x slower; blows up RL budget |
| **Reconstruct chunks** | Rebuild every N steps | Middle ground | Complex; unclear payoff |
| **No reconstructor, direct diffusion** | Diffuse frame-to-frame conditioned on action | Purely 2D | Weaker spatial consistency; not what NeoVerse is built for |

**When to switch:**
- Per-step reconstruction: if we tackle dynamic obstacles in Stage 2 (later)

---

## 10. Taxonomy priority order (SAM3 later-wins rule)

**Current pick**: 30-class order defined in `sam3_precompute_labels.CLASSES` and `diffsynth/utils/semantics.CLASS_COLORS`. Under SAM3's "later wins on overlap" rule, class ID = priority.

**Known issues (2026-07-14, discovered during reward validation)**:
- **Vegetation (20) is too late** — as the general "leafy/plant" catch-all, it currently overrides everything before it, including things it should lose to. Observed: park concrete surfaces mislabeled as vegetation; wooded scenes under-report tree pixels (tree=19 loses to vegetation=20).
- Vegetation should come BEFORE: tree, grass, rock, building, wall, water, dirt, mud, log — basically all specific classes that can visually overlap with plant matter.
- Ideal placement: around position 2-3 (right after void and sky), so specific classes overwrite it.

**Cost to fix**: HIGH. Class IDs are hardcoded in:
- `sam3_precompute_labels.CLASSES` (order of prompts)
- `diffsynth/utils/semantics.CLASS_COLORS` (per-ID palette)
- All existing SAM3 label `.npz` files (integer class IDs baked in)
- All finetuned semantic-diffusion checkpoints (v3, v4, ...) trained on current IDs
- `nav-rl/config/traversability.yaml`, `nav-rl/src/eval/palette.py`

Reordering means: re-label every clip, re-train from scratch (checkpoints become mis-mapped). ~1 week end-to-end.

**Cheap workaround for the reward function only**: lower vegetation's traversability score below the collision threshold (0.2 → 0.05) so wooded scenes trigger collision penalties even if the "vegetation" label absorbs tree pixels. Config-only change, no re-labeling.

**When to fix properly**: after v4/v5 model verdict; not worth doing until we know Track A works end-to-end.

---

## How to use this doc

- When we hit a wall, come here first — the alternative might be documented already
- When Jing suggests something, cross-reference here to see if it fits an existing axis
- When we make a new decision, add it to `DECISIONS.md` AND update this doc if it's an option-switch
