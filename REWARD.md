# Reward function — what it is, how components are chosen, how they're computed

Reference for Track B's reward function as it exists in Milestone A (offline
validation on 2D semantic images). Living document.

---

## 0. What is the reward and what shape does it have?

At every RL timestep, the robot is at some pose and takes an action. The
**reward function** returns a scalar number saying "how good was that step?"
Positive = good. Negative = bad. The RL policy learns to maximize the sum
of rewards over an episode.

Ours is a **decomposed reward** — it's a weighted sum of separate components:

```
R(state, action) = w_sem  · R_semantic(state)
                 + w_goal · R_goal(state, prev_state)
                 - w_col  · R_collision(state)
```

We keep the components separated (rather than one blob number) because:

1. **Interpretability**: we can plot each component over time and see WHY a state got a specific reward. Jing explicitly asked for this.
2. **Ablatability**: we can zero out each weight one-at-a-time to see what carries the signal.
3. **Tuning**: if the policy learns weird behavior later, we can trace it to a specific component and fix that one.

---

## 1. Semantic traversability — `R_semantic`

### What it captures
"Is the class of terrain under the robot's next body position walkable for a Go2?"

### How it's chosen
Each of the 30 SAM3 classes has a **hand-picked traversability score in [0, 1]** stored in `config/traversability.yaml`. 1.0 = ideal walkable surface. 0.0 = do not step. Examples:
- grass = 0.95 (ideal)
- sidewalk = 0.95 (designed for pedestrians)
- dirt = 0.9 (primary off-road surface)
- gravel = 0.8 (loose, some slip risk)
- road = 0.6 (paved but cars use it — pedestrian robot should prefer sidewalk)
- rock = 0.4 (uneven, tripping risk)
- mud = 0.3 (slippery, not lethal)
- vegetation = 0.2 (bushes, might push through)
- water = 0.0 (do not step)
- tree, building, person = 0.0

Scores are Go2-specific guesses right now. Notes in the yaml flag which ones need verification on the real robot (sand, mulch, stairs).

### How it's computed

1. Compute a **rectangle in the world frame** where the robot's body will be after this step. Currently 0.3m × 0.6m centered `look_ahead_dist` (default 1.5m) ahead of the current position.
2. **Project the four corners** of that rectangle onto the current semantic image using the camera intrinsics `K` and world-to-camera pose `w2c`.
3. **Rasterize the polygon** to a boolean mask of pixels inside it (using PIL).
4. Look up the **traversability score** for every pixel-class in that mask.
5. `R_semantic = mean of those scores.`

Result: a number in [0, 1] that represents "how walkable, on average, the region under the robot's next position is."

**Example**: polygon covers 5000 pixels — 4000 dirt (0.9), 800 vegetation (0.2), 200 tree (0.0). Mean = (4000·0.9 + 800·0.2 + 200·0.0) / 5000 = **0.752**.

**Code**: `src/eval/reward_2d.py:compute_reward` step 1.

---

## 2. Goal progress — `R_goal`

### What it captures
"Did the robot get closer to the goal during this step than it was last step?"

### How it's chosen
Simplest possible goal-tracking signal: positive when the robot closes distance, negative when it moves away.

### How it's computed

```
R_goal = ||prev_position - goal||  -  ||current_position - goal||
```

Both distances are in world coords. On the first frame (no previous position), `R_goal = 0`.

**Example**: robot was 5.0m from goal last frame, is 4.7m now. `R_goal = 5.0 - 4.7 = +0.3`.

**Caveat**: with our current SYNTHETIC trajectory (robot walks 0.3m forward every frame), this is a constant +0.3 per frame. It becomes informative only once we plug in real reconstructor-derived trajectories.

**Code**: `src/eval/reward_2d.py:compute_reward` step 2.

---

## 3. Collision — `R_collision`

### What it captures
"What fraction of the robot's next-body-position is on classes that would physically block it?"

### How it's chosen
Classes with traversability score ≤ 0.1 count as "collision-worthy" (a threshold `--collision_threshold`, default 0.1). Currently 15/30 classes qualify:
- void, sky, water, tree, log, building, wall, fence
- pole, traffic_sign, traffic_light
- vehicle, motorcycle, bicycle, person

### How it's computed

1. Same polygon as `R_semantic` (projected robot footprint).
2. Count what fraction of those pixels have collision-worthy classes:
   ```
   collision_frac = (# pixels with score ≤ 0.1) / (total pixels in polygon)
   ```
3. `R_collision = collision_frac` (a number in [0, 1]).

**In the final reward, it's SUBTRACTED with a big weight** (`-w_col · R_collision`), so it's a punishment.

**Example**: polygon covers 5000 pixels — 500 of them are tree. `R_collision = 500/5000 = 0.1`. With `w_col = 5.0`, this contributes `-0.5` to the total.

**Code**: `src/eval/reward_2d.py:compute_reward` step 3.

---

## 4. Weights — `w_sem`, `w_goal`, `w_col`

### What they are
Positive floats that trade off the components against each other.

### How they're chosen (current defaults)

| Weight | Value | Rationale |
|---|---|---|
| `w_sem` | 1.0 | Semantic is a [0,1] signal; unit weight is the natural baseline. |
| `w_goal` | 0.5 | Half the semantic weight so that "walk on grass" beats "walk toward goal off a cliff". |
| `w_col` | 5.0 | Collision is catastrophic. 5x the semantic weight ensures the model never trades a small semantic gain for stepping on something dangerous. |

These are **hand-picked defaults**, not learned. They live in `RewardWeights` (`src/eval/reward_2d.py`) and can be overridden on the CLI:

```
--w_sem 1.0 --w_goal 0.5 --w_col 5.0
```

That's what `scripts/ablate_reward.py` uses to sweep configurations.

### How to tune them
Two strategies (open — see OPTIONS.md #3):
1. **Fixed defaults from human intuition** (current approach).
2. **Learned from expert demonstrations** (inverse RL — a future project).

For now: fixed. Revisit if Milestone C shows the policy exploiting a weight imbalance.

---

## 5. Step cost — `w_step` (added 2026-07-14)

### Why we added it

Without a step cost, a robot that just STANDS STILL on grass gets:
```
R = 1.0 · 0.95  +  0.5 · 0  −  5.0 · 0  =  +0.95 per step, forever
```
The policy could learn "do nothing" as a viable strategy because it keeps accumulating positive reward without needing to move. Real navigation RL always includes a small negative per-step cost to punish inaction.

### How it's chosen
Constant `−w_step` subtracted from the total every frame. Default `w_step = 0.05`. Small enough that walking on grass is still net-positive (+0.90 instead of +0.95), large enough to punish idle behavior over long episodes.

### How it's computed
```python
step_term = -w_step
```
That's it. No dependency on state.

**Now a robot standing still nets +0.90/step; a robot walking toward the goal nets ~+1.05/step. The difference (0.15/step) creates the gradient the policy needs to prefer movement.**

---

## 6. The full formula

```python
R = w_sem  · R_semantic(state)               # in [0, w_sem]
  + w_goal · R_goal(state, prev_state)       # in ~[-w_goal · max_step, +w_goal · max_step]
  - w_col  · R_collision(state)              # in [-w_col, 0]
  - w_step                                    # constant per-frame negative
```

With defaults (`w_sem=1, w_goal=0.5, w_col=5, w_step=0.05`), typical single-frame range:
- Best case (all walkable, moving toward goal, no collision): ~ +1.0 + 0.15 - 0 - 0.05 = **+1.10**
- Worst case (all tree/water, moving away from goal, full collision): ~ 0 - 0.15 - 5 - 0.05 = **−5.20**

Observed on rugd_trail_00 after step cost: total ranges from **−4.84 to +1.00**, matches the theoretical bounds.

---

## 7. Why NOT zero-center the semantic score?

Alternative we considered: change `traversability.yaml` from `[0, 1]` to `[-0.5, +0.5]`. Grass = +0.45, sand = 0 (neutral), water = -0.5. Aligns with the RL-class rule of thumb "baseline should be zero, positive for good, negative for bad."

We didn't go this way. Three reasons:

1. **There's no natural "neutral" surface.** For a Go2 quadruped, grass is genuinely GOOD to walk on — it's not "meh, neutral." Making it score 0 misrepresents ground truth. The scoring should reflect physical suitability, not conform to a symmetric range.

2. **RL algorithms are largely invariant to constant reward shifts.** PPO's advantage function subtracts a learned baseline, so a persistent +0.5 offset just gets absorbed. What matters is the DIFFERENCES between states (grass vs. water vs. tree). Our scoring already has plenty of dynamic range: 0.95 grass vs 0.2 vegetation vs 0.0 water.

3. **The "stand still gaming" problem is really about missing progress incentive**, not about the sign of semantic. Adding a step cost directly solves it. Zero-centering the semantic would ALSO reduce the standing-still reward, but it wouldn't reward moving. So it's a less surgical fix.

The "positive baseline + step cost + collision spike" pattern we ended up with is standard in navigation RL literature (e.g. World4RL, DriveWorld). We're in good company.

---

## 8. What's NOT in the reward yet (deferred)

Documented in `OPTIONS.md #3`:

- **Clearance** — vertical headroom above the ground (branches, awnings). Requires depth or 3D Gaussians. Will add when we have real trajectories.
- **Stability** — terrain roughness (rocky/rooty surface where the robot might trip). Currently approximated by semantic (rock has score 0.4).
- **Energy** — cost of movement (uneven terrain = more effort). Deferred as low-priority.
- **Smoothness** — penalize jerky actions. Add if policy is jittery in Milestone C.
- **Time cost** — small per-step negative reward to encourage efficient trajectories. Add if trajectories are correct-but-slow.

---

## 9. Design decisions and caveats

- The polygon math uses camera projection, so it depends on **accurate intrinsics + poses**. With synthetic trajectory (current), the polygon lands on the ground assuming a straight-forward walk. With real poses (once we run the reconstructor), the polygon reflects where the actual robot was pointing.
- SAM3's labels have noise (vegetation absorbs rocks and trees — being fixed via the priority-order refactor). Reward accuracy is capped by label accuracy. The reward function correctly reads whatever the labels say.
- The reward is designed to be **RL-friendly** — bounded, continuous, decomposed. No sparse "reached goal!" bonus at the end of an episode (that's a separate decision for Milestone C).

---

## 10. Where the code lives

| Piece | File |
|---|---|
| Per-class score table | `config/traversability.yaml` |
| Score loader + class names | `src/eval/traversability.py` |
| Palette (for label overlays) | `src/eval/palette.py` |
| Main reward function | `src/eval/reward_2d.py` |
| Clip loader (video + labels + poses) | `src/eval/load_clip.py` |
| CLI entry point | `scripts/validate_reward.py` |
| 4-config ablation runner | `scripts/ablate_reward.py` |

Alternative Gaussian-based reward stack (on ice for Milestone C):
`src/eval/reward_gaussian.py`, `ground_plane.py`, `footprint.py`, `mock_data.py`.
