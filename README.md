# nav-rl — RL policy for outdoor navigation using a semantic 4D world model

Trains a Unitree Go2 navigation policy inside a NeoVerse-based world-model
simulator, using semantic traversability as the reward signal. Real deployment
on Go2 uses RGB only — semantics live only in the sim's reward, not in the
policy's observation.

## Where this fits in the bigger project

- **Track A** (in `../NeoVerse/`) — fine-tunes NeoVerse's diffusion to output
  clean semantic labels jointly with RGB. Long-term source of cheap semantics.
- **Track B** (this repo) — RL policy training. **Doesn't wait for Track A.**
  Uses NeoVerse's existing RGB pipeline + SAM3-on-rendered-RGB for reward.

When Track A lands, we swap the reward backend from "SAM3 on Wan output" to
"diffusion's own semantic channels" — a one-line change in `env/reward.py`.

## Pipeline at each RL step

```
policy chooses action (v_forward, omega_yaw)
    -> new robot pose
    -> world model rendered at pose (RGB from NeoVerse + Wan diffusion)
    -> SAM3 on that RGB -> semantic mask
    -> project next-foot 3D position -> pixel in the image
    -> semantic class at that pixel -> traversability -> reward
```

**Observation to policy**: RGB image + goal vector. That's it.
**Deployment**: same observation shape (RGB from a real camera + goal), same policy.

## Repository layout

```
src/
  env/
    scene_env.py       # Gym env: SceneEnv(reset, step, render, ...)
    pose_cache.py      # Precomputed / cached world-model renders per pose
    reward.py          # Traversability reward from semantic mask + pose
  policies/
    (Go2 walker checkpoint goes here later)
  eval/
    reward_accuracy.py # Phase 1d — Jing's priority; validate reward vs GT
  train.py             # SB3 PPO entry point
configs/
  stage1_static.yaml   # 5-10 waypoints, discrete-ish env, RGB obs
data/
  cached_scenes/       # gitignored large binary caches
```

## Status

- [ ] Phase 1a: gym env skeleton
- [ ] Phase 1b: pose-cache backend
- [ ] Phase 1c: reward function
- [ ] Phase 1d: reward accuracy eval (Jing's priority — do BEFORE PPO)
- [ ] Phase 2:  toy PPO smoke-test + W&B integration
- [ ] Phase 3:  refine pretrained Go2 walker on 5-10 scenes
- [ ] Phase 4:  swap reward backend to Track A output

## Key design decisions (locked)

- Observation = RGB + goal. Semantics stay in the reward, not the observation.
- Action = continuous `(v_forward, omega_yaw)`, `Box(-1, 1, (2,))`, scaled inside `step()`.
- Pipeline pattern from World4RL (arxiv 2509.19080): refine a pretrained policy
  with few high-quality rollouts inside a frozen world model. **Not** from-scratch PPO.
- Reward backend swappable: "sam3-on-rgb" (default, works today) or
  "diffusion-semantics" (Track A output, when ready).
