# World-model limits — what we measured, what it means for training

Measured boundaries of the reconstructed world model used as the RL simulator
(RUGD scenes, WorldMirror reconstruction -> 15.2M Gaussians -> gsplat raster ->
Wan 2.1 diffusion polish). Every number below has an artifact behind it; see
the index at the bottom for filenames and how to view each one.

## 1. The numbers

| Property | Result | How measured |
|---|---|---|
| Metric scale | camera height anchored to the RUGD platform spec (<25 cm -> 0.25 m); honest uncertainty **±20%** | implied walking speeds 0.18–0.47 m/s vs 1.0 m/s platform top speed (sanity holds) |
| Ground height | **~2.7 cm** mean error | p40 percentile of splat heights within 0.75 m, validated against camera-height ground truth along the trajectory |
| View coverage, rotation | front **81–100%**, sides **~0%**, back **0%** | 360° spin every 15°, coverage = fraction of pixels with rasterizer alpha > 0.5, 3 probe points x 5 scenes |
| View coverage, lateral | **100% / 80% / 67% / 46%** at 0 / 0.5 / 1 / 2 m off-path | same alpha metric, offsets perpendicular to the path (one side only — see caveats) |
| Diffusion at the boundary | cleans within the corridor; **invents** beyond it, sometimes plausibly-walkable fiction | paired raster/diffused sheets at 7 poses x 4 scenes |
| Temporal consistency | invented content is stable across frames in the probed modes (qualitative) | 81-frame video-diffusion passes: spin / lateral slide / 1 m-offset traverse, raster-vs-diffused side by side |

**One-sentence summary: the world model is a *forward corridor* along the
recorded trajectory — trustworthy within roughly ±1 m laterally and ±45° of
heading, empty behind and beside, and beyond that boundary the diffusion
fills the frame with coherent fiction.**

## 2. The part that becomes a capability: coverage is a free, online trust signal

The coverage number (fraction of pixels with rasterizer alpha > 0.5) is not a
probe-time luxury — it is computed from the alpha channel we already render
at **every policy step, at zero extra cost**. That turns "know the boundary"
into "the robot knows the boundary":

- **Per-view trust score.** Each observation carries its own coverage value.
  The probe curves calibrate the threshold: above ~0.8 the diffusion is
  cleanup, below ~0.5 it is majority fiction.
- **Action thresholding (the meeting-notes idea).** The policy (or a wrapper)
  can treat low-coverage directions as unknown terrain: penalize or veto
  actions that steer into views below the threshold. Same mechanism a real
  robot uses for "I can't see there" — here it falls out of the renderer.
- **Per-pixel honesty mask.** Alpha localizes *which* pixels the diffusion
  invented, not just how many. Any downstream consumer (reward, uncertainty
  estimate, viz) can mask to real-geometry pixels.
- **Already load-bearing for the reward.** The traversability reward reads
  rasterized Gaussian labels — it scores only where geometry exists and is
  structurally unable to reward hallucinated terrain. The diffusion polishes
  what the policy *sees*, never what it is *paid* for.

What coverage does **not** catch: geometry that exists but is wrong
(mis-reconstructed surfaces, occlusion errors). The planned measurement for
that is held-out view synthesis: reconstruct from a subset of frames, render
the held-out poses, compare to the real frames (PSNR/SSIM + label IoU). That
upgrades "accuracy" from a corridor map to a per-scene quality score.

## 3. Design decisions these limits already drove

1. **Reward from geometry, not from diffusion** (hallucinated walkability is
   poison; rasterized labels can't hallucinate).
2. **Diffused observations confined to the measured ribbon** (±1 m / ±45°):
   the planned diffusion-in-the-loop training only requests diffused views
   inside the trust corridor, where the probe shows cleanup rather than
   invention.
3. **Goal placement strategy**: goals validated against walkability and splat
   density (goals.json + photo sheets with metric ground rings) — a goal
   outside the corridor would be a goal the policy can neither see nor be
   honestly rewarded for reaching.
4. **Spawn/goal curricula stay on-corridor**: spawns sample the recorded
   trajectory; per-episode random goals sample trajectory positions, so no
   episode is asked to traverse unmodeled space.

## 4. Caveats (say them before anyone else does)

- Lateral offsets probed on **one side** of the path only; asymmetric scenes
  (treeline vs open grass) could differ per side. Trivial to extend (negative
  offsets, one job).
- Diffusion pairs cover 4 of 5 probe scenes (5th hit a GPU memory limit;
  single-scene resubmit if needed).
- Temporal-consistency verdict is qualitative (visual). A quantitative
  version: warp frame t to t+1 with the known camera motion + rendered depth,
  measure photometric error separately in high-alpha vs low-alpha regions.
- Metric scale is ±20%; all meter figures inherit that band.

## 5. Artifact index

All under `outputs/probe/` unless noted. Left/top = rasterizer, right/bottom = diffused.

| Artifact | What it shows |
|---|---|
| `probe_curves.png` | coverage vs yaw angle and vs lateral offset, all scenes on one plot |
| `<scene>_spin_strip.png`, `<scene>_offset_strip.png` | raster views every 45° / at each offset, coverage % printed per tile |
| `<scene>_diffusion_pairs.png` (4 scenes) | 7 poses from safe to absurd; where cleanup becomes invention |
| `<scene>_spin_pair.mp4` | full 360° at mid-trail; back hemisphere is pure invention — watch whether reality reappears in place when the camera comes back around |
| `<scene>_slide_pair.mp4` | lateral drift 0 -> 2 m; invention taking over in real time as coverage drains |
| `<scene>_walk1m_pair.mp4` | full traverse at 1 m offset — the deployment-relevant view |
| `outputs/goal_maps/goals.json`, `*_goalphoto.png` | goal placement + the metric ground-ring photo checks |

Suggested demo order for a meeting: `spin_pair.mp4` (the boundary, viscerally)
-> `slide_pair.mp4` (the boundary, continuously) -> `probe_curves.png` (the
boundary, quantified) -> section 2 of this doc (what the robot does about it).
