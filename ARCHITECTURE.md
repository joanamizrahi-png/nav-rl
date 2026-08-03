# Architecture and loss functions

## 1. Semantic world model

### 1.1 Pipeline

```
monocular video (81 frames, 560x336)
   │
   ▼
Reconstruction: feed-forward network (WorldMirror/VGGT) → ~15M 4D Gaussians
   + SAM3 semantic labels attached to each Gaussian
   │
   ▼
Rendering: rasterize the Gaussians at any camera pose and time
   → rough RGB, depth, holey semantic map, alpha
   │
   ▼
Diffusion: Wan 2.1 (14B) video diffusion refines the rough render
   → clean RGB + dense semantic map
```

The alpha channel gives, for every rendered pixel, whether real geometry is
present. It is used as a per-view / per-pixel confidence measure.

### 1.2 How semantics are added to the diffusion

The diffusion model originally denoises 16 latent channels (RGB after VAE
encoding). We add 16 more for semantics: the class map is colorized
(class id → RGB color), encoded with the same VAE, and concatenated with the
RGB latents. The network is expanded with parallel, zero-initialized modules
for the semantic channels (input patch-embedding, output head, control-branch
input), so the pretrained RGB behavior is untouched at initialization. A
rank-8 LoRA on the transformer trunk lets semantic information flow through
the frozen network.

Conditioning: the rasterized rough RGB + depth + holey semantic map.
Training input (hint): SAM3 labels rendered from the Gaussians.
Training target: RUGD dense human annotations (SAM3 for clips without GT).

So the model learns: given a holey SAM3-quality map, produce a dense
human-quality map.

### 1.3 Loss functions

Base diffusion loss (flow matching). Latents are noised as
x_t = (1−σ)·x_0 + σ·ε, and the network predicts the velocity v = ε − x_0:

    L_rgb = || v_pred − (ε − x_0) ||²        (RGB channels, unchanged)

v8 changes the semantic-channel losses (three additions):

1. x0-prediction. The semantic channels predict the clean latent directly
   instead of the velocity:

       L_sem = || x̂_0 − x_0 ||²

   Reason: noise/velocity regression on label maps gives no stable learning
   signal for segmentation (shown in prior work); this was the cause of the
   speckle. At inference the prediction is converted back to a velocity,
   v = (x_t − x̂_0)/σ, so the sampler is unchanged.

2. Cross-entropy in image space (weight 0.1). At low-noise timesteps
   (σ ≤ 0.7), decode x̂_0 with the VAE and read per-pixel class logits with a
   small conv head:

       L_ce = CrossEntropy( head(VAE_decode(x̂_0)), GT class map )

   Reason: the latent MSE never checks whether a pixel is the right class;
   this loss does, at full image resolution.

3. Segment homogeneity. SAM2 provides class-agnostic segments ("these pixels
   belong together"). Each pixel is penalized for deviating from its
   segment's mean class distribution p̄_s (detached):

       L_seg = − mean over pixels i [ p̄_s(i) · log p_i ]

   A uniform segment costs ~0. Speckle is disagreement inside a segment, so
   this loss removes it directly, without needing to know the correct class.

Total: L = L_rgb + 2.0·L_sem + 0.1·L_ce + λ_seg·L_seg.

### 1.4 Inference

Two passes. RGB comes from the original (vanilla) weights, semantics from the
finetuned model. This keeps RGB clean: semantic training slightly degrades
the shared trunk. Planned improvement: anchor the semantic pass by fixing its
RGB channels to the vanilla pass's latent trajectory at every denoising step,
which forces the semantic map to describe exactly the RGB image shown.

## 2. RL policy

### 2.1 Environment

The world model is the simulator. The robot state is a pose (position +
heading) in meters.

- Observation: rendered RGB at the robot's eye height (0.25 m) + the goal
  vector (dx, dy, dyaw) in the robot's frame.
- Action: (forward, turn), continuous in [−1,1], scaled to at most 0.25 m
  and 0.3 rad per step.
- Episode: spawn on the recorded trajectory; goal = a trajectory position
  (sampled per episode between frames 15–70 since v6). Success = within
  0.75 m of the goal. Timeout = 60 steps.

### 2.2 Reward

Per step:

    r = 1.0 · (s_terrain − 1)          terrain, as a cost (≤ 0)
      + 1.5 · (d_prev − d_now)         progress toward the goal
      − 1.0 · f_collision              fraction of footprint on obstacles
      − 0.3 · f_void                   fraction on missing geometry
      − 0.05                           per-step cost
      − 0.05 · |turn|                  spin cost
      + 50 on reaching the goal       (once, terminal)

s_terrain is the mean traversability score of the ground within 1.5 m ahead,
read from the semantic labels rasterized from the Gaussians (scores per class
in config/traversability.yaml, e.g. grass 0.95, mud 0.3, tree 0.0). Because
these labels only exist where geometry exists, the reward cannot be earned on
hallucinated terrain.

Design note: all per-step terms are ≤ 0 and the only large positive is the
terminal bonus. An earlier version paid positive terrain reward per step and
the policy learned to pace back and forth forever instead of finishing.

### 2.3 Policy network

Two input branches: a small CNN for the image, an MLP for the goal vector.
Features are merged and passed through two 64-unit layers, ending in an actor
(a Gaussian over the two actions) and a critic (value estimate).

### 2.4 Training

PPO (Stable-Baselines3): collect 128 environment steps, then several
gradient passes maximizing the clipped surrogate objective

    L_ppo = E[ min( ρ·A, clip(ρ, 1−0.2, 1+0.2)·A ) ]

where ρ is the new/old action-probability ratio and A the advantage (GAE).
Learning rate 1e-4; a KL trust region (target 0.02) stops the passes on a
batch if the policy drifts too far from the one that collected it.
Checkpoints every 2k steps.

Before PPO, the policy is warm-started with behavior cloning on 2,480
state→action pairs extracted from the real trajectories (actions recovered
from consecutive pose differences).

## 3. How the two sides connect

- The policy's observations come from the world model's renderer (currently
  the rasterizer; diffused observations planned, restricted to the measured
  trust region of ±1 m / ±45° around the recorded path).
- The reward's semantic scores come from the Gaussians directly.
- At deployment, the same semantic model runs on real camera images, so the
  perception used in training transfers to the robot.
