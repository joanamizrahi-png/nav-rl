# Results — semantic world model + RL navigation

Architecture and loss functions: ARCHITECTURE.md (in this folder).
Diagrams: "Diffusion internals.png", "RL gym.png".

## 01_world_model_validation
Checks that the reconstructed environment matches reality.
- replay_v3_vs_original.mp4 → drive the env along the real robot path,
  side by side with the original video
- EXPERT_REPLAY_reference.mp4 → same replay with the reward overlay
  (reward stays positive on the real path)
- projection_test videos → the robot's next real position projected into the
  current image lands on the path it then walked
- ground/ → ground height estimated from the gaussians (~3 cm mean error)

## 02_geometric_boundary
Where the world model can be trusted (measured).
- probe_curves.png → fraction of real (non-hole) pixels vs rotation angle and
  vs lateral offset. Reliable within ~±1 m / ±45° of the path, empty behind
- strips/ → the rendered views behind those numbers
- diffusion_pairs/ → rasterized vs diffused views at increasing offsets:
  near the path the diffusion cleans up, far from it it invents
- motion_videos/ → same comparison with a moving camera
  (left = rasterized, right = diffused)

## 03_goal_setting
- goals.json → one goal per scene, scored by walkability + reconstruction
  density
- photos/ → each goal projected into the camera view with a real-scale 0.3 m
  ground ring, + the view standing at the goal

## 04_rl_policy
Policy versions and evaluations (20 episodes each; videos show the
first-person view + a minimap: white = path walked, green = goal).
- eval_v4_success_videos → first working policy (fixed goal): 100%
- eval_v5_traverse → full traverses from the trail start: 100%
- eval_v5_goal60_FAIL → same policy, goal 2x further: 0% (walks past the
  goal and never turns back → motivated random goal training)
- eval_v6_peak_goal30 / goal60 → random-goal training: far goal fixed
  (100%), close goals weak (0% at 1.5 m → 100% at 4 m, from a 9-distance
  sweep)
- v6d_evals → current best single-scene policy: 100% at all nine goal
  distances + zero-shot on two unseen scenes (100%, no collisions)
- reward_audit/ → each reward term's actual per-step contribution

## 05_semantics
The diffusion model learning to output semantic maps.
- compare_v6_v7_semantic.mp4 → training on dense human annotations cleaned
  the maps and corrected labeling errors
- compare_v8val5_2x2.mp4 / compare_v8_stages_1_2_3.mp4 → the three v8 loss
  stages at 5 epochs (attribution check before the full run)
- heldout_v9/ → v9 (14 classes) evaluated on two scenes excluded from
  training: 78.8% / 69.9% pixel accuracy. See its README
- sam2_segment_examples/ → class-agnostic segments used by the
  segment-consistency loss
- class_palette_legend.png → color → class key (old 30-class palette;
  heldout_v9 has the 14-class one)
