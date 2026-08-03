# Results — semantic world model + RL navigation

Architecture and loss functions: ARCHITECTURE.md (in this folder).

## 01_world_model_validation
The reconstructed environment compared against reality.
- replay_v3_vs_original.mp4 — the environment rendered along the real robot's
  recorded path, side by side with the original video.
- EXPERT_REPLAY_reference.mp4 — same replay with the reward overlay; the
  reward stays positive along the real path.
- projection_test videos — the robot's next real position projected into the
  current image lands on the path it then walked.
- ground/ — ground-height estimation from the Gaussians (~3 cm mean error).

## 02_geometric_boundary
Where the world model can be trusted, measured.
- probe_curves.png — fraction of real (non-hole) pixels vs rotation angle and
  vs lateral offset from the recorded path. Summary: reliable within about
  ±1 m and ±45° of the path; empty behind and to the sides.
- strips/ — the rendered views behind those numbers.
- diffusion_pairs/ — rasterized vs diffusion-completed views at increasing
  offsets: near the path the diffusion cleans up, far from it it invents.
- motion_videos/ — the same comparison with a moving camera (rotation,
  sideways drift, 1 m-offset walk); left = rasterized, right = diffused.

## 03_goal_setting
- goals.json — one goal per scene, scored by walkability and reconstruction
  density.
- photos/ — each goal projected into the camera view with a real-scale 0.3 m
  ground ring, plus the view standing at the goal.

## 04_rl_policy
Policy versions and evaluations (20 episodes each; videos show the
first-person view with a minimap: white = path walked, green = goal).
- eval_v4_success_videos — first working policy (fixed goal): 100% success.
- eval_v5_traverse — full traverses from the trail start: 100%.
- eval_v5_goal60_FAIL — the same policy on a goal twice as far: 0%. It walks
  past the goal and never turns back. This motivated random goal training.
- eval_v6_peak_goal30 / goal60 — random-goal training: the far goal is fixed
  (100%) and the remaining weakness is close goals (mapped by a 9-distance
  sweep: 0% at 1.5 m rising to 100% at 4 m).
- reward_audit/ — each reward term's actual per-step contribution.

## 05_semantics
The diffusion model learning to output semantic maps.
- compare_v6_v7_semantic.mp4 — training against dense human annotations
  cleaned the maps and corrected SAM3's errors (e.g. trees labeled building).
- compare_v8val5_2x2.mp4 — current version after 5 epochs: RGB, the holey
  input hint, and the model's raw and final semantic output. The earlier
  speckle pattern is gone; remaining weakness is low confidence (soft
  colors), addressed by the two loss additions now in testing.
- sam2_segment_examples/ — class-agnostic segments used by the new
  segment-consistency loss.
