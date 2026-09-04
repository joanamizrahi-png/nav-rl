# TODO — shared between Joana and Claude

Edit freely. Dates are hard: results freeze **Sept 9**, submit **Sept 15** (2026).

## Now
- [ ] Read the six clean evals (464601–464606) when they land; gate on `[spawn range] frames [10,75)` and on `CRASH` appearing in outcomes
- [ ] Figures for them: `plot_blind_vs_sighted.py --episodes 20 --overview_out`, `side_by_side.py` (neoverse python)
- [ ] Re-snapshot 463169's checkpoints (`SNAP_463169_warm256704`) — 463168 is overwriting them
- [ ] Dashboard every 2–3 h: `python3 scripts/fleet_dashboard.py <12 jobs> --out fleet_$(date +%H%M).png`

## Corner test (the memorable figure, if it works)
- [ ] Find a corner in an existing scene: recorded path bending around grass (overview maps show it; `gnd_AU_180` / `gnd_AUw360` are named for how far the walk turns)
- [ ] Eval: pin a goal past the bend (`GOAL_XY`), spawn before it, blind vs sighted, overhead paths with outcome symbols
- [ ] Only if present-but-weak: warm-start the best arm on that one scene (`SCENES=<one>`); `gnd_AUw60` loads alone if it is the corner scene. **No new data — no time.**

## Real robot
- [ ] Inference wrapper: camera frame -> crop/resize 224x336 -> policy -> (throttle, yaw) at ~2 Hz; goal vector in the policy's frame from odometry; joystick override
- [ ] Course: one sidewalk with a flat grass verge, so geometry alone says "traversable" and semantics says "no" -- the case the reward exists for
- [ ] Same (start, goal) set for every method tested; same arrival radius; same safety-stop rule
- [ ] Blind vs sighted on the robot: same wrapper, image zeroed
- [ ] Metrics per trial: reached / steps to goal / fraction of steps on non-walkable ground / interventions

## Paper (start now; these do not wait on results)
- [ ] Method section (reward spec artifact is the source)
- [ ] Related work: 3 paragraphs, anchors in the 09-03 chat / STATUS
- [ ] §4.5 failure analysis: phantom terrain vs coverage, stop-at-edge geometry, no-curriculum freeze
- [ ] Intro + contribution bullets (after results freeze)
- [ ] Abstract last

## Later / next generation
- [ ] v26b / v28b finish ~hour 43 → grade against v26; new fleet on the winner
- [ ] Ancestor pair at its OWN range (5–10 m) for a clean train/eval match, if the 2.5–6.5 comparison needs it
- [ ] Merge the video rollout into the metrics loop (double rollout is wasted compute; sampler is seed=0 so not a mismatch)
- [ ] `void_terminate_frac` — off by choice; revisit if unsupported-ground-under-coherent-view keeps showing up

## 2026-09-04 morning checklist (written 01:45)

Ids now: 465703 A warm, 465704 A cold, 465705 B warm, 465706 B cold, 465707 W,
465708 T1 hybrid, 465709 T3 W-cold, 465710 T2 W-near-box. All 12 h wall.
Still running from the first batch (48 h, end Sep 5 ~01:10): 463164, 463165,
463168, 463170; 464143 ends Sep 5 13:11; semantics 463197/463198 finish
their 20 epochs ~Sep 4 20:30.

1. `git status` on the Mac -- every commit of the night is pushed (last:
   "Report view alpha on the eval path; refuse evals whose coherence terms
   would be inert").
2. `squeue -u jmizrahi`: which of the eight started. For each running one:
   `grep -E 'map reward|crash=|reward_source|map_inflate|collision_look_ahead|halt_terminate_steps|halt_penalty_scale|terrain_speed|REFUSED|Error' /scratch/m000204-pm06b/joana/slurm-ppo-real-<id>.out | head -14`
   Expect on every arm: six `[map reward]` lines, `crash=2500.0`,
   `halt_terminate_steps 3`, `halt_penalty_scale 0.3`. Then per arm:
   A -> `reward_source generated`; B -> also `terrain_speed_scaled True`;
   W/T3 -> `reward_source map`, `map_inflate_m 0.1`; T1 -> `map_then_generated`;
   T2 -> also `collision_look_ahead_m 0.6`. Anything missing: scancel it, paste.
3. First wandb dump of each: reward/collision and reward/semantic in A's
   range; diag/phantom, diag/label_agree, diag/map_void_frac visible;
   diag/used_generated on T1; diag/collision_off_frame on T2 (expected high).
4. Eval 465642 (grass sighted warm): `crashes_that_were_phantoms` + the
   by-alpha table (alpha unknown -- pre-fix). Then rerun on the fixed code:
   the grass sighted line (GOALRADIUS=0.65) and 463170@32k sighted
   (GOALRADIUS=0.75). Both must print `mean_coverage` as a number.
   `REFUSED` = alpha still missing -> stop and look.
5. Dashboard: neoverse python, id list = 463164 463165 463168 463170 464143
   465703 465704 465705 465706 465707 465708 465709 465710. Panels 14/15
   fill from the first new dump.
6. Semantics epoch 15 (~09:30): render + grade + panels, unchanged commands;
   rows grass / sidewalk / obstacle; decide whether either model replaces
   v26 e10 for the next launches.
7. Corner probe on gnd_AUd210 (`plot_goal_on_map.py --scene gnd_AUd210
   --goal_xy 16.1,-26.4 --spawn_frames 45,49`).
8. When a 12-h arm ends: decide continue (WARMSTART from its last
   checkpoint, curricula restart) or stop.
