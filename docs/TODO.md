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

## 2026-09-04 morning checklist (written 01:20)

1. `git status` on the Mac: the 01:20 commit ("Report view alpha on the eval
   path; refuse evals whose coherence terms would be inert") must be pushed
   before any eval is launched.
2. Queue: which of A 465184/465185, B 465186/465187, W 465661, T1 465662,
   T3 465663, T2 465666 started. For each running one:
   `grep -E 'map reward|crash=|reward_source|map_inflate|collision_look_ahead|halt_terminate_steps|halt_penalty_scale|REFUSED|Error' slurm-ppo-real-<id>.out | head -14`
   Expect six `[map reward]` lines on every arm (diagnostics build the map
   on A and B too), `crash=2500.0`, `halt_terminate_steps 3`,
   `halt_penalty_scale 0.3`; `reward_source map` on W/T3/T2,
   `map_then_generated` on T1, `map_inflate_m 0.1` on the map arms,
   `collision_look_ahead_m 0.6` on T2 only, `terrain_speed_scaled True` on B.
3. 465642 (grass sighted, warm, r0.65): read `crashes_that_were_phantoms`
   and the by-alpha table (alpha will be unknown -- launched before the fix).
4. Rerun on the fixed code once 465642 is gone from the queue: the grass
   sighted line (GOALRADIUS=0.65) and 463170@32k sighted (GOALRADIUS=0.75).
   Both must print `mean_coverage` as a number now; a REFUSED means alpha
   still does not arrive -- stop and look.
5. Dashboard with the new arms appended to the id list; panels 14/15
   (phantom rate, label agreement) fill as soon as any new arm dumps.
6. Semantics epoch 15 (~mid-morning): the render + grade + panel commands
   from the night, unchanged; compare the grass/sidewalk/obstacle rows.
7. Corner probe: verify the gnd_AUd210 pair on the map
   (`plot_goal_on_map.py --scene gnd_AUd210 --goal_xy 16.1,-26.4 --spawn_frames 45,49`).

