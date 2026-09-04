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
