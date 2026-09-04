# How to evaluate a policy so the numbers mean something

Written 2026-09-03, after a day in which **six** separate settings silently
differed between training and eval. Every one produced plausible numbers rather
than an error.

**Runs launched after 2026-09-03 do not need this file.** They write
`env_config.json` (all reward and termination settings) and
`curriculum_state.json` (where the curriculum actually reached), and
`eval_policy` adopts both automatically and prints `MISMATCH` on any
disagreement. This file is for **checkpoints from before that**, which have
neither.

---

## 1. The shared block

Export it once per shell. Everything here matches what the 2026-09-02/03 arms
actually trained with.

```bash
export EV="LIVE=1 NOGATE=1 \
LIVECKPT=/scratch/m000204-pm06b/joana/runs/train_semantic_v26_campus/checkpoint-epoch-10.safetensors \
SEMPAL=4 GOAL360=1 GOALCONE=50 GOALFRAMERANGE=15,70 GOALSUPPORT=0.6 VIDEOS=20 \
SCENE=gnd_AUw360 TRAV=config/traversability_v14_walkway.yaml \
COLLTERM=0.35 COLLPEN=1000 COH=10 COHTAU=0.4 \
SEMW=5 REWSCALE=0.01 SMOOTHCOST=1 MAXSTEPS=90 FWDONLY=1 \
SPAWNMIN=10 SPAWNJYAW=20 SPAWNJLAT=0.4 \
GOALWEIGHT=10 GOALBONUS=1000 TIMEOUTPEN=100 TIMEOUTDIST=1 \
PROXW=10 PROXMARGIN=5 PROXDELTA=1 VOIDCOST=0.3 STEPCOST=0.05 COLLWEIGHT=1"
```

**Before every `env $EV ...` launch, in every new shell:**

```bash
echo "$EV" | wc -w
```

About 30 means the export is alive. `0` means this is a new login and the
export is gone -- re-run the block above first. On 2026-09-03 jobs
464831/464832 were launched with an empty `EV`: no `LIVE`, no `LIVECKPT`, no
`NOGATE`, no reward knobs. They ran the cloud rasterizer, finished in one
minute, and printed metrics. The signature of that failure, so it is
recognised on sight:

| symptom | reading |
|---|---|
| `Elapsed` about a minute | the diffusion pipeline was never loaded (it alone takes 25-40 min) |
| `mean_coverage: None` | no alpha, so no generated view |
| `ground_share` with `rough`, `water`, `person`, or bare ids like `15` | raw cloud labels, not the campus palette |

The launcher now refuses a `ppo_live_*` checkpoint without `LIVE=1`, and
`LIVE=1` without `LIVECKPT`.

## Per-arm overrides -- the block above is the SHARED config only

Every arm isolates one knob, and the eval must carry it or it measures a
different task. On 2026-09-03 a wave of evals carried COHTERM per arm and
nothing else: 463879 was scored with crash terminal 0.35 (trains at 0.5) and
the halt arms were scored WITHOUT the halt terminal (halts became timeouts).
Both pairs had to be rerun.

| arm | add to the launch |
|---|---|
| 463170 | `COHTERM=0.05` |
| 463879 | `COLLTERM=0.5` |
| 463224 | `COLLPEN=200` |
| 464143 | `SMOOTHCOST=5` |
| 464428 / 464429 | `HALT=5` |
| tonight's A (halt+tie) | `HALT=3 HALTEPS=0.15 HALTSCALE=0.3 COLLPEN=2500` |
| tonight's B / C (+speed) | `HALT=3 HALTEPS=0.15 HALTSCALE=0.3 COLLPEN=2500 SPEEDCOST=1` |
| map-reward arms (`_rmap` in the dir) | add `REWSRC=map` to whatever the arm's other knobs are |
| map-reward arms with the near crash box (`_ca0.6` in the dir) | `REWSRC=map COLLAHEAD=0.6` plus the arm's other knobs |
| map-reward arms with `_mi<x>` in the dir | add `MAPINFL=<x>` |
| warm arms | `GOALRADIUS` from the run's `curriculum_state.json` (automatic) |

The run's own record is `outputs/<run>/launch.txt` (the ledger entry):
`grep knobs` it before every eval and copy every knob that is not in the
shared block.

For a policy trained on **v21** semantics (e.g. the ancestor `ppo_240704`),
override two:

```bash
LIVECKPT=/scratch/m000204-pm06b/joana/runs/train_semantic_v21/checkpoint-epoch-12.safetensors SEMPAL=1
```

`VIDEOS=20` records every episode. Each video episode is rolled out twice (the
video pass, then the metrics pass -- same seed, and the sampler is `seed=0`, so
they are the same episode), which adds ~10-15 min to an eval. The episode you
want to watch is always the one that was not recorded; record them all.

## 2. The per-arm tail

Three things change per checkpoint. Get them from the arm's own training log.

```bash
grep -A 6 "curriculum/" /scratch/m000204-pm06b/joana/slurm-ppo-real-<JOBID>.out | tail -7
```

* `GOALRANGE` = that arm's `goal_dist_lo,goal_dist_hi` — the range it was
  training on WHEN YOU STOPPED IT, not the `2,8` it started at.
* `GOALRADIUS` = its `goal_radius` (the radius curriculum tightens 1.0 -> 0.5).
* `COHTERM` = its own `coherence_terminate_tau` (0.1, or 0.05 on that arm).

```bash
cd /scratch/m000204-pm06b/joana/nav-rl && env $EV \
  GOALRANGE=<lo,hi> COHTERM=<tau> GOALRADIUS=<r> CKPT=<...zip> \
  sbatch scripts/slurm/eval_policy.sh
```

Blind half: the identical line with `BLIND=1` after `env $EV`. Both halves share
`--eval_seed 7`, so episode N is the same spawn and goal in both. **Run the pair
back to back** if either uses `$(ls -t ...)`, or they resolve to different
checkpoints. **Keep `VIDEOS` equal** across the pair, or `save_rollout_video`
consumes extra resets and the goal sequences desynchronise.

## 3. Check these three lines before reading any result

```bash
grep -m1 "spawn range" /scratch/m000204-pm06b/joana/slurm-eval-policy-<JOB>.out
grep -oE "^ep +[0-9]+: [A-Z]+" /scratch/m000204-pm06b/joana/slurm-eval-policy-<JOB>.out | awk '{print $3}' | sort | uniq -c
grep -E "adopted training env|no env_config" /scratch/m000204-pm06b/joana/slurm-eval-policy-<JOB>.out
```

1. **`frames [10,75) of 81 (80% of the walk)`.** Anything narrower and the robot
   was never placed where the interesting terrain is.
2. **`CRASH` must appear** among the outcomes. Only `GOAL`/`TIMEOUT` means a
   terminal is off and the policy walked through terrain that would have ended
   the episode in training.
3. `no env_config.json` is expected for old checkpoints — it means every setting
   came from this file rather than from disk.

## 4. Why each knob is there

| knob | what goes wrong without it |
|---|---|
| `NOGATE=1` | eval GATES the labels: low-coverage pixels become void, void leaves the collision fraction (`void_cost > 0`), so **it does not crash where training would** |
| `GOALFRAMERANGE=15,70` | the spawn bound falls back to `goal_frame(30) - 5`, so eval spawns in frames **0-25** while training uses 10-75 |
| `MAXSTEPS=90` | eval defaults to **60** — a third of every episode cut off |
| `COLLTERM` + `COLLPEN` | crash termination defaults **off** (frac 0) |
| `COH`/`COHTAU`/`COHTERM` | coherence termination defaults off |
| `GOALSUPPORT=0.6` | ~14.5% of goals have no reconstruction under them and can never be reached |
| `SPAWNJYAW`/`SPAWNJLAT` | eval spawns exactly on the recorded pose — easier than training |
| `SEMPAL` | v26 is palette 4, v21 is palette 1; wrong value decodes every class wrong |
| `TRAV` | defaults to `traversability_v14.yaml`, not the walkway table |
| `GOALRADIUS` | the radius curriculum moved; `0.5` is where it ENDS, not where a given arm is |
| reward weights | `goal_bonus` 50 vs 1000, `timeout_penalty` 0 vs 100 — these do not change behaviour, only the reported `return=`, which is then on a different scale from training |

## 5. Reading the output

* **`ground_share[c]`** = fraction of STEPS whose *dominant class inside the
  reward footprint* was `c`. The footprint is 1.5 m AHEAD -- this is what the
  robot is heading into, not what it stands on. Pooled across episodes, so long
  episodes dominate.
* **`ground_share['grass']`**, not `mean_trespass_steps`, is the terrain metric.
  The latter is a COUNT and is confounded by episode length: a policy that times
  out at 90 steps logs more grass steps than one that arrives in 10, at a lower
  rate.
* **`closed_frac`** = `1 - d_final/d_start`, so it goes strongly negative when an
  episode ends farther away than it began. A mean near zero can hide 16 arrivals
  and 4 bad wanders.
* **`CRASH 79st`** means it walked 79 steps and THEN crashed, and the episode
  ended there. It is not still running.

## 6. Figures

```bash
python3 scripts/plot_blind_vs_sighted.py --scene <scene> --episodes 20 \
    --sighted <eval_dir> --blind <eval_dir> \
    --out paths.png --overview_out overview.png

/users/jmizrahi/.conda/envs/neoverse/bin/python scripts/side_by_side.py \
    --episodes 6 --sighted <eval_dir> --blind <eval_dir> --out_dir <dir>
```

`side_by_side` needs the **neoverse** interpreter -- the login node's `imageio`
has no ffmpeg backend. The overview prints spawn spread, goal spread, and the
distance from each spawn to the nearest grass: if that median is large, a grass
metric of 0.000 says nothing about the policy.
