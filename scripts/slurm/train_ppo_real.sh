#!/usr/bin/env bash
#SBATCH --job-name=ppo-real-smoke
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-ppo-real-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-ppo-real-%j.err
#SBATCH --exclude=n04,n13,n17,n24

# First-version RL loop (Thursday deliverable): PPO on rugd_trail_00 with
# rasterizer-only observations + Gaussian-label reward, all in real meters.
# Success = loop runs, reward logs, return trends up on the trivial task.

set -euo pipefail

module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r

# RL deps (one-time; no-ops afterwards)
python - <<'PY' || python -m pip install --quiet "stable-baselines3[extra]" gymnasium
import stable_baselines3, gymnasium
PY

NAVRL_ROOT=/scratch/m000204-pm06b/joana/nav-rl
cd "$NAVRL_ROOT"

# Ladder-aware: if the demos file exists we run the BC rung (v3); otherwise the
# pure shaped rung (v2). Output dirs are versioned accordingly — no overwrites.
# DEMOS_FILE overrides which demo set BC uses (e.g. demos_v2.npz = post-fix
# convention + cached diffused obs). Default demos_v1.npz is from 2026-07-27 —
# PRE-yaw-fix (mirrored turns, raster obs): poisons cached runs. NOBC=1 skips BC.
DEMOS=${DEMOS_FILE:-/scratch/m000204-pm06b/joana/outputs/demos_v1.npz}
if [ "${NOBC:-0}" = "1" ]; then
    DEMOS=/nonexistent
fi
if [ -f "$DEMOS" ]; then
    BC_ARGS="--bc_demos $DEMOS"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v4_cost_trail00
else
    BC_ARGS=""
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v2_shaped_trail00
fi
STEPS=200000
SEED=${SEED:-0}
if [ "${LIVE:-0}" = "1" ]; then
    # LIVE per-action diffusion (no cache, 2026-08-23): one 5-frame semantic
    # generation per env step at the exact continuous pose. Measured 1.35 s/step
    # at 560x336 -> 50k steps ~19 h, 200k ~75 h. Submit with:
    #   sbatch --mem=96G --time=36:00:00 ... (fine-tune)  /  --time=96:00:00 (200k)
    # SMOKE=1 -> 500-step gate run (obs sanity + timing + VRAM before real runs).
    BC_ARGS="--live --goal_frame_range 15 70 --goal_min_sep 1.5 --trav_path config/traversability_v14.yaml --labels_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels_v14"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_live_trail00
    STEPS=200000
    if [ "${NOGATE:-0}" = "1" ]; then
        BC_ARGS="$BC_ARGS --no_alpha_gate"
        OUT=${OUT}_UNGATED
    fi
    if [ "${SMOKE:-0}" = "1" ]; then
        OUT=${OUT}_SMOKE
        STEPS=500
    fi
    # LIVE_DEMOS: BC-prime the live run on LIVE-rendered demos (B-prime rescue
    # for the cold-start dream-marination). NEVER pass raster/cached demo files
    # here — observation source must match training.
    if [ -n "${LIVE_DEMOS:-}" ]; then
        BC_ARGS="$BC_ARGS --bc_demos $LIVE_DEMOS"
        OUT=${OUT}_bc
    fi
elif [ "${CACHE:-0}" = "1" ]; then
    # v14-DIFFUSED (2026-08-15, headline ask): the 6d recipe on the
    # corrected right-handed frame, observations from the ribbon cache (v10 +
    # reader diffused views), reward from the cache's alpha-masked diffused
    # labels scored by the v14 table. Requires cache_gen.sh to have populated
    # outputs/ribbon_cache/<scene>. SMOKE=1 -> 10k-step gate run.
    # OBS_CACHE selects which cache to train on: default = v1 path-threaded
    # cache (ribbon_cache); OBS_CACHE=ribbon_cache_spin -> spin cache v2.
    # Non-default caches get their tag appended to the output dir.
    OBS_CACHE=${OBS_CACHE:-ribbon_cache}
    # comma list = HYBRID (multiple caches loaded, sticky-family lookup)
    CACHE_PATHS=$(echo "$OBS_CACHE" | sed 's#[^,]*#/scratch/m000204-pm06b/joana/outputs/&#g')
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70 --goal_min_sep 1.5 --obs_cache ${CACHE_PATHS} --trav_path config/traversability_v14.yaml --labels_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels_v14"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v14diff_trail00
    if [[ "$OBS_CACHE" == *,* ]]; then
        OUT=${OUT}_hybrid
    elif [ "$OBS_CACHE" != "ribbon_cache" ]; then
        OUT=${OUT}_${OBS_CACHE#ribbon_cache_}
    fi
    STEPS=800000
    if [ "${NOGATE:-0}" = "1" ]; then
        BC_ARGS="$BC_ARGS --no_alpha_gate"
        OUT=${OUT}_UNGATED
    fi
    if [ -n "${COLLTERM:-}" ]; then
        # crash termination: footprint overlap >= COLLTERM ends the episode
        # (no goal bonus, -COLLPEN). Without it, walking through a tree costs
        # ~1% of the arrival bonus and the policy correctly ignores obstacles.
        BC_ARGS="$BC_ARGS --collision_terminate_frac $COLLTERM --collision_terminate_penalty ${COLLPEN:-20}"
        OUT="${OUT}_ct${COLLTERM}"
    fi
    if [ -n "${BACKCOST:-}" ]; then
        BC_ARGS="$BC_ARGS --backward_cost $BACKCOST"
        OUT=${OUT}_bk${BACKCOST}
    fi
    # MOTIONFOOT=1: footprint scores along the commanded motion direction;
    # reversing prices as worst-case terrain (unseen ground) instead of free.
    if [ "${MOTIONFOOT:-0}" = "1" ]; then
        BC_ARGS="$BC_ARGS --footprint_along_motion"
        OUT=${OUT}_mf
    fi
    if [ "${NOBC:-0}" = "1" ]; then
        OUT=${OUT}_noBC
    elif [ -n "${DEMOS_FILE:-}" ]; then
        OUT=${OUT}_bc2
    fi
    if [ "${SMOKE:-0}" = "1" ]; then
        OUT=${OUT}_SMOKE
        STEPS=10000
    fi
    # Run C (2026-08-23): SCENES="s1 s2 ..." trains ONE policy over several
    # cached scenes (round-robin per episode; cache loaded per scene from the
    # same root). RAM scales with scene count -> submit with --mem=96G.
    if [ -n "${SCENES:-}" ]; then
        BC_ARGS="$BC_ARGS --scenes $SCENES"
        OUT=${OUT}_multi$(echo "$SCENES" | wc -w | tr -d ' ')
    fi
elif [ "${RUNG7B:-0}" = "1" ]; then
    # 7b = 7 with the KL leash OFF (it truncated 77% of update rounds).
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70 --goal_min_sep 1.5 --target_kl 0 --scenes rugd_trail_00 rugd_park-1_00 rugd_park-2_00 rugd_trail-4_00"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v7b_multiscene4_unleashed
    STEPS=800000
elif [ "${RUNG7:-0}" = "1" ]; then
    # v7 = multi-scene: the proven 6d recipe (random goals, min_sep 1.5) over
    # four scenes at once. LRU cache eviction keeps GPU residency bounded.
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70 --goal_min_sep 1.5 --scenes rugd_trail_00 rugd_park-1_00 rugd_park-2_00 rugd_trail-4_00"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v7_multiscene4
    STEPS=800000
elif [ "${RUNG6D:-0}" = "1" ]; then
    # v6d = v6c with min_sep back at v6's 1.5 (the one env difference from the
    # only config that ever learned). Paired with a 6c seed-rerun, this splits
    # "min_sep poisoned it" from "PPO seed variance".
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70 --goal_min_sep 1.5"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v6d_minsep15_trail00
    STEPS=400000
elif [ "${RUNG6C:-0}" = "1" ]; then
    # v6c = v6b curriculum (random goals 15-70, min_sep 1.0) at CONSTANT lr,
    # target_kl as the only collapse protection (v6b post-mortem: decay -> 0%).
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v6c_randomgoal_trail00
    STEPS=400000
elif [ "${RUNG6B:-0}" = "1" ]; then
    # v6b = v6 recipe + close-range exposure (min_sep 1.0) + collapse protection
    # (lr decay + target_kl). Code-side changes; only the out dir differs here.
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v6b_randomgoal_trail00
    STEPS=400000
elif [ "${RUNG6:-0}" = "1" ]; then
    # random goals (15-70) + spawns over the whole trail; harder task -> 2x steps
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v6_randomgoal_trail00
    STEPS=400000
elif [ "${RUNG5:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --spawn_max_frame 3"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v5_traverse_trail00
fi
if [ "$SEED" != "0" ]; then
    BC_ARGS="$BC_ARGS --seed $SEED"
    OUT="${OUT}_seed${SEED}"
fi
# CHUNK: trajectory arm (plan B) — policy outputs CHUNK action pairs
# per decision, re-observes only after the chunk executes. 1/unset = per-action.
if [ -n "${CHUNK:-}" ] && [ "${CHUNK}" != "1" ]; then
    BC_ARGS="$BC_ARGS --action_chunk $CHUNK"
    OUT="${OUT}_chunk${CHUNK}"
fi
# PROX: proximity-cost weight (geometric stay-away term; needs scene clouds
# from dump_scene_cloud.py under outputs/scene_clouds/clouds). The clearance
# fix motivated by Run A's grazing exploit.
if [ -n "${PROX:-}" ]; then
    BC_ARGS="$BC_ARGS --proximity_weight $PROX --proximity_margin ${PROXMARGIN:-0.6} --clouds_dir /scratch/m000204-pm06b/joana/outputs/scene_clouds/clouds"
    OUT="${OUT}_prox${PROX}"
fi
# TARGETKL: override the PPO KL leash (0 = off). Multi-scene and warm-start
# runs need 0 — the leash aborts every update when the policy must shift.
if [ -n "${TARGETKL:-}" ]; then
    BC_ARGS="$BC_ARGS --target_kl $TARGETKL"
fi
# WARMSTART: continue training an existing policy (.zip). Works with any rung
# (live fine-tune of the cached champion = LIVE=1 WARMSTART=<champion ckpt>).
# STEPS then counts the ADDITIONAL steps on top of the checkpoint's counter.
if [ -n "${WARMSTART:-}" ]; then
    BC_ARGS="$BC_ARGS --warmstart $WARMSTART"
    OUT="${OUT}_warm"
fi
# STEPS_OVERRIDE: extend any rung without a new branch (v6d was still climbing
# at its 400k cap -> 800k continuation). Output dir gets the step count.
if [ -n "${STEPS_OVERRIDE:-}" ]; then
    STEPS=$STEPS_OVERRIDE
    OUT="${OUT}_${STEPS}"
fi
echo "==> rung: ${BC_ARGS:-pure-shaped}  steps: $STEPS  out: $OUT"
python scripts/train_ppo_real.py \
    --scene rugd_trail_00 \
    --clips_dir /scratch/m000204-pm06b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm06b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels \
    --total_steps $STEPS \
    --output_dir "$OUT" \
    $BC_ARGS \
    --use_wandb

echo "==> done: $OUT (rollout.mp4 + curves)"
