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
DEMOS=/scratch/m000204-pm06b/joana/outputs/demos_v1.npz
if [ -f "$DEMOS" ]; then
    BC_ARGS="--bc_demos $DEMOS"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v4_cost_trail00
else
    BC_ARGS=""
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v2_shaped_trail00
fi
STEPS=200000
SEED=${SEED:-0}
if [ "${CACHE:-0}" = "1" ]; then
    # v14-DIFFUSED (2026-08-15, advisor's headline ask): the 6d recipe on the
    # corrected right-handed frame, observations from the ribbon cache (v10 +
    # reader diffused views), reward from the cache's alpha-masked diffused
    # labels scored by the v14 table. Requires cache_gen.sh to have populated
    # outputs/ribbon_cache/<scene>. SMOKE=1 -> 10k-step gate run.
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70 --goal_min_sep 1.5 --obs_cache /scratch/m000204-pm06b/joana/outputs/ribbon_cache --trav_path config/traversability_v14.yaml --labels_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels_v14"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v14diff_trail00
    STEPS=800000
    if [ "${NOGATE:-0}" = "1" ]; then
        BC_ARGS="$BC_ARGS --no_alpha_gate"
        OUT=${OUT}_UNGATED
    fi
    if [ "${SMOKE:-0}" = "1" ]; then
        OUT=${OUT}_SMOKE
        STEPS=10000
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
