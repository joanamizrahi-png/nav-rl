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
if [ "${RUNG6:-0}" = "1" ]; then
    # random goals (15-70) + spawns over the whole trail; harder task -> 2x steps
    BC_ARGS="$BC_ARGS --goal_frame_range 15 70"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v6_randomgoal_trail00
    STEPS=400000
elif [ "${RUNG5:-0}" = "1" ]; then
    BC_ARGS="$BC_ARGS --spawn_max_frame 3"
    OUT=/scratch/m000204-pm06b/joana/outputs/ppo_v5_traverse_trail00
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
