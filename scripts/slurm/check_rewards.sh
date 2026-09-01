#!/usr/bin/env bash
#SBATCH --job-name=check-rew
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --exclude=n04,n06,n13,n14,n17,n21,n24,n26,n30,n31
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-check-rew-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-check-rew-%j.err

# Reward mechanism check: GATE / VOID / IMGVOID measured on real renders.
# Renders ONCE, then sweeps the alpha-gate threshold over identical pixels.
# Knobs: SCENE, TRAV, EPISODES, STEPS, SWEEP, GATETAU, GOALXY, SPAWNFRAME, TAG.

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm06b/joana/nav-rl

EXTRA=()
if [ -n "${GOALXY:-}" ]; then
    EXTRA+=(--goal_xy "${GOALXY}")
fi
if [ -n "${SPAWNFRAME:-}" ]; then
    EXTRA+=(--spawn_frame "${SPAWNFRAME}")
fi
if [ -n "${TAG:-}" ]; then
    EXTRA+=(--tag "${TAG}")
fi

python scripts/check_rewards.py \
    --scene "${SCENE:-gnd_AUw360}" \
    --trav_path "${TRAV:-config/traversability_v14_walkway.yaml}" \
    --episodes "${EPISODES:-4}" \
    --steps "${STEPS:-8}" \
    --sweep "${SWEEP:-0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8}" \
    --gate_tau "${GATETAU:-0.5}" \
    --walk "${WALK:-straight}" \
    --live_ckpt "${LIVECKPT:-/scratch/m000204-pm06b/joana/runs/train_semantic_v21/checkpoint-epoch-12.safetensors}" \
    "${EXTRA[@]}"

echo "==> reward check done"
