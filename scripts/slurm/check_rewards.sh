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

# Reward mechanism check: GATE / VOID / IMGVOID measured on real renders,
# gated vs ungated over identical poses. Knobs: SCENE, TRAV, EPISODES, STEPS.

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm06b/joana/nav-rl

python scripts/check_rewards.py \
    --scene "${SCENE:-gnd_AUw360}" \
    --trav_path "${TRAV:-config/traversability_v14_walkway.yaml}" \
    --episodes "${EPISODES:-4}" \
    --steps "${STEPS:-8}" \
    --live_ckpt "${LIVECKPT:-/scratch/m000204-pm06b/joana/runs/train_semantic_v21/checkpoint-epoch-12.safetensors}"

echo "==> reward check done"
