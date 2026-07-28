#!/usr/bin/env bash
#SBATCH --job-name=eval-policy
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-eval-policy-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-eval-policy-%j.err
set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm06b/joana/nav-rl
CKPT=${CKPT:?set CKPT=/path/to/checkpoint.zip via --export=CKPT=...}
python scripts/eval_policy.py \
    --checkpoint "$CKPT" \
    --scene rugd_trail_00 --episodes 20 \
    --clips_dir /scratch/m000204-pm06b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm06b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels \
    --out_dir /scratch/m000204-pm06b/joana/outputs/eval_$(basename "$CKPT" .zip)
echo "==> eval done"
