#!/usr/bin/env bash
#SBATCH --job-name=report-card
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-report-card-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-report-card-%j.err

# Reconstruction report card: replay fidelity + coverage-vs-offset per scene.
#   sbatch --export=SCENE=rugd_trail_00 scripts/slurm/report_card.sh
# Results: outputs/report_card/<scene>.json (+ printed verdict in this log).

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm06b/joana/nav-rl
echo "commit: $(git log --oneline -1)"
SCENE=${SCENE:?set SCENE=... via --export}
python scripts/scene_report_card.py --scene "$SCENE" \
    --clips_dir /scratch/m000204-pm06b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm06b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels \
    --model_path /scratch/m000204-pm06b/joana/NeoVerse/models \
    --reconstructor_path /scratch/m000204-pm06b/joana/NeoVerse/models/NeoVerse/reconstructor.ckpt \
    --out_dir /scratch/m000204-pm06b/joana/outputs/report_card
echo "==> report card done for $SCENE"
