#!/usr/bin/env bash
#SBATCH --job-name=drive-prev
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --exclude=n04,n13,n17,n21,n24
#SBATCH --output=/scratch/m000204-pm06b/joana/slurm-drive-prev-%j.out
#SBATCH --error=/scratch/m000204-pm06b/joana/slurm-drive-prev-%j.err

# Moving-through-the-world diffusion preview: consecutive recorded poses,
# any resolution/steps. HEIGHT/WIDTH multiples of 112. Knobs:
#   HEIGHT WIDTH NSTEPS FRAMES SCENE START TARGET LIVECKPT
#   GOALFRAME (goal = pose at this frame) / GOALXY "x,y" — goal marker in
#   both panels + topdown inset + distance HUD (goal-placement design tool)

set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm06b/joana/nav-rl

python scripts/drive_preview.py \
    --scene "${SCENE:-rugd_trail_00}" \
    --clips_dir /scratch/m000204-pm06b/joana/data/rugd_clips \
    --poses_dir /scratch/m000204-pm06b/joana/outputs/poses \
    --labels_dir /scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels_v14 \
    --height "${HEIGHT:-336}" \
    --width "${WIDTH:-560}" \
    --num_steps "${NSTEPS:-4}" \
    --frames "${FRAMES:-40}" \
    --start "${START:-5}" \
    --heading "${HEADING:-tangent}" \
    ${TARGET:+--target_xy "$TARGET"} \
    ${GOALFRAME:+--goal_frame "$GOALFRAME"} \
    ${GOALXY:+--goal_xy "$GOALXY"} \
    ${SPIN:+--spin} \
    ${SPINDEG:+--spin_deg "$SPINDEG"} \
    --live_ckpt "${LIVECKPT:-/scratch/m000204-pm06b/joana/runs/train_semantic_v10/checkpoint-epoch-30.safetensors}" \
    --out /scratch/m000204-pm06b/joana/outputs/drive_preview

echo "==> drive preview done"
