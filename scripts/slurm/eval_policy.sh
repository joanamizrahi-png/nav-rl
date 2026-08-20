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
#SBATCH --exclude=n04,n13,n17,n24
set -euo pipefail
module load conda/24.3.0-0
module load cuda12.9/toolkit/12.9.1
export PATH=/users/jmizrahi/.conda/envs/neoverse/bin:$PATH
export PYTHONNOUSERSITE=1
hash -r
cd /scratch/m000204-pm06b/joana/nav-rl
CKPT=${CKPT:?set CKPT=/path/to/checkpoint.zip via --export=CKPT=...}
# eval dir carries the run name (ppo_v4_... vs ppo_v5_...) so evals never collide
RUN_NAME=$(basename "$(dirname "$(dirname "$CKPT")")")
EXTRA_ARGS=()
if [[ "${SPAWN_MAX:-}" != "" ]]; then
    EXTRA_ARGS+=(--spawn_max_frame "$SPAWN_MAX")   # match the training rung
fi
OUT_SUFFIX=""
if [[ "${GOAL_FRAME:-}" != "" ]]; then
    EXTRA_ARGS+=(--goal_frame "$GOAL_FRAME")       # generalization: goal the policy never trained on
    OUT_SUFFIX="_goal${GOAL_FRAME}"
fi
if [[ "${COLLTERM:-}" != "" ]]; then
    EXTRA_ARGS+=(--collision_terminate_frac "$COLLTERM"
                 --collision_terminate_penalty "${COLLPEN:-20}")
    OUT_SUFFIX="${OUT_SUFFIX}_ct${COLLTERM}"
fi
if [[ "${MAX_STEPS:-}" != "" ]]; then
    EXTRA_ARGS+=(--max_steps "$MAX_STEPS")   # GND/SCAND need a longer budget
    OUT_SUFFIX="${OUT_SUFFIX}_s${MAX_STEPS}"
fi
if [[ "${GOAL_XY:-}" != "" ]]; then
    # designed obstacle test: GOAL_XY="x,y" pins the goal off-trajectory
    # (e.g. behind a tree). GOAL_FRAME still caps the spawn range.
    EXTRA_ARGS+=(--goal_xy "$GOAL_XY")
    OUT_SUFFIX="${OUT_SUFFIX}_gxy${GOAL_XY/,/_}"
fi
SCENE=${SCENE:-rugd_trail_00}
if [ "$SCENE" != "rugd_trail_00" ]; then
    OUT_SUFFIX="${OUT_SUFFIX}_${SCENE}"    # zero-shot evals get their own dir
fi
# Cached-obs policies must be evaluated on the SAME cache they trained on.
#   OBS_CACHE=ribbon_cache | ribbon_cache_spin ...   NOGATE=1 for ungated runs
LABELS_DIR=/scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels
if [[ "${OBS_CACHE:-}" != "" ]]; then
    CACHE_PATHS=$(echo "$OBS_CACHE" | sed 's#[^,]*#/scratch/m000204-pm06b/joana/outputs/&#g')
    EXTRA_ARGS+=(--obs_cache "${CACHE_PATHS}"
                 --trav_path config/traversability_v14.yaml)
    LABELS_DIR=/scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels_v14
    if [[ "$OBS_CACHE" == *,* ]]; then
        OUT_SUFFIX="${OUT_SUFFIX}_hybrid"
    else
        OUT_SUFFIX="${OUT_SUFFIX}_${OBS_CACHE#ribbon_cache}"
    fi
fi
[[ "${NOGATE:-0}" == "1" ]] && EXTRA_ARGS+=(--no_alpha_gate)
if [[ "${VIDEOS:-}" != "" ]]; then EXTRA_ARGS+=(--videos "$VIDEOS"); fi
echo "==> eval: $RUN_NAME / $(basename "$CKPT") scene=$SCENE spawn_max=${SPAWN_MAX:-default} goal=${GOAL_FRAME:-train(30)}"
python scripts/eval_policy.py \
    --checkpoint "$CKPT" \
    --scene "$SCENE" --episodes 20 \
    --clips_dir "${CLIPS_DIR:-/scratch/m000204-pm06b/joana/data/rugd_clips}" \
    --poses_dir /scratch/m000204-pm06b/joana/outputs/poses \
    --labels_dir $LABELS_DIR \
    --out_dir /scratch/m000204-pm06b/joana/outputs/eval_${RUN_NAME}_$(basename "$CKPT" .zip)${OUT_SUFFIX} \
    "${EXTRA_ARGS[@]}"
echo "==> eval done"
