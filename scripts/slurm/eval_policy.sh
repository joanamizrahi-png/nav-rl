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
# SPAWN_MIN pins the near end of the spawn range. Essential with GOAL_XY: the
# goal is a fixed world point, so without a floor the spawns spread back to the
# start of the path and most episodes run out of steps before reaching it.
# Kinematics + reverse clamp must mirror the training rung or the policy is
# evaluated outside the action model it learned (2026-09-01).
if [[ "${STEP:-}" != "" ]]; then
    EXTRA_ARGS+=(--step_size_m "$STEP")
fi
if [[ "${YAW:-}" != "" ]]; then
    EXTRA_ARGS+=(--yaw_step_rad "$YAW")
fi
if [[ "${FWDONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--forward_only)
fi
if [[ "${SPAWN_MIN:-}" != "" ]]; then
    EXTRA_ARGS+=(--spawn_min_frame "$SPAWN_MIN")
fi
OUT_SUFFIX=""
if [[ "${GOAL_FRAME:-}" != "" ]]; then
    EXTRA_ARGS+=(--goal_frame "$GOAL_FRAME")       # generalization: goal the policy never trained on
    OUT_SUFFIX="_goal${GOAL_FRAME}"
fi
# GOAL360 / GOALRANGE / GOALCONE / GOALFRAMERANGE: sample goals the way TRAINING
# does instead of pinning one goal_xy. Eval could previously only use a fixed
# goal or the default goal frame, so it never reproduced the distribution the
# policy learned, and with a wide spawn range d_start varied uncontrolled.
if [ "${GOAL360:-0}" = "1" ]; then
    EXTRA_ARGS+=(--goal_dir_360)
fi
if [ -n "${GOALRANGE:-}" ]; then
    EXTRA_ARGS+=(--goal_dist_range "${GOALRANGE}")
fi
if [ -n "${GOALCONE:-}" ]; then
    EXTRA_ARGS+=(--goal_cone_deg "${GOALCONE}")
fi
if [ -n "${GOALFRAMERANGE:-}" ]; then
    EXTRA_ARGS+=(--goal_frame_range "${GOALFRAMERANGE}")
    OUT_SUFFIX="${OUT_SUFFIX}_gfr${GOALFRAMERANGE/,/-}"
fi
# BLIND=1: zero the rgb observation (goal vector intact) — the does-the-policy-
# actually-look ablation. Videos still record the real frames.
if [[ "${BLIND:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--blind)
    OUT_SUFFIX="${OUT_SUFFIX}_blind"
fi
# HEIGHT/WIDTH: obs resolution — MUST match the checkpoint's training res
# (the CNN is size-locked; RW5-v2 arms train at 336x224).
if [[ -n "${HEIGHT:-}" || -n "${WIDTH:-}" ]]; then
    EXTRA_ARGS+=(--obs_height "${HEIGHT:-336}" --obs_width "${WIDTH:-560}")
    OUT_SUFFIX="${OUT_SUFFIX}_r${WIDTH:-560}x${HEIGHT:-336}"
fi
if [[ -n "${RENDERH:-}" || -n "${RENDERW:-}" ]]; then
    EXTRA_ARGS+=(--render_height "${RENDERH:-336}" --render_width "${RENDERW:-560}")
    OUT_SUFFIX="${OUT_SUFFIX}_rr${RENDERW:-560}x${RENDERH:-336}"
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
if [[ -n "${CHUNK:-}" && "${CHUNK}" != "1" ]]; then
    EXTRA_ARGS+=(--action_chunk "$CHUNK")
    OUT_SUFFIX="${OUT_SUFFIX}_chunk${CHUNK}"
fi
# MOTIONFOOT=1: match training's motion-direction footprint rule
if [[ "${MOTIONFOOT:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--footprint_along_motion)
    OUT_SUFFIX="${OUT_SUFFIX}_mf"
fi
# FWDONLY=1: match training's forward-only clamp
if [[ "${FWDONLY:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--forward_only)
    OUT_SUFFIX="${OUT_SUFFIX}_fwd"
fi
# LIVE=1: serve live diffusion observations (evals of --live-trained policies).
# ~1.4 s/step: submit with  sbatch --mem=96G --time=03:00:00
if [[ "${LIVE:-0}" == "1" ]]; then
    # TRAV: the traversability table the eval scores with. MUST match the
    # checkpoint's TRAINING table or the eval measures a different task —
    # the default v14 table scores grass 0.75 (walkable!), while the J-arms
    # train with grass 0.0. Default kept for backwards compatibility.
    EXTRA_ARGS+=(--live --trav_path "${TRAV:-config/traversability_v14.yaml}")
    echo "==> eval traversability table: ${TRAV:-config/traversability_v14.yaml}"
    if [[ -n "${LIVECKPT:-}" ]]; then
        EXTRA_ARGS+=(--live_ckpt "$LIVECKPT")
    fi
    LABELS_DIR=/scratch/m000204-pm06b/joana/NeoVerse/outputs/sam3_labels_v14
    OUT_SUFFIX="${OUT_SUFFIX}_live"
fi
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
