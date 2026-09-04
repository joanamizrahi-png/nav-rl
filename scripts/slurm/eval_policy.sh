#!/usr/bin/env bash
#SBATCH --job-name=eval-policy
#SBATCH --account=marlowe-m000204-pm06b
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
# 01:00:00 loses the run to the PIPELINE LOAD. A live eval spends 25-40 min
# loading the diffusion pipeline before the first episode, so a 1 h wall leaves
# ~20 min for 20 episodes and job 463931 (2026-09-03) timed out having written
# videos but no metrics -- while its BLIND partner, which happened to load
# faster, completed. A pair where only one half survives is worse than no pair.
#SBATCH --time=2:00:00
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
# 2026-09-03: jobs 464831/464832 ran the cloud RASTERIZER instead of the world
# model because the launch was `env $EV ...` from a shell where EV was empty
# (an export does not survive a new login). They finished in one minute,
# reported mean_coverage None and raw label ids -- and read like a result.
# A policy trained on live observations is evaluated on live observations,
# and a live eval always names the semantics checkpoint the policy trained on.
if [[ "$CKPT" == *ppo_live_* && "${LIVE:-0}" != "1" ]]; then
    echo "REFUSED: $(basename "$(dirname "$(dirname "$CKPT")")") trained on live observations but LIVE is not 1."
    echo "         Did the EV export survive this shell? Check: echo \"\$EV\" | wc -w   (expect about 30)"
    exit 2
fi
if [[ "${LIVE:-0}" == "1" && -z "${LIVECKPT:-}" ]]; then
    echo "REFUSED: LIVE=1 without LIVECKPT. Name the semantics checkpoint the policy trained on."
    exit 2
fi
# eval dir carries the run name (ppo_v4_... vs ppo_v5_...) so evals never collide
RUN_NAME=$(basename "$(dirname "$(dirname "$CKPT")")")
# 2026-09-02: eval_<run>_<ckpt><flags> blew past the 255-byte filename limit
# (OSError 36) and every eval of the new arms would have failed the same way --
# their names are longer still. Drop the tokens that are identical on EVERY run
# and therefore distinguish nothing, then hard-cap as a backstop. The full name
# stays in the log line above, so nothing is lost.
RUN_SHORT=$(echo "$RUN_NAME" \
    | sed -e 's/ppo_live_trail00_UNGATED_//' -e 's/ppo_live_trail00_//' \
          -e 's/_r336x224//' -e 's/_rr560x336//' -e 's/_smin10//' \
          -e 's/_spcls//' -e 's/_gc50//' -e 's/_sjy20//' -e 's/_sjl0\.4//' \
          -e 's/_pal4//' -e 's/v21obs_//' -e 's/_trstrict//' -e 's/_semw5//' \
          -e 's/_rs0\.01//' -e 's/_g360//')
RUN_SHORT=${RUN_SHORT:0:110}
EXTRA_ARGS=()
# Accept the training spellings too. train_ppo_real.sh uses SPAWNMIN/SPAWNMAX,
# this script used SPAWN_MIN/SPAWN_MAX, and a mismatched name is silently
# ignored -- the same trap as MAXSTEPS vs MAX_STEPS (2026-09-03).
SPAWN_MAX="${SPAWN_MAX:-${SPAWNMAX:-}}"
SPAWN_MIN="${SPAWN_MIN:-${SPAWNMIN:-}}"
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
EXTRA_ARGS+=(--sem_palette "${SEMPAL:-4}")
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
[ -n "${SPAWNJYAW:-}" ] && EXTRA_ARGS+=(--spawn_yaw_jitter "$SPAWNJYAW")
[ -n "${SPAWNJLAT:-}" ] && EXTRA_ARGS+=(--spawn_lat_jitter "$SPAWNJLAT")
[ -n "${VOIDTERM:-}" ]  && EXTRA_ARGS+=(--void_terminate_frac "$VOIDTERM")
[ -n "${HALT:-}" ] && EXTRA_ARGS+=(--halt_terminate_steps "$HALT")
[ -n "${HALTSCALE:-}" ] && EXTRA_ARGS+=(--halt_penalty_scale "$HALTSCALE")
[ -n "${HALTEPS:-}" ] && EXTRA_ARGS+=(--halt_throttle_eps "$HALTEPS")
[ "${SPEEDCOST:-0}" = "1" ] && EXTRA_ARGS+=(--terrain_speed_scaled)
[ -n "${REWSRC:-}" ] && { EXTRA_ARGS+=(--reward_source "$REWSRC"); OUT_SUFFIX="${OUT_SUFFIX}_r${REWSRC/map_then_generated/hyb}"; }
[ -n "${MAPINFL:-}" ] && { EXTRA_ARGS+=(--map_inflate_m "$MAPINFL"); OUT_SUFFIX="${OUT_SUFFIX}_mi${MAPINFL}"; }
[ -n "${MAPINFLCLS:-}" ] && { EXTRA_ARGS+=(--map_inflate_classes "$MAPINFLCLS"); OUT_SUFFIX="${OUT_SUFFIX}_mic${MAPINFLCLS//,/-}"; }
[ -n "${GOALMIX:-}" ] && { EXTRA_ARGS+=(--goal_traversable_mix "$GOALMIX"); OUT_SUFFIX="${OUT_SUFFIX}_gm${GOALMIX}"; }
[ -n "${SPAWNSUPPORT:-}" ] && { EXTRA_ARGS+=(--spawn_support_tries "$SPAWNSUPPORT"); OUT_SUFFIX="${OUT_SUFFIX}_ss${SPAWNSUPPORT}"; }

# Reward weights, so the reported `return=` is on TRAINING's scale. They do not
# change a frozen policy's actions, but a return computed with goal_bonus 50
# against training's 1000 is not comparable to anything.
[ -n "${GOALWEIGHT:-}" ]  && EXTRA_ARGS+=(--goal_weight "$GOALWEIGHT")
[ -n "${COLLWEIGHT:-}" ]  && EXTRA_ARGS+=(--collision_weight "$COLLWEIGHT")
[ -n "${GOALBONUS:-}" ]   && EXTRA_ARGS+=(--goal_bonus "$GOALBONUS")
[ -n "${TIMEOUTPEN:-}" ]  && EXTRA_ARGS+=(--timeout_penalty "$TIMEOUTPEN")
[ "${TIMEOUTDIST:-0}" = "1" ] && EXTRA_ARGS+=(--timeout_distance_scaled)
[ -n "${PROXW:-}" ]       && EXTRA_ARGS+=(--proximity_weight "$PROXW")
[ -n "${PROXMARGIN:-}" ]  && EXTRA_ARGS+=(--proximity_margin "$PROXMARGIN")
[ "${PROXDELTA:-0}" = "1" ] && EXTRA_ARGS+=(--proximity_delta)
[ -n "${VOIDCOST:-}" ]    && EXTRA_ARGS+=(--void_cost "$VOIDCOST")
[ -n "${STEPCOST:-}" ]    && EXTRA_ARGS+=(--step_cost "$STEPCOST")

# GOALSUPPORT: reject goals with no reconstruction under them, exactly as
# training does. It existed in check_rewards.sh but NOT here, so every eval
# before 2026-09-03 12:45 sampled goals with the support check OFF (default
# 0.0) while training ran 0.6 -- about 14.5% of eval goals had no world under
# them and could never be reached.
[ -n "${GOALSUPPORT:-}" ] && EXTRA_ARGS+=(--goal_support_radius "$GOALSUPPORT")
[ -n "${GOALSUPPORTFRAC:-}" ] && EXTRA_ARGS+=(--goal_support_min_frac "$GOALSUPPORTFRAC")

# The SEMANTICS MODEL goes in the output name. Two evals of the same policy
# checkpoint under different world models (v21 vs v26) resolved to the same
# directory and the second silently overwrote the first (2026-09-03, jobs
# 464345 -> 464601). Identity = the run directory the checkpoint came from.
if [[ -n "${LIVECKPT:-}" ]]; then
    _semid="$(basename "$(dirname "$LIVECKPT")" | sed 's/^train_semantic_//')_$(basename "$LIVECKPT" .safetensors | sed 's/checkpoint-//; s/epoch-/e/')"
    OUT_SUFFIX="${OUT_SUFFIX}_${_semid}"
fi

# COH / COHTAU / COHTERM: coherence termination, matching training. Without
# these the eval never ends an episode for leaving the reconstructed world,
# while training does -- which is exactly the train/eval mismatch that made
# five evals report ZERO crashes on 2026-09-03.
if [[ -n "${COH:-}" || -n "${COHTERM:-}" ]]; then
    EXTRA_ARGS+=(--coherence_cost_weight "${COH:-0}"
                 --coherence_tau "${COHTAU:-0.4}"
                 --coherence_terminate_tau "${COHTERM:-0}"
                 --coherence_terminate_penalty "${COHPEN:-100}")
    OUT_SUFFIX="${OUT_SUFFIX}_coh${COH:-0}t${COHTAU:-0.4}ct${COHTERM:-0}"
fi
# 2026-09-03 22:20: a radius-1.0 diagnostic OVERWROTE the radius-0.75 eval of
# the same checkpoint because none of these knobs reached the directory name.
# Every knob that changes the task now does.
[ -n "${GOALRADIUS:-}" ] && { EXTRA_ARGS+=(--goal_radius "$GOALRADIUS"); OUT_SUFFIX="${OUT_SUFFIX}_r${GOALRADIUS}"; }
[ -n "${HALT:-}" ] && OUT_SUFFIX="${OUT_SUFFIX}_halt${HALT}"
[ -n "${HALTEPS:-}" ] && OUT_SUFFIX="${OUT_SUFFIX}_he${HALTEPS}"
[ -n "${HALTSCALE:-}" ] && OUT_SUFFIX="${OUT_SUFFIX}_hs${HALTSCALE}"
[ "${SPEEDCOST:-0}" = "1" ] && OUT_SUFFIX="${OUT_SUFFIX}_spd"
[ -n "${COLLPEN:-}" ] && [ "${COLLPEN}" != "1000" ] && OUT_SUFFIX="${OUT_SUFFIX}_cp${COLLPEN}"
[ -n "${SEMW:-}" ]       && EXTRA_ARGS+=(--semantic_weight "$SEMW")
[ -n "${REWSCALE:-}" ]   && EXTRA_ARGS+=(--reward_scale "$REWSCALE")
[ -n "${SMOOTHCOST:-}" ] && EXTRA_ARGS+=(--action_smooth_cost "$SMOOTHCOST")
if [[ "${COLLTERM:-}" != "" ]]; then
    EXTRA_ARGS+=(--collision_terminate_frac "$COLLTERM"
                 --collision_terminate_penalty "${COLLPEN:-20}")
    OUT_SUFFIX="${OUT_SUFFIX}_ct${COLLTERM}"
fi
# Accept MAXSTEPS as well as MAX_STEPS. train_ppo_real.sh spells it MAXSTEPS,
# this script spelled it MAX_STEPS, and nothing warned -- so on 2026-09-03 six
# evals were launched with MAXSTEPS=90 and silently ran the DEFAULT 60 against
# training's 90. The ancestor's headline result (19/20 TIMEOUT) was measured on
# a two-thirds episode budget; several of those episodes had closed 50-64% of
# the distance when the clock ran out.
MAX_STEPS="${MAX_STEPS:-${MAXSTEPS:-}}"
if [[ "${MAX_STEPS:-}" != "" ]]; then
    EXTRA_ARGS+=(--max_steps "$MAX_STEPS")
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
    --scene "$SCENE" --episodes "${EPISODES:-20}" \
    --clips_dir "${CLIPS_DIR:-/scratch/m000204-pm06b/joana/data/rugd_clips}" \
    --poses_dir /scratch/m000204-pm06b/joana/outputs/poses \
    --labels_dir $LABELS_DIR \
    --out_dir /scratch/m000204-pm06b/joana/outputs/eval_${RUN_SHORT}_$(basename "$CKPT" .zip)${OUT_SUFFIX} \
    "${EXTRA_ARGS[@]}"
echo "==> eval done"
