#!/usr/bin/env bash
# One place that knows how to launch an eval, so nothing gets dropped by hand.
#
#   bash scripts/submit_eval.sh <trainjobid> [scene ...] [--dry]
#   bash scripts/submit_eval.sh 499460 --ckpt 40000        # a specific checkpoint
#   bash scripts/submit_eval.sh 499460 --all               # all 5 scenes, not just the core 3
#
# Defaults to the 3-scene core set. Reads goal + spawn bound from configs/eval_scenes.env.
set -euo pipefail
cd "$(dirname "$0")/.."
. configs/eval_scenes.env

JOB=${1:?usage: submit_eval.sh <trainjobid> [scenes...] [--ckpt N] [--dry] [--all]}; shift
DRY=0; CKPT_WANT=""; SCENES=""
while [ $# -gt 0 ]; do
    case "$1" in
        --dry)  DRY=1; shift;;
        --all)  SCENES="$(awk 'NF && $1 !~ /^#/ {print $1}' <<< "$EVAL_SCENES")"; shift;;
        --ckpt) CKPT_WANT=$2; shift 2;;
        *)      SCENES="$SCENES $1"; shift;;
    esac
done
[ -z "${SCENES// }" ] && SCENES="$EVAL_CORE"

OUT=/scratch/m000204-pm06b/joana/outputs
SEM=${LIVECKPT:-/scratch/m000204-pm06b/joana/runs/train_semantic_v35b_p6_ppl5/checkpoint-epoch-3.safetensors}
D=$(ls -d $OUT/*_j${JOB} 2>/dev/null | head -1) || true
[ -n "${D:-}" ] || { echo "REFUSED: no run directory for job $JOB"; exit 1; }
if [ -n "$CKPT_WANT" ]; then
    C=$D/checkpoints/ppo_${CKPT_WANT}_steps.zip
else
    C=$D/checkpoints/$(cd "$D/checkpoints" && ls ppo_*_steps.zip | sort -t_ -k2 -n | tail -1)
fi
[ -f "$C" ] || { echo "REFUSED: no checkpoint $C"; exit 1; }
CUR=$(python3 -c "import json;print(json.load(open('$D/curriculum_state.json'))['goal_dist'])" 2>/dev/null || echo "none")

echo "=== eval job $JOB   $(basename "$C")   curriculum reached ${CUR} m"
for S in $SCENES; do
    LINE=$(awk -v s="$S" '$1==s {print; exit}' <<< "$EVAL_SCENES")
    [ -n "$LINE" ] || { echo "REFUSED: $S is not in configs/eval_scenes.env"; exit 1; }
    GOAL=$(awk '{print $2}' <<< "$LINE"); SPAWN=$(awk '{print $3}' <<< "$LINE")
    unset GOAL_FRAME GOALFRAMERANGE SPAWN_MAX SPAWNMAX   # never leak between scenes
    export LIVE=1 LIVECKPT="$SEM" CKPT="$C" SCENE="$S" EPISODES="${EPISODES:-8}" \
           CLIPS_DIR=/scratch/m000204-pm06b/joana/data/campus_clips \
           MAXSTEPSPERM="${MAXSTEPSPERM:-16}" MAXSTEPSBASE="${MAXSTEPSBASE:-40}"
    eval "export $GOAL"
    [ "$SPAWN" != "-" ] && eval "export $SPAWN"
    printf "    %-13s %-22s %s\n" "$S" "$GOAL" "${SPAWN/-/spawn unbounded (goal range, safe)}"
    [ "$DRY" = "1" ] && continue
    sbatch --export=ALL scripts/slurm/eval_policy.sh >/dev/null 2>&1 \
        && echo "        submitted" || echo "        LIMIT REACHED -- retry later"
done
[ "$DRY" = "1" ] && echo "    (dry run: nothing submitted)"
