#!/usr/bin/env bash
#SBATCH --job-name=ew_pipe
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Events -> event WM + cost-to-go (+ offline checks) -> skill -> closed-loop smoke.
#   CODE=<code.pt> ENV=visual-puzzle-4x5-play-v0 INV=<inventory dir> sbatch slurm/pipeline.sh
#   STAGES="events planner skill loop" selects stages; EPISODES = closed-loop episodes per task.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=${ENV:-visual-puzzle-4x5-play-v0}; SIZE=$(echo "$ENV" | cut -d- -f3)
CODE=${CODE:?CODE}; INV=${INV:-$RUN_ROOT/render_56942/torch_egl}  # 56924 task frames are black (CPU-node OSMesa)
OUT=${OUT:-$RUN_ROOT/pipe_${SLURM_JOB_ID}_${ENV}}
CACHE=$RUN_ROOT/cache/$ENV
# Stage directories can point at an earlier run's outputs (reuse without overwriting).
EV=${EVENTS_DIR:-$OUT/events}; PL=${PLAN_DIR:-$OUT/planner}; SK=${SKILL_DIR:-$OUT/skill}
record_source "$OUT"
STAGES=${STAGES:-events planner skill loop}
for S in $STAGES; do
  case $S in
    events)  "$TORCH_PY" "$PROJECT/scripts/build_events.py" --cache "$CACHE" --env "$ENV" --code "$CODE" --out "$EV" ;;
    planner) "$TORCH_PY" "$PROJECT/scripts/train_planner.py" --events "$EV" --cache "$CACHE" --env "$ENV" --code "$CODE" \
               --tasks "$INV/tasks_${SIZE}.npz" --out "$PL" ${PLANNER_ARGS:-} ;;
    skill)   "$TORCH_PY" "$PROJECT/scripts/train_skill.py" --cache "$CACHE" --events "$EV" --code "$CODE" --out "$SK" ${SKILL_ARGS:-} ;;
    loop)    for ARM in ${ARMS:-"skill learned" "scripted learned" "skill oracle"}; do
               set -- $ARM
               "$TORCH_PY" "$PROJECT/scripts/closed_loop.py" --env "$ENV" --planner "$PL/planner.pt" --skill "$SK/skill.pt" \
                 --low "$1" --high "$2" --episodes "${EPISODES:-2}" --out "$OUT/loop_${1}_${2}" ${LOOP_ARGS:-}
             done ;;
  esac
done
