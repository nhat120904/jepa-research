#!/usr/bin/env bash
#SBATCH --job-name=ew_round2
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Round 2 of the full 4x5 method: refined code -> events (count-based vocabulary) -> planner (cost-to-go
# width 2048 + xor) -> two skills (tau-conditioned; tau-free on the last 16 frames) -> closed-loop arms.
#   PART=train  sbatch slurm/round2.sh          (events, planner, skills)
#   PART=loop OUT=<train run dir> sbatch slurm/round2.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=visual-puzzle-4x5-play-v0; CACHE=$RUN_ROOT/cache/$ENV
CODE=${CODE:-$RUN_ROOT/refine_56951_visual-puzzle-4x5-play-v0/code.pt}
TASKS=$RUN_ROOT/render_56942/torch_egl/tasks_4x5.npz
OUT=${OUT:-$RUN_ROOT/round2_${SLURM_JOB_ID}}
record_source "$OUT/src_${SLURM_JOB_ID}"
case ${PART:?PART} in
  train)
    "$TORCH_PY" "$PROJECT/scripts/build_events.py" --cache "$CACHE" --env "$ENV" --code "$CODE" --out "$OUT/events"
    "$TORCH_PY" "$PROJECT/scripts/train_planner.py" --events "$OUT/events" --cache "$CACHE" --env "$ENV" --code "$CODE" \
      --tasks "$TASKS" --h-xor --h-steps ${HSTEPS:-150000} --out "$OUT/planner"
    "$TORCH_PY" "$PROJECT/scripts/train_skill.py" --cache "$CACHE" --events "$OUT/events" --code "$CODE" --out "$OUT/skill_tau"
    "$TORCH_PY" "$PROJECT/scripts/train_skill.py" --cache "$CACHE" --events "$OUT/events" --code "$CODE" \
      --no-tau --max-tau 15 --out "$OUT/skill_notau15"
    ;;
  loop)
    for ARM in ${ARMS:-"skill_notau15 skill learned" "skill_tau skill learned" "skill_notau15 scripted learned" "skill_notau15 skill oracle" "skill_tau skill oracle"}; do
      set -- $ARM
      "$TORCH_PY" "$PROJECT/scripts/closed_loop.py" --env "$ENV" --planner "$OUT/planner/planner.pt" --skill "$OUT/$1/skill.pt" \
        --low "$2" --high "$3" --episodes "${EPISODES:-6}" --workers 14 --out "$OUT/loop_${1}_${2}_${3}" ${LOOP_ARGS:-}
    done
    ;;
esac
