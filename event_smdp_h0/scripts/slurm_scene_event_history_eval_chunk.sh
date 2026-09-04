#!/usr/bin/env bash
#SBATCH --job-name=scene_event_history_eval
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_event_history_eval_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
RUN_KIND=${RUN_KIND:?submit with RUN_KIND=pilot or confirmatory}
START_INDEX=${START_INDEX:?submit with START_INDEX}
END_INDEX=${END_INDEX:?submit with END_INDEX}
case "$RUN_KIND" in
  pilot)        BASE=86000; PER_TASK=16 ;;
  confirmatory) BASE=88000; PER_TASK=64 ;;
  # The smoke band burns (task 4, 87300) and (task 5, 87400); the confirmatory
  # base was moved to 88000 so no evaluated reset shares a seed with it.
  smoke)        BASE=86900; PER_TASK=1  ;;
  *) echo "unknown RUN_KIND=$RUN_KIND" >&2; exit 2 ;;
esac
LAST=$(( 2 * PER_TASK - 1 ))
if (( START_INDEX < 0 || END_INDEX > LAST || START_INDEX > END_INDEX )); then
  echo "invalid index interval ${START_INDEX}--${END_INDEX} for ${RUN_KIND}" >&2
  exit 2
fi
RUN_ID=${RUN_ID:-${RUN_KIND}_20260904}
export STABLEWM_HOME="$RUNTIME"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} KIND=${RUN_KIND} RANGE=${START_INDEX}-${END_INDEX} RUN=$RUN_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_core.py" "$PROJECT/scene_abstract_smdp.py" \
  "$PROJECT/scene_event_history.py" \
  "$PROJECT/scripts/eval_scene_event_history.py" \
  "$PROJECT/scripts/slurm_scene_event_history_eval_chunk.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for (( INDEX=START_INDEX; INDEX<=END_INDEX; INDEX++ )); do
  if (( INDEX < PER_TASK )); then TASK=4; LOCAL=$INDEX; else TASK=5; LOCAL=$((INDEX - PER_TASK)); fi
  RESET_SEED=$((BASE + 100 * TASK + LOCAL))
  OUT="$PROJECT/outputs/scene_event_history/eval/${RUN_ID}/${INDEX}"
  echo "INDEX=$INDEX TASK=$TASK SEED=$RESET_SEED START=$(date -u +%FT%TZ)"
  "$RUNTIME/.venv/bin/python" "$PROJECT/scripts/eval_scene_event_history.py" \
    --task-id "$TASK" --reset-seed "$RESET_SEED" --observer-seeds 0,1,2 \
    --observer-root "$PROJECT/outputs/scene_event_history/checkpoints" \
    --transition-checkpoint \
      "$PROJECT/outputs/scene_h1b/checkpoints/seed0/abstract_smdp.pt" \
    --budget 112 --horizon 4 --out-dir "$OUT"
done
