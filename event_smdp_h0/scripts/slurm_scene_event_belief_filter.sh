#!/usr/bin/env bash
#SBATCH --job-name=scene_event_belief_filter
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-31%8
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_event_belief_filter_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
INDEX=${SLURM_ARRAY_TASK_ID:?}
if (( INDEX < 16 )); then
  TASK=4
  LOCAL=$INDEX
  RESET_SEED=$((85400 + LOCAL))
else
  TASK=5
  LOCAL=$((INDEX - 16))
  RESET_SEED=$((85501 + LOCAL))
fi
RUN_ID=${RUN_ID:-${SLURM_ARRAY_JOB_ID:?}}
OUT="$PROJECT/outputs/scene_event_belief_filter/eval/${RUN_ID}/${INDEX}"
export STABLEWM_HOME="$RUNTIME"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} ARRAY=${SLURM_ARRAY_JOB_ID} INDEX=$INDEX TASK=$TASK SEED=$RESET_SEED RUN=$RUN_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_abstract_smdp.py" "$PROJECT/scene_event_perception.py" \
  "$PROJECT/scripts/eval_scene_event_belief_filter.py" \
  "$PROJECT/scripts/slurm_scene_event_belief_filter.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/eval_scene_event_belief_filter.py" \
  --task-id "$TASK" --reset-seed "$RESET_SEED" --observer-seeds 0,1,2 \
  --observer-root "$PROJECT/outputs/scene_event_perception/checkpoints" \
  --transition-checkpoint "$PROJECT/outputs/scene_h1b/checkpoints/seed0/abstract_smdp.pt" \
  --budget 112 --horizon 4 --out-dir "$OUT"
