#!/usr/bin/env bash
#SBATCH --job-name=scene_h1_eval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-15%8
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_h1_eval_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
INDEX=${SLURM_ARRAY_TASK_ID:?}
MODEL_SEED=${MODEL_SEED:-0}
if (( INDEX < 8 )); then TASK=4; LOCAL=$INDEX; else TASK=5; LOCAL=$((INDEX - 8)); fi
RESET_SEED=$((83000 + 100 * TASK + LOCAL))
OUT="$PROJECT/outputs/scene_h1/eval/seed${MODEL_SEED}/${INDEX}"
export STABLEWM_HOME="$RUNTIME"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} ARRAY=${SLURM_ARRAY_JOB_ID} INDEX=$INDEX TASK=$TASK SEED=$RESET_SEED MODEL_SEED=$MODEL_SEED $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_core.py" "$PROJECT/scene_learning.py" \
  "$PROJECT/scripts/eval_scene_h1.py" "$PROJECT/scripts/slurm_scene_h1_eval.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/eval_scene_h1.py" \
  --task-id "$TASK" --reset-seed "$RESET_SEED" --model-seed "$MODEL_SEED" \
  --checkpoint-root "$PROJECT/outputs/scene_h1/checkpoints" \
  --budgets 14,28 --horizon 4 --out-dir "$OUT"

