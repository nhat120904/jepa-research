#!/usr/bin/env bash
#SBATCH --job-name=event_scene_pilot
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --array=0-31%8
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/event_scene_pilot_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
INDEX=${SLURM_ARRAY_TASK_ID:?}
if (( INDEX < 16 )); then
  TASK=4
  LOCAL_INDEX=$INDEX
else
  TASK=5
  LOCAL_INDEX=$((INDEX - 16))
fi
RESET_SEED=$((74000 + 100 * TASK + LOCAL_INDEX))
OUT="$PROJECT/outputs/scene_gate0/pilot/shards/${INDEX}"
export STABLEWM_HOME="$RUNTIME"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} ARRAY=${SLURM_ARRAY_JOB_ID} INDEX=$INDEX TASK=$TASK SEED=$RESET_SEED $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_core.py" "$PROJECT/scripts/run_scene_gate0.py" "$PROJECT/scripts/slurm_scene_gate0_pilot.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/run_scene_gate0.py" \
  --task-id "$TASK" --reset-seed "$RESET_SEED" --budgets 14,28 --horizon 4 \
  --skip-support-check --out-dir "$OUT"

