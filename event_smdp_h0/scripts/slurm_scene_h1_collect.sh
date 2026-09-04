#!/usr/bin/env bash
#SBATCH --job-name=scene_h1_collect
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=03:00:00
#SBATCH --array=0-39%8
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_h1_collect_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
INDEX=${SLURM_ARRAY_TASK_ID:?}
if (( INDEX < 16 )); then
  SPLIT=train; TASK=4; LOCAL=$INDEX
elif (( INDEX < 32 )); then
  SPLIT=train; TASK=5; LOCAL=$((INDEX - 16))
elif (( INDEX < 36 )); then
  SPLIT=val; TASK=4; LOCAL=$((INDEX - 32))
else
  SPLIT=val; TASK=5; LOCAL=$((INDEX - 36))
fi
if [[ "$SPLIT" == train ]]; then BASE=81000; else BASE=82000; fi
DEFAULT_RESET_SEED=$((BASE + 100 * TASK + LOCAL))
# A canonical skill path can very occasionally fall outside the scripted
# controller's support on a particular reset.  Permit a recorded, explicit
# replacement seed without changing the shard's split/task/index mapping.
RESET_SEED=${RESET_SEED_OVERRIDE:-$DEFAULT_RESET_SEED}
OUT="$PROJECT/outputs/scene_h1/data/${SPLIT}/${INDEX}"
export STABLEWM_HOME="$RUNTIME"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} ARRAY=${SLURM_ARRAY_JOB_ID} INDEX=$INDEX SPLIT=$SPLIT TASK=$TASK SEED=$RESET_SEED $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_core.py" "$PROJECT/scene_learning.py" \
  "$PROJECT/scripts/collect_scene_h1.py" "$PROJECT/scripts/slurm_scene_h1_collect.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/collect_scene_h1.py" \
  --task-id "$TASK" --reset-seed "$RESET_SEED" --split "$SPLIT" --out-dir "$OUT"
