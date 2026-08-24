#!/usr/bin/env bash
#SBATCH --job-name=perd_eval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --array=0,8,16,24,32,40,48,56,64,72,80,88,96,104,112,120%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/perd_eval_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/physical_search_distillation"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
TASK_ID_SECOND=$((TASK_ID + 4))
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=$TASK_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/core.py" "$PROJECT/scripts/evaluate_zero_query.py" "$PROJECT/scripts/slurm_03_eval.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/evaluate_zero_query.py" \
  --snapshot-index "$TASK_ID" --manifest "$PROJECT/outputs/h0/manifest.json" \
  --checkpoints "$PROJECT/outputs/h0/checkpoints" --out-dir "$PROJECT/outputs/h0/eval"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/evaluate_zero_query.py" \
  --snapshot-index "$TASK_ID_SECOND" --manifest "$PROJECT/outputs/h0/manifest.json" \
  --checkpoints "$PROJECT/outputs/h0/checkpoints" --out-dir "$PROJECT/outputs/h0/eval"
