#!/usr/bin/env bash
#SBATCH --job-name=perd_collect_b
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=64-127%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/perd_collect_b_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/physical_search_distillation"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
export STABLEWM_HOME="$STAGE0_ROOT" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=$TASK_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/core.py" "$PROJECT/scripts/collect_populations.py" "$PROJECT/scripts/slurm_01_collect_second.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/collect_populations.py" \
  --snapshot-index "$TASK_ID" --manifest "$PROJECT/outputs/h0/manifest.json" \
  --out-dir "$PROJECT/outputs/h0/populations"
