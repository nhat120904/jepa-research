#!/usr/bin/env bash
#SBATCH --job-name=rrg_fresh_eval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --array=0-31%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/rrg_fresh_eval_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/rollout_repair_gate"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
SNAPSHOT_INDEX=$((SLURM_ARRAY_TASK_ID * 4))
export STABLEWM_HOME="$STAGE0_ROOT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/evaluate_fresh_cem.py" \
  --snapshot-index "$SNAPSHOT_INDEX" \
  --manifest "$REPO/physical_search_distillation/outputs/h0/manifest.json" \
  --checkpoint-dir "$PROJECT/outputs/stage1/checkpoints" \
  --out-dir "$PROJECT/outputs/stage1/fresh_eval"

