#!/usr/bin/env bash
#SBATCH --job-name=lscope_c_encode
#SBATCH --partition=main
#SBATCH --array=0-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/stage_c_encode_%A_%a.out

set -euo pipefail

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
POLICY_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_baseline
STAGE_C_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_c

mkdir -p "$STAGE_C_ROOT/features" "$STAGE_C_ROOT/logs"
export HF_HOME="$STAGE_C_ROOT/hf_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

env PYTHONPATH="$PROJECT_ROOT/latent_scope_20260909" \
    "$POLICY_ROOT/policy_venv/bin/python" \
    "$PROJECT_ROOT/latent_scope_20260909/scripts/encode_stage_c_offline.py" \
    --config "$PROJECT_ROOT/latent_scope_20260909/configs/stage_c_screen.json" \
    --task-index "$SLURM_ARRAY_TASK_ID" \
    --output-dir "$STAGE_C_ROOT/features/task_$SLURM_ARRAY_TASK_ID"
