#!/usr/bin/env bash
#SBATCH --job-name=rrg_expert_cache
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=01:30:00
#SBATCH --array=0-15%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/rrg_expert_cache_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/rollout_repair_gate"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/cache_expert_sequences.py" \
  --manifest "$PROJECT/outputs/stage1/expert_manifest.json" \
  --shard-index "${SLURM_ARRAY_TASK_ID:?}" --num-shards 16 \
  --out-dir "$PROJECT/outputs/stage1/expert_cache"

