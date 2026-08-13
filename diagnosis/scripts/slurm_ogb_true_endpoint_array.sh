#!/usr/bin/env bash
#SBATCH --job-name=ogb_te_array
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --array=0-31%8
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_true_endpoint_array_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
DIAG="$REPO/diagnosis"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
export STABLEWM_HOME="$STAGE0_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$DIAG"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=$TASK_ID $(date -u +%FT%TZ)"
sha256sum \
  scripts/72_ogb_stage0_candidate_audit.py \
  scripts/76_ogb_true_endpoint_corrected.py \
  scripts/slurm_ogb_true_endpoint_array.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$STAGE0_ROOT/.venv/bin/python" scripts/76_ogb_true_endpoint_corrected.py \
  --candidate-artifacts "$STAGE0_ROOT/artifacts/audit_locked_array" \
  --reference-physical-shards results/ogb_pfcg/locked_v2_shards \
  --snapshot-index "$TASK_ID" \
  --physical-atol 1e-5 \
  --latent-atol 1e-5 \
  --domain-ratio-max 0.25 \
  --out-dir "results/ogb_true_endpoint_corrected/locked_shards/$TASK_ID"

echo "===== OGB_TRUE_ENDPOINT_ARRAY_TASK_DONE task=$TASK_ID ===== $(date -u +%FT%TZ)"
