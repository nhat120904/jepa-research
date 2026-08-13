#!/usr/bin/env bash
#SBATCH --job-name=ogb_s0_arr
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --array=0-31%8
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_stage0_audit_array_%A_%a.out
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

cd "$DIAG"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} TASK=$TASK_ID $(date -u +%FT%TZ)"
sha256sum scripts/72_ogb_stage0_candidate_audit.py scripts/slurm_ogb_stage0_audit_array.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$STAGE0_ROOT/.venv/bin/python" scripts/72_ogb_stage0_candidate_audit.py \
  --manifest-snapshots 32 \
  --snapshot-index "$TASK_ID" \
  --num-snapshots 32 \
  --goal-offset 25 \
  --seed 20260810 \
  --horizon 5 \
  --action-block 5 \
  --num-samples 300 \
  --cem-steps 30 \
  --topk 30 \
  --var-scale 1.0 \
  --encode-batch 64 \
  --bootstrap 1000 \
  --out-dir "results/ogb_stage0/audit_locked_shards/$TASK_ID" \
  --artifact-dir "$STAGE0_ROOT/artifacts/audit_locked_array"

echo "===== OGB_STAGE0_AUDIT_ARRAY_TASK_DONE task=$TASK_ID ===== $(date -u +%FT%TZ)"
