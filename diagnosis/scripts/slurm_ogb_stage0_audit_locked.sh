#!/usr/bin/env bash
#SBATCH --job-name=ogb_s0_audit
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_stage0_audit_locked_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
DIAG="$REPO/diagnosis"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$STAGE0_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch

cd "$DIAG"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
echo "repo_commit=$(git -C "$REPO" rev-parse HEAD)"
echo "stable_worldmodel_commit=$(git -C "$DIAG/external/stable-worldmodel" rev-parse HEAD)"
sha256sum "$DIAG/scripts/72_ogb_stage0_candidate_audit.py" "$DIAG/scripts/slurm_ogb_stage0_audit_locked.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$STAGE0_ROOT/.venv/bin/python" scripts/72_ogb_stage0_candidate_audit.py \
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
  --bootstrap 10000 \
  --out-dir results/ogb_stage0/audit_locked \
  --artifact-dir "$STAGE0_ROOT/artifacts/audit_locked"

echo "===== OGB_STAGE0_AUDIT_LOCKED_DONE ===== $(date -u +%FT%TZ)"
