#!/usr/bin/env bash
#SBATCH --job-name=ogb_s0_audsmk
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_stage0_audit_smoke_%j.out
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
sha256sum "$DIAG/scripts/72_ogb_stage0_candidate_audit.py" "$DIAG/scripts/slurm_ogb_stage0_audit_smoke.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$STAGE0_ROOT/.venv/bin/python" scripts/72_ogb_stage0_candidate_audit.py \
  --num-snapshots 2 \
  --num-samples 32 \
  --cem-steps 3 \
  --topk 8 \
  --bootstrap 1000 \
  --encode-batch 32 \
  --out-dir results/ogb_stage0/audit_smoke \
  --artifact-dir "$STAGE0_ROOT/artifacts/audit_smoke"

echo "===== OGB_STAGE0_AUDIT_SMOKE_DONE ===== $(date -u +%FT%TZ)"
