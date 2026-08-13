#!/usr/bin/env bash
#SBATCH --job-name=ogb_te_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_true_endpoint_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
DIAG="$REPO/diagnosis"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
SNAPSHOT_INDEX=${TRUE_ENDPOINT_SNAPSHOT_INDEX:?set TRUE_ENDPOINT_SNAPSHOT_INDEX}
OUT_TAG=${TRUE_ENDPOINT_OUT_TAG:-smoke_${SNAPSHOT_INDEX}}
export STABLEWM_HOME="$STAGE0_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$DIAG"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} SNAPSHOT=$SNAPSHOT_INDEX $(date -u +%FT%TZ)"
sha256sum \
  scripts/72_ogb_stage0_candidate_audit.py \
  scripts/76_ogb_true_endpoint_corrected.py \
  scripts/slurm_ogb_true_endpoint_smoke.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$STAGE0_ROOT/.venv/bin/python" scripts/76_ogb_true_endpoint_corrected.py \
  --candidate-artifacts "$STAGE0_ROOT/artifacts/audit_locked_array" \
  --reference-physical-shards results/ogb_pfcg/locked_v2_shards \
  --snapshot-index "$SNAPSHOT_INDEX" \
  --physical-atol 1e-5 \
  --latent-atol 1e-5 \
  --domain-ratio-max 0.25 \
  --out-dir "results/ogb_true_endpoint_corrected/$OUT_TAG"

echo "===== OGB_TRUE_ENDPOINT_SMOKE_DONE ===== $(date -u +%FT%TZ)"
