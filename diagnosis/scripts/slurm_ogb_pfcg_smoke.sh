#!/usr/bin/env bash
#SBATCH --job-name=ogb_pfcg_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/ogb_pfcg_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
DIAG="$REPO/diagnosis"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
SNAPSHOT_INDEX=${PFCG_SNAPSHOT_INDEX:-0}
OUT_TAG=${PFCG_OUT_TAG:-smoke}
export STABLEWM_HOME="$STAGE0_ROOT"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$DIAG"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum planning/pfcg.py scripts/74_ogb_pfcg_pilot.py scripts/slurm_ogb_pfcg_smoke.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$STAGE0_ROOT/.venv/bin/python" scripts/74_ogb_pfcg_pilot.py \
  --candidate-artifacts "$STAGE0_ROOT/artifacts/audit_locked_array" \
  --snapshot-index "$SNAPSHOT_INDEX" \
  --probe-pairs 32 \
  --relative-eigen-floor 1e-6 \
  --ridge-fraction 0.1 \
  --replay-physical-atol 1e-5 \
  --out-dir "results/ogb_pfcg/$OUT_TAG"

echo "===== OGB_PFCG_SMOKE_DONE ===== $(date -u +%FT%TZ)"
