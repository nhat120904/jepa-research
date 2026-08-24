#!/usr/bin/env bash
#SBATCH --job-name=gfpr_feat_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/gfpr_feat_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/gfpr_h0"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
SMOKE_INDEX=${GFPR_SMOKE_INDEX:-0}
export STABLEWM_HOME="$STAGE0_ROOT"
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/core.py" "$PROJECT/scripts/extract_snapshot_features.py" "$PROJECT/scripts/slurm_feature_smoke.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/extract_snapshot_features.py" \
  --snapshot-index "$SMOKE_INDEX" \
  --out-dir "$PROJECT/outputs/features_smoke_v3/$SMOKE_INDEX"
