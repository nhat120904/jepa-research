#!/usr/bin/env bash
#SBATCH --job-name=mwm_g2_smoke
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g2_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
DATA="$PROJECT/outputs/gate2_smoke/glides.pt"
FEATURES="$PROJECT/outputs/gate2_smoke/anchors.pt"
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sha256sum "$PROJECT/scripts/generate_glide_dataset.py" \
          "$PROJECT/scripts/extract_anchor_features.py" \
          "$PROJECT/scripts/fit_anchor_certificates.py" \
          "$PROJECT/scripts/slurm_gate2_smoke.sh"
"$PY" "$PROJECT/scripts/generate_glide_dataset.py" \
  --out "$DATA" --episodes 96 --batch-size 48 --device cuda
"$PY" "$PROJECT/scripts/extract_anchor_features.py" \
  --dataset "$DATA" --out "$FEATURES" --batch-size 24 --device cuda
"$PY" "$PROJECT/scripts/fit_anchor_certificates.py" \
  --dataset "$DATA" --features "$FEATURES" \
  --out-dir "$PROJECT/outputs/gate2_smoke/certificates" \
  --epochs 2 --patience 2 --batch-size 24 --seeds 0 --bootstrap 200 \
  --device cuda
