#!/usr/bin/env bash
#SBATCH --job-name=acm_calib
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_calib_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/scripts/_common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" "$PROJECT/scripts/train_probe.py" \
  --manifest "$MANIFEST" --n-episodes 120 --steps-per-episode 4 \
  --out-dir "$PROJECT/outputs/probe_bridge"
echo "GATE A COMPLETE $(date -u +%FT%TZ)"
