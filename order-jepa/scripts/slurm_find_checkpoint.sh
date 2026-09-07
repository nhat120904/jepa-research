#!/usr/bin/env bash
#SBATCH --job-name=order_findckpt
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_findckpt_%j.out
set -euo pipefail
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
find /mnt/data/nhatnc129 -type f \( -name model_latest.pth -o -name hydra.yaml \) -print 2>/dev/null \
  | grep -Ei 'pusht|push.?t|dino.?wm' \
  | head -500
