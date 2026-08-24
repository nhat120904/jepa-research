#!/usr/bin/env bash
#SBATCH --job-name=hys_curv
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_curv_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
S=../hys_h0
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date)"
for M in dino_wm_metaworld jepa_wm_metaworld; do
  echo "########## $M ##########"
  .venv/bin/python $S/scripts/01_curvature_by_regime.py \
    --cache data/precomputed_latents/metaworld__${M}.h5 \
    --out $S/outputs/curvature_${M}.json
done
echo "===== CURV_DONE ===== $(date)"
