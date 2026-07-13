#!/usr/bin/env bash
#SBATCH --job-name=jepa_h_cfana
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseH_heldout_analysis_%j.out
# Submit with --dependency=afterok:<dino-array>:<jepa-array>.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
export PATH="$PWD/.venv/bin:$PATH"

for M in dino_wm_droid jepa_wm_droid; do
  .venv/bin/python scripts/44_aggregate_cf_seeds.py \
    --model "$M" --frozen-tag heldout_frozen \
    --seeds heldout_s0 heldout_s1 heldout_s2 heldout_s3 \
    --out "results/cf_heldout_${M}_summary.md"
done

echo "===== PHASEH_HELDOUT_ANALYSIS_DONE ===== $(date)"
