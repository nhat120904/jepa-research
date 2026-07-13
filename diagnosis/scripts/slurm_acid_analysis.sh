#!/usr/bin/env bash
#SBATCH --job-name=jepa_acid_ana
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acid_analysis_%j.out
#
# Submit after the eight-cell ACID evaluation array succeeds:
#   sbatch --dependency=afterok:<ACID_EVAL_JOB_ID> scripts/slurm_acid_analysis.sh
set -euo pipefail

cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
export CAI_JEPA_TORCH_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PATH="$PWD/.venv/bin:$PATH"

SEED0=${ACID_SEED0:-22000}
EPISODES=${ACID_EPISODES:-32}
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date) seed0=$SEED0 n=$EPISODES"

.venv/bin/python scripts/53_analyze_acid_baseline.py \
  --results-dir results --checkpoint-dir checkpoints \
  --seed0 "$SEED0" --episodes "$EPISODES" \
  --out-csv results/acid_paired_summary.csv \
  --out-report results/acid_paired_report.md

echo "===== ACID_ANALYSIS_DONE ===== $(date)"
