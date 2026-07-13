#!/usr/bin/env bash
#SBATCH --job-name=jepa_trmana
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/trm_analysis_%j.out
# Submit after successful completion of slurm_trm_eval.sh.  Although lightweight,
# this stays on a compute node under the repository's CSV-analysis policy.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
export PATH="$PWD/.venv/bin:$PATH"
.venv/bin/python scripts/52_analyze_trm.py \
  --seed0 "${TRM_SEED0:-30000}" --episodes "${TRM_EPISODES:-64}" \
  --out-json results/trm_heldout_summary.json \
  --out-report results/trm_heldout_report.md
