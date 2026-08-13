#!/usr/bin/env bash
#SBATCH --job-name=jepa_snr_analyze
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/task_snr_analyze_%j.out
# Gate 1 analysis (docs/plans/2026-08-11-goal-marginalization-design.md).
# CPU-only, but still routed through Slurm per project convention (login node
# is inspection/monitoring only).
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python

IN=${SNR_IN:-results/task_snr_pilot.csv}
OUT=${SNR_ANALYZE_OUT:-results/task_snr_law.md}

echo "HOST $(hostname) $(date)"
sha256sum scripts/81_analyze_task_snr.py

$PY scripts/81_analyze_task_snr.py --snr "$IN" --out "$OUT"
echo "===== TASK_SNR_ANALYZE_DONE ====="; date
cat "$OUT"
