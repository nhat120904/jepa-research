#!/usr/bin/env bash
#SBATCH --job-name=jepa_goalmarg_analyze
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/goal_marg_analyze_%j.out
# Gate 2 analysis (docs/plans/2026-08-11-goal-marginalization-design.md).
# Applies the pre-registered decision rule; CPU-only but still via Slurm.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python

IN=${GOALMARG_IN:-results/goal_marginalization_mw-push_seed90000_n16.csv}
OUT=${GOALMARG_ANALYZE_OUT:-results/goal_marginalization_report.md}

echo "HOST $(hostname) $(date)"
sha256sum scripts/83_analyze_goal_marginalization.py

$PY scripts/83_analyze_goal_marginalization.py --csv "$IN" --task mw-push --out "$OUT"
echo "===== GOAL_MARG_ANALYZE_DONE ====="; date
cat "$OUT"
