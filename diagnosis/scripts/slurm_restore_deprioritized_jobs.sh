#!/usr/bin/env bash
#SBATCH --job-name=jepa_restore_q
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/restore_deprioritized_%j.out
# Restore queued work that was reversibly held for the pre-selection audit.
set -euo pipefail
for job in 26508 26610 27982; do
  scontrol release "$job" || true
done
echo "Restored held non-audit jobs after shared-branch terminal state: $(date)"
