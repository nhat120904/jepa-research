#!/usr/bin/env bash
#SBATCH --job-name=ew_diagframes
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
OUT=$RUN_ROOT/diag_20261005/frames_${SLURM_JOB_ID}
mkdir -p "$OUT"; cp "$PROJECT/docs/diag_events_20261005/extract_frames.py" "$OUT/"
"$OGB_PY" "$PROJECT/docs/diag_events_20261005/extract_frames.py" "$OUT"
