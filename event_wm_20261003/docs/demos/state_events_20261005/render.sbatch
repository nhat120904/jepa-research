#!/usr/bin/env bash
#SBATCH --job-name=ew_strender
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export LP_NUM_THREADS=1
OUT=$RUN_ROOT/demo_state_events_${SLURM_JOB_ID}
mkdir -p "$OUT"; cp "$PROJECT/docs/demos/state_events_20261005/render_frames.py" "$OUT/"
"$OGB_PY" "$PROJECT/docs/demos/state_events_20261005/render_frames.py" "$OUT"
