#!/usr/bin/env bash
#SBATCH --job-name=ew_uident
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Identity stage only (CPU): sam2_frames.py on cached SAM 2 segments (SEG=<run>/<env>/front) -> u_events.
# Usage: SEG=... ENV=... TAG=chroma FRONT_ARGS="--colour-space chroma" sbatch slurm/u_ident.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
OUT=$RUN_ROOT/uident_${SLURM_JOB_ID}_${TAG:-x}/$ENV
record_source "$OUT/src"
"$TORCH_PY" "$PROJECT/scripts/sam2_frames.py" --cache "$RUN_ROOT/cache/$ENV" --segments "$SEG" --out "$OUT/front" ${FRONT_ARGS:-}
"$TORCH_PY" "$PROJECT/scripts/u_events.py" --entities "$OUT/front" --cache "$RUN_ROOT/cache/$ENV" --out "$OUT/events"
