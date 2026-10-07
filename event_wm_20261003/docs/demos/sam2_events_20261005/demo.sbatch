#!/usr/bin/env bash
#SBATCH --job-name=ew_frontdemo
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export OMP_NUM_THREADS=1
OUT=$RUN_ROOT/demo_sam2_events_${SLURM_JOB_ID}
mkdir -p "$OUT"
cp "$PROJECT/docs/demos/sam2_events_20261005/export_demo.py" "$OUT/export_demo.py"
record_source "$OUT/source_record"
"$TORCH_PY" "$OUT/export_demo.py" --root "$RUN_ROOT" --out "$OUT"
echo "DEMO_DONE=$OUT"
