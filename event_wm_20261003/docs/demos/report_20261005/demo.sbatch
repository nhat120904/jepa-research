#!/usr/bin/env bash
#SBATCH --job-name=ew_reportdemo
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
OUT=$RUN_ROOT/demo_report_${SLURM_JOB_ID}
mkdir -p "$OUT/inputs" "$OUT/source"
cp "$PROJECT/docs/demos/report_20261005/export_report.py" "$OUT/export_report.py"
cp "$RUN_ROOT/demo_unified_cube_57497/source/"*.py "$OUT/source/"
cp "$RUN_ROOT/demo_unified_cube_57497/"task*_ep0.json "$OUT/inputs/"
cp "$RUN_ROOT/demo_sam2_events_57539/"*_reader_*.json "$OUT/inputs/"
cp "$RUN_ROOT/uident_57366_cf/visual-cube-triple-play-v0/self1000/loop_v4final/u_closed_loop.json" "$OUT/inputs/reference.json"
record_source "$OUT/source_record"
export PYTHONPATH="$OUT/source:$PYTHONPATH"
"$TORCH_PY" "$OUT/export_report.py" --root "$RUN_ROOT" --out "$OUT"
