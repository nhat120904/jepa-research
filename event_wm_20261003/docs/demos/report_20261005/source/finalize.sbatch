#!/usr/bin/env bash
#SBATCH --job-name=ew_reportfinal
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export LP_NUM_THREADS=1
export REPORT_RERENDER=1
OUT=$RUN_ROOT/demo_report_final_${SLURM_JOB_ID}
mkdir -p "$OUT/source"
cp "$PROJECT/docs/demos/report_20261005/"{export_report,finalize_predictions}.py "$OUT/"
cp "$RUN_ROOT/demo_unified_cube_57497/source/"*.py "$OUT/source/"
record_source "$OUT/source_record"
export PYTHONPATH="$OUT/source:$PYTHONPATH"
"$TORCH_PY" "$OUT/finalize_predictions.py" --src "$RUN_ROOT/demo_report_57577" --out "$OUT"
