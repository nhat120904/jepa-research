#!/usr/bin/env bash
#SBATCH --job-name=ew_demo
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export OMP_NUM_THREADS=1
OUT=$RUN_ROOT/demo_unified_cube_${SLURM_JOB_ID}
mkdir -p "$OUT/source"
cp "$PROJECT"/scripts/*.py "$OUT/source/"
cp "$PROJECT/docs/demos/unified_cube_20261005/record_demo.py" "$OUT/record_demo.py"
record_source "$OUT/source_record"
export PYTHONPATH="$OUT/source:$PYTHONPATH"
REF=$RUN_ROOT/uident_57366_cf/visual-cube-triple-play-v0/self1000/loop_v4final
"$TORCH_PY" "$OUT/record_demo.py" --reference "$REF/u_closed_loop.json" --out "$OUT"
"$TORCH_PY" "$OUT/source/u_loop_inspect.py" --loop "$REF" > "$OUT/reference_inspection.txt"
echo "DEMO_DONE=$OUT"
