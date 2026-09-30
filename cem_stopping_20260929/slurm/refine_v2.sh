#!/usr/bin/env bash
#SBATCH --job-name=cemstop_refine_v2
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/cem_stopping/logs/%x_%j.out
set -euo pipefail
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
RUN=/mnt/data/nhatnc129/jepa/cem_stopping/refine_v2_${SLURM_JOB_ID}
mkdir -p "$RUN/code"
cp -a "$PROJECT/cemstop" "$PROJECT/scripts" "$PROJECT/tests" "$PROJECT/docs" "$RUN/code/"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
"$PY" "$RUN/code/tests/test_continuation.py"
"$PY" "$RUN/code/scripts/refine_v2.py" \
  --data /mnt/data/nhatnc129/jepa/cem_stopping/dev_v1 --out "$RUN/results"
