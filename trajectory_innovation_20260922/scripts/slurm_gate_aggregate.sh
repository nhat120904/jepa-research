#!/usr/bin/env bash
#SBATCH --job-name=ti_gate_agg
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/gate_agg_%j.out
# Usage: sbatch slurm_gate_aggregate.sh <out_dir> <main_run_dir> [more_run_dirs...]
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
OUT=${1:?out dir}; shift
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
PREP=/mnt/data/nhatnc129/jepa/trajectory_innovation/prepare_53776
[[ ! -e "$OUT" ]] || exit 2
mkdir -p "$OUT/code"
cp -r "$PROJECT/ti_wm" "$PROJECT/scripts" "$OUT/code/"
export PYTHONPATH="$OUT/code:$PREP/deps:$PREP/upstream" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
"$PY" "$OUT/code/scripts/aggregate_abc.py" --runs "$@" --prep "$PREP" --out "$OUT"
