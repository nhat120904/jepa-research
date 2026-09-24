#!/usr/bin/env bash
#SBATCH --job-name=ti_c2c_agg
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/c2c_agg_%j.out
# Usage: sbatch slurm_c2_confirm_aggregate.sh <out_dir> <confirm_eval_run_dir> <r1_train_run_dir> <c2_eval_run_dir>
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
OUT=${1:?out}; EVAL=${2:?eval run}; R1=${3:?r1 run}; C2=${4:?c2 eval run}
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
PREP=/mnt/data/nhatnc129/jepa/trajectory_innovation/prepare_53776
[[ ! -e "$OUT" ]] || exit 2
mkdir -p "$OUT/code"
cp -r "$PROJECT/ti_wm" "$PROJECT/scripts" "$OUT/code/"
export PYTHONPATH="$OUT/code:$PREP/deps:$PREP/upstream" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
"$PY" "$OUT/code/scripts/aggregate_c2_confirm.py" --run "$EVAL" --r1 "$R1" --c2_run "$C2" --out "$OUT"
