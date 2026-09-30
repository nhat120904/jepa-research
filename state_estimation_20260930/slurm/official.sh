#!/usr/bin/env bash
#SBATCH --job-name=se_official
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/state_estimation/logs/%x_%A_%a.out
# Released evaluator per task (array index -> task). MODE=smoke|full|m10|offp (m10: only the
# min-start part; offp: released sampling with partial prefill). OUTDIR reuses an earlier run directory; existing outputs are skipped.
#   full: official sampling h1 / h3 (seeds 42-44), then min-start 10 sampling
#         h1 / h3 / h3+prefill (seeds 42-44), all 50 episodes.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/state_estimation_20260930/slurm/env.sh
TASKS=(reacher pusht cube tworoom)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
MODE=${MODE:-full}
OUT=${OUTDIR:-$RUN_ROOT/official_${MODE}_${SLURM_ARRAY_JOB_ID}}/$TASK
record_source "$OUT"
run() {  # skip arms whose output already exists (resume after a failure)
  local out=${@: -1}
  [ -f "$out" ] && { echo "skip $out"; return; }
  "$PY" "$PROJECT/scripts/official_eval.py" --task "$TASK" "$@"
}
if [ "$MODE" = smoke ]; then
  run --history 1 --seed 42 --num-eval 4 --min-start 10 --out "$OUT/h1_s42.json"
  run --history 3 --seed 42 --num-eval 4 --min-start 10 --prefill --out "$OUT/h3p_s42.json"
  exit 0
fi
if [ "$MODE" = offp ]; then  # released sampling, history 3 + (partial) prefill
  for SEED in 42 43 44; do
    run --history 3 --prefill --seed $SEED --out "$OUT/official_h3p_s${SEED}.json"
  done
  exit 0
fi
if [ "$MODE" != m10 ]; then
for SEED in 42 43 44; do
  run --history 1 --seed $SEED --out "$OUT/official_h1_s${SEED}.json"
  run --history 3 --seed $SEED --out "$OUT/official_h3_s${SEED}.json"
done
fi
for SEED in 42 43 44; do
  run --history 1 --seed $SEED --min-start 10 --out "$OUT/m10_h1_s${SEED}.json"
  run --history 3 --seed $SEED --min-start 10 --out "$OUT/m10_h3_s${SEED}.json"
  run --history 3 --seed $SEED --min-start 10 --prefill --out "$OUT/m10_h3p_s${SEED}.json"
done
