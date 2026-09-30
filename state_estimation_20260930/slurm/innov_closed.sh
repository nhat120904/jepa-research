#!/usr/bin/env bash
#SBATCH --job-name=se_innovcl
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/state_estimation/logs/%x_%A_%a.out
# Closed-loop innovation feedback (array index -> task), min-start 16, seeds 42-44:
# h3 (released warm-up), h3+prefill, h4+prefill (= h3+prefill control), h4+prefill+innovation.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/state_estimation_20260930/slurm/env.sh
TASKS=(reacher pusht cube tworoom)
TASK=${TASKS[$SLURM_ARRAY_TASK_ID]}
GAMMAS=${GAMMAS:-"dist:0.5 post:0.5 dist:1.0"}
OUT=$RUN_ROOT/innov_closed_${SLURM_ARRAY_JOB_ID}/$TASK
record_source "$OUT"
run() { "$PY" "$PROJECT/scripts/official_eval.py" --task "$TASK" --min-start 16 "$@"; }
for SEED in 42 43 44; do
  run --history 3 --seed $SEED --out "$OUT/h3_s${SEED}.json"
  run --history 3 --prefill --seed $SEED --out "$OUT/h3p_s${SEED}.json"
  run --history 4 --prefill --seed $SEED --out "$OUT/h4p_s${SEED}.json"
  for G in $GAMMAS; do
    M=${G%%:*}; V=${G##*:}
    run --history 4 --prefill --innov-mode $M --innov-gamma $V --seed $SEED --out "$OUT/h4p_${M}${V}_s${SEED}.json"
  done
done
