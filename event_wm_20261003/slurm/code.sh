#!/usr/bin/env bash
#SBATCH --job-name=ew_code
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=01:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Slow binary event code on frozen patch tokens. A short smoke run (same code path, tiny data)
# runs first so the evaluation is exercised before the full training.
#   ENV=visual-puzzle-4x5-play-v0 BASE=<train_wm.py wm.pt> BITS=32 sbatch slurm/code.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=${ENV:-visual-puzzle-4x5-play-v0}
BASE=${BASE:-$CG_RUNS/p1_56657_visual-puzzle-4x5-play-v0/wm.pt}
OUT=$RUN_ROOT/code_${SLURM_JOB_ID}_${ENV}
record_source "$OUT"
ARGS=(--cache "$RUN_ROOT/cache/$ENV" --env "$ENV" --base "$BASE" --bits "${BITS:-32}" ${EXTRA:-})
"$TORCH_PY" "$PROJECT/scripts/train_code.py" "${ARGS[@]}" --steps 200 --train-frames 50000 --val-frames 20000 --out "$OUT/smoke"
"$TORCH_PY" "$PROJECT/scripts/train_code.py" "${ARGS[@]}" --steps "${STEPS:-15000}" --out "$OUT"
