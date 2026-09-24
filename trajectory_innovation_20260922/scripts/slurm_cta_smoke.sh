#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_smoke
#SBATCH --partition=main
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_smoke_%j.out
# CPU code-path check of the whole CTA chain on 2+1 roots (8 + 2 decisions) (collect -> encode -> train -> closed loop -> aggregate).
# Tiny step counts; nothing here is a result. Usage: sbatch slurm_cta_smoke.sh
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
RUN="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_smoke_${SLURM_JOB_ID}"
CODE="$RUN/code"
[[ ! -e "$RUN" ]] || exit 2
export WANDB_MODE="${WANDB_MODE:-offline}" CTA_TAG=smoke
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
unit_tests
mkdir -p "$RUN/collect" "$RUN/features" "$RUN/train" "$RUN/cl"
"$PY" "$CODE/scripts/d_collect.py" --out "$RUN/collect" --prep "$PREP" --smoke "$SMOKE" --first 39994 --count 2 \
  --device cpu --max-decisions 8
"$PY" "$CODE/scripts/cta_encode.py" --collect "$RUN/collect" --out "$RUN/features" --smoke "$SMOKE" --device cpu --smoke-mode
"$PY" "$CODE/scripts/cta_train.py" --run "$RUN/train" --features "$RUN/features" --device cpu \
  --steps1 12 --steps2 12 --steps3 10 --lam 0.1 --src-block 5 --wm-block 5 --adapt-steps 4 --samples 2 \
  --decisions 2 --wm-decisions 2 --warmup 4 --eval-every 6 --eval-n 2 --log-every 3
"$PY" "$CODE/scripts/cta_closed_loop.py" --run "$RUN/cl" --prep "$PREP" --smoke "$SMOKE" --train-run "$RUN/train" \
  --dev-shard "$RUN/collect/shard_39994_39995.npz" --first 39996 --count 1 \
  --arms P0,PHYS8,FULL8,CODE8,CTA8,DIRECT8 --max-decisions 2 --device cpu
"$PY" "$CODE/scripts/cta_aggregate.py" --run "$RUN/cl" --out "$RUN/agg" --first 39996 --count 1
echo SMOKE_OK
