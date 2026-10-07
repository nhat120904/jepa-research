#!/usr/bin/env bash
#SBATCH --job-name=ew_ueval
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Official-protocol evaluation of frozen unified models (OGBench default: 5 tasks x 20 episodes) on env seeds
# not used during development. Models, reader, skill and events are passed in unchanged.
#   LOOP_ENV=<eval env> P=<run dir with wm/ skill/ events/> READER=<u_reader.pt> CACHE_ENV=<play env> SEED=<seed> sbatch slurm/u_eval.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export OMP_NUM_THREADS=1
LOOP_ENV=${LOOP_ENV:?LOOP_ENV}; P=${P:?P}; READER=${READER:?READER}; CACHE_ENV=${CACHE_ENV:?CACHE_ENV}; SEED=${SEED:?SEED}
OUT=$P/eval_seed${SEED}_${SLURM_JOB_ID}
mkdir -p "$OUT"; record_source "$OUT/src"
"$TORCH_PY" "$PROJECT/scripts/u_closed_loop.py" --env "$LOOP_ENV" --model "$P/wm/u_model.pt" --reader "$READER" \
  --skill "$P/skill/u_skill.pt" --events "$P/events" --cache "$RUN_ROOT/cache/$CACHE_ENV" \
  --episodes ${EPISODES:-20} --seed "$SEED" --workers ${WORKERS:-10} --out "$OUT/loop"
