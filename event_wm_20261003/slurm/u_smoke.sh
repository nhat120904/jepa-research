#!/usr/bin/env bash
#SBATCH --job-name=ew_usmoke
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# CPU smoke test of the unified backend chain on a pilot event set (tiny step counts; numbers meaningless).
# EV=<identity-stage env dir with events/> ENV=<play env> LOOP_ENV=<eval env> sbatch slurm/u_smoke.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
CACHE=$RUN_ROOT/cache/${ENV:?ENV}; OUT=$RUN_ROOT/usmoke_${SLURM_JOB_ID}
record_source "$OUT/src"
"$TORCH_PY" "$PROJECT/scripts/u_reader.py" --cache "$CACHE" --events "$EV/events" --steps 30 --batch 32 --device cpu --out "$OUT/reader"
"$TORCH_PY" "$PROJECT/scripts/u_wm.py" --events "$EV/events" --wm-steps 30 --h-steps 20 --device cpu --out "$OUT/wm"
"$TORCH_PY" "$PROJECT/scripts/u_skill.py" --cache "$CACHE" --events "$EV/events" --train-frames 12000 --val-frames 6000 --steps 30 --batch 32 --device cpu --out "$OUT/skill"
"$TORCH_PY" "$PROJECT/scripts/u_closed_loop.py" --env "${LOOP_ENV:?LOOP_ENV}" --model "$OUT/wm/u_model.pt" --reader "$OUT/reader/u_reader.pt" \
  --skill "$OUT/skill/u_skill.pt" --events "$EV/events" --cache "$CACHE" --episodes 1 --tasks 1 2 --max-expansions 300 --timeout 60 --device cpu --out "$OUT/loop"
