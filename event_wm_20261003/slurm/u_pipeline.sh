#!/usr/bin/env bash
#SBATCH --job-name=ew_unified
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Unified method pipeline (same code and configuration for every task family).
#   ENV=visual-cube-triple-play-v0 LOOP_ENV=visual-cube-triple-v0 OUT=<dir> PARTS="tracks events reader wm skill" sbatch slurm/u_pipeline.sh
#   PARTS=loop ... sbatch --cpus-per-task=16 slurm/u_pipeline.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=${ENV:?ENV}; CACHE=$RUN_ROOT/cache/$ENV
OUT=${OUT:-$RUN_ROOT/unified_${ENV}_${SLURM_JOB_ID}}
record_source "$OUT/src_${SLURM_JOB_ID}"
for PART in ${PARTS:?PARTS}; do
  echo "=== $PART $(date -u +%T)"
  case $PART in
    tracks) "$TORCH_PY" "$PROJECT/scripts/sam2_frames.py" --cache "$CACHE" --out "$OUT/front" ${TRACK_ARGS:-} ;;
    events) "$TORCH_PY" "$PROJECT/scripts/u_events.py" --entities "$OUT/front" --cache "$CACHE" --out "$OUT/events" ;;
    reader) "$TORCH_PY" "$PROJECT/scripts/u_reader.py" --cache "$CACHE" --events "$OUT/events" --out "$OUT/reader" ;;
    wm)     "$TORCH_PY" "$PROJECT/scripts/u_wm.py" --events "$OUT/events" --out "$OUT/wm" ${WM_ARGS:-} ;;
    skill)  "$TORCH_PY" "$PROJECT/scripts/u_skill.py" --cache "$CACHE" --events "$OUT/events" --out "$OUT/skill" ${SKILL_ARGS:-} ;;
    loop)   "$TORCH_PY" "$PROJECT/scripts/u_closed_loop.py" --env "${LOOP_ENV:?LOOP_ENV}" --model "$OUT/wm/u_model.pt" \
              --reader "$OUT/reader/u_reader.pt" --skill "$OUT/skill/${SKILL_CKPT:-u_skill.pt}" --events "$OUT/${EVENTS:-events_ref}" --cache "$CACHE" \
              --episodes "${EPISODES:-6}" --workers "${WORKERS:-14}" --out "$OUT/loop${TAG:-}" ${LOOP_ARGS:-} ;;
  esac
done
