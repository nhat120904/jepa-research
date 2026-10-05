#!/usr/bin/env bash
#SBATCH --job-name=ew_ufam
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=14:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# FROZEN unified pipeline (2026-10-05) on one task family, end to end, no per-family settings:
#   SAM 2 segments + identities (150 train / 30 val episodes) -> events -> self-training round (reader + agent
#   head labels NTRAIN episodes, events with round-1 thresholds, WM, refine, skill) -> closed loop on the
#   official tasks (final skill checkpoint).
#   ENV=visual-cube-double-play-v0 LOOP_ENV=visual-cube-double-v0 sbatch slurm/u_family.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export OMP_NUM_THREADS=3
ENV=${ENV:?ENV}; LOOP_ENV=${LOOP_ENV:?LOOP_ENV}; CACHE=$RUN_ROOT/cache/$ENV; N=${NTRAIN:-1000}
OUT=${OUT:-$RUN_ROOT/family_${ENV}_${SLURM_JOB_ID}}; S=$OUT/self$N
mkdir -p "$S"; record_source "$OUT/src_${SLURM_JOB_ID}"
echo "=== segments + identities $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/sam2_frames.py" --cache "$CACHE" --out "$OUT/front"
echo "=== events $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_events.py" --entities "$OUT/front" --cache "$CACHE" --out "$OUT/events"
echo "=== reader + agent head $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_reader.py" --cache "$CACHE" --events "$OUT/events" --agent-head --entities "$OUT/front" --out "$S/reader"
echo "=== reader entities ($N episodes) $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_reader_entities.py" --cache "$CACHE" --front "$OUT/front" --reader "$S/reader/u_reader.pt" \
  --train-episodes "$N" --val-episodes 100 --out "$S/front"
echo "=== self events $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_events.py" --entities "$S/front" --cache "$CACHE" --thresholds-from "$OUT/events" --out "$S/events"
echo "=== WM (background) + refine + skill $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_wm.py" --events "$S/events" --out "$S/wm" > "$S/wm.log" 2>&1 &
P=$!
"$TORCH_PY" "$PROJECT/scripts/u_refine.py" --cache "$CACHE" --events "$S/events" --reader "$S/reader/u_reader.pt" --out "$S/events_ref"
TF=$(( (N + 2) * 1001 ))
"$TORCH_PY" "$PROJECT/scripts/u_skill.py" --cache "$CACHE" --events "$S/events_ref" --train-frames "$TF" --val-frames 101000 --steps 40000 --out "$S/skill"
wait $P
tail -n 2 "$S/wm.log"
echo "=== closed loop $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_closed_loop.py" --env "$LOOP_ENV" --model "$S/wm/u_model.pt" --reader "$S/reader/u_reader.pt" \
  --skill "$S/skill/u_skill.pt" --events "$S/events_ref" --cache "$CACHE" --episodes 6 --workers 6 --out "$S/loop"
echo "=== done $(date -u +%T)"
