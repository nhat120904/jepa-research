#!/usr/bin/env bash
#SBATCH --job-name=ew_upf
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Per-frame event round on an existing self-training round, whose reader already read every frame
# (self1000/front/entities_*.npz): events with u_events.py --per-frame (round-1 change thresholds) -> WM + cost-to-go
# || skill (no u_refine: per-frame events are already timed on the per-frame reading) -> closed loop with the
# round's reader. Reader, entity table, thresholds and evaluation protocol are the round's own, so the only change
# against that round is the event extraction.
#   ENV=<play env> LOOP_ENV=<eval env> S=<.../self1000> R1=<round-1 events dir> sbatch slurm/u_pf.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export OMP_NUM_THREADS=3
ENV=${ENV:?ENV}; LOOP_ENV=${LOOP_ENV:?LOOP_ENV}; S=${S:?S}; R1=${R1:?R1}; CACHE=$RUN_ROOT/cache/$ENV; N=${NTRAIN:-1000}
P=$S/pf_${SLURM_JOB_ID}
mkdir -p "$P"; record_source "$P/src"
echo "=== per-frame events $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_events.py" --entities "$S/front" --cache "$CACHE" --thresholds-from "$R1" --per-frame --out "$P/events"
echo "=== WM (background) + skill $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_wm.py" --events "$P/events" --out "$P/wm" > "$P/wm.log" 2>&1 &
PID=$!
TF=$(( (N + 2) * 1001 ))
"$TORCH_PY" "$PROJECT/scripts/u_skill.py" --cache "$CACHE" --events "$P/events" --train-frames "$TF" --val-frames 101000 \
  --steps ${SKILL_STEPS:-40000} --out "$P/skill"
wait $PID
tail -n 2 "$P/wm.log"
echo "=== closed loop $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_closed_loop.py" --env "$LOOP_ENV" --model "$P/wm/u_model.pt" --reader "$S/reader/u_reader.pt" \
  --skill "$P/skill/u_skill.pt" --events "$P/events" --cache "$CACHE" --episodes 6 --workers 6 --out "$P/loop"
echo "=== done $(date -u +%T)"
