#!/usr/bin/env bash
#SBATCH --job-name=ew_state
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# STATE TRACK of the unified method on one official OGBench state play dataset (the data of the published
# state-based baselines): s_entities.py (object-factored observation -> entity tables, thresholds) -> u_events.py
# --per-frame -> WM + cost-to-go || state skill (40k steps, final checkpoint) -> closed loop on the state env:
# dev (env seed 0, 6 episodes x 5 tasks), then the OGBench protocol (5 tasks x 20 episodes) on env seed 1.
# Same backend code and settings for every family.
#   ENV=cube-triple-play-v0 LOOP_ENV=cube-triple-v0 sbatch slurm/s_pipeline.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export OMP_NUM_THREADS=2
ENV=${ENV:?ENV}; LOOP_ENV=${LOOP_ENV:?LOOP_ENV}; N=${NTRAIN:-3000}
OUT=${OUT:-$RUN_ROOT/state_${ENV}_${SLURM_JOB_ID}}; CACHE=$OUT/cache/$ENV
mkdir -p "$OUT"; record_source "$OUT/src"
echo "=== entities $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/s_entities.py" --data "$DATA" --env "$ENV" --train-episodes "$N" --val-episodes 100 --out "$OUT"
echo "=== events $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_events.py" --entities "$OUT/front" --cache "$CACHE" --thresholds-from "$OUT/thr" --per-frame --out "$OUT/events"
echo "=== WM (background) + skill $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_wm.py" --events "$OUT/events" --out "$OUT/wm" > "$OUT/wm.log" 2>&1 &
PID=$!
TF=$(( (N + 2) * 1001 ))
"$TORCH_PY" "$PROJECT/scripts/u_skill.py" --cache "$CACHE" --events "$OUT/events" --train-frames "$TF" --val-frames 101000 --steps 40000 --out "$OUT/skill"
wait $PID
tail -n 2 "$OUT/wm.log"
LOOP=("$TORCH_PY" "$PROJECT/scripts/u_closed_loop.py" --env "$LOOP_ENV" --model "$OUT/wm/u_model.pt" --state-layout "$OUT/front/layout.json"
      --skill "$OUT/skill/u_skill.pt" --events "$OUT/events" --cache "$CACHE" --workers 10)
echo "=== closed loop, dev (seed 0) $(date -u +%T)"
"${LOOP[@]}" --episodes 6 --seed 0 --out "$OUT/loop_dev"
echo "=== closed loop, OGBench protocol (5 tasks x 20 episodes, seed 1) $(date -u +%T)"
"${LOOP[@]}" --episodes 20 --seed 1 --out "$OUT/loop_seed1"
echo "=== done $(date -u +%T)"
