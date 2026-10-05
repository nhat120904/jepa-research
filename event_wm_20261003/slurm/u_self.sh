#!/usr/bin/env bash
#SBATCH --job-name=ew_uself
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Self-training round: the reader (trained on the SAM 2 round's pseudo-labels, + agent head) labels N train
# episodes; events with the round-1 change thresholds; reader-refined timing; WM and skill on the larger set.
#   ENV=<play env> OUT=<round-1 dir with front/ and events/> NTRAIN=1000 sbatch slurm/u_self.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export OMP_NUM_THREADS=3
ENV=${ENV:?ENV}; CACHE=$RUN_ROOT/cache/$ENV; OUT=${OUT:?OUT}; N=${NTRAIN:-1000}; S=$OUT/self${N}
mkdir -p "$S"; record_source "$S/src_${SLURM_JOB_ID}"
echo "=== reader + agent head $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_reader.py" --cache "$CACHE" --events "$OUT/events" --agent-head --entities "$OUT/front" --out "$S/reader"
echo "=== reader entities ($N episodes) $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_reader_entities.py" --cache "$CACHE" --front "$OUT/front" --reader "$S/reader/u_reader.pt" \
  --train-episodes "$N" --val-episodes 100 --out "$S/front"
echo "=== events $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_events.py" --entities "$S/front" --cache "$CACHE" --thresholds-from "$OUT/events" --out "$S/events"
echo "=== WM (background) + refine + skill $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_wm.py" --events "$S/events" --out "$S/wm" ${WM_ARGS:-} > "$S/wm.log" 2>&1 &
P=$!
"$TORCH_PY" "$PROJECT/scripts/u_refine.py" --cache "$CACHE" --events "$S/events" --reader "$S/reader/u_reader.pt" --out "$S/events_ref"
TF=$(( (N + 2) * 1001 ))
"$TORCH_PY" "$PROJECT/scripts/u_skill.py" --cache "$CACHE" --events "$S/events_ref" --train-frames "$TF" --val-frames 101000 --out "$S/skill" ${SKILL_ARGS:-}
wait $P
tail -n 2 "$S/wm.log"
echo "=== done $(date -u +%T)"
