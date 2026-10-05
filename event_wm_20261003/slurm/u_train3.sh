#!/usr/bin/env bash
#SBATCH --job-name=ew_utrain
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Unified backend training on one GPU: reader and world model + cost-to-go concurrently; after the reader,
# event-timing refinement (u_refine) and the skill on the refined segments.
#   ENV=<play env> OUT=<run dir with events/> sbatch slurm/u_train3.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export OMP_NUM_THREADS=2
ENV=${ENV:?ENV}; CACHE=$RUN_ROOT/cache/$ENV; OUT=${OUT:?OUT}
record_source "$OUT/src_train_${SLURM_JOB_ID}"
"$TORCH_PY" "$PROJECT/scripts/u_reader.py" --cache "$CACHE" --events "$OUT/events" --out "$OUT/reader" ${READER_ARGS:-} > "$OUT/reader.log" 2>&1 &
P1=$!
"$TORCH_PY" "$PROJECT/scripts/u_wm.py" --events "$OUT/events" --out "$OUT/wm" ${WM_ARGS:-} > "$OUT/wm.log" 2>&1 &
P2=$!
st=0
wait $P1 || st=1
# refine event timing with the reader, then train the skill on the refined segments
if [ $st = 0 ]; then
  "$TORCH_PY" "$PROJECT/scripts/u_refine.py" --cache "$CACHE" --events "$OUT/events" --reader "$OUT/reader/u_reader.pt" --out "$OUT/events_ref" > "$OUT/refine.log" 2>&1 || st=1
fi
if [ $st = 0 ]; then
  "$TORCH_PY" "$PROJECT/scripts/u_skill.py" --cache "$CACHE" --events "$OUT/events_ref" --out "$OUT/skill" ${SKILL_ARGS:-} > "$OUT/skill.log" 2>&1 || st=1
fi
wait $P2 || st=1
tail -n 3 "$OUT/reader.log" "$OUT/wm.log" "$OUT/refine.log" "$OUT/skill.log"
exit $st
