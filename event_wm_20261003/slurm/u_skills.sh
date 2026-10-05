#!/usr/bin/env bash
#SBATCH --job-name=ew_uskills
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Re-refine event timing (current u_refine rules) and train skill variants concurrently on one GPU.
#   ENV=<play env> S=<self-training dir with events/ reader/> NTRAIN=1000 VARIANTS="full target" STEPS=60000 sbatch slurm/u_skills.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export OMP_NUM_THREADS=3
ENV=${ENV:?ENV}; CACHE=$RUN_ROOT/cache/$ENV; S=${S:?S}; N=${NTRAIN:-1000}; TAGR=${TAGR:-ref2}
record_source "$S/src_skills_${SLURM_JOB_ID}"
"$TORCH_PY" "$PROJECT/scripts/u_refine.py" --cache "$CACHE" --events "$S/events" --reader "$S/reader/u_reader.pt" --out "$S/events_$TAGR"
TF=$(( (N + 2) * 1001 ))
pids=()
for V in ${VARIANTS:-full target}; do
  "$TORCH_PY" "$PROJECT/scripts/u_skill.py" --cache "$CACHE" --events "$S/events_$TAGR" --train-frames "$TF" --val-frames 101000 \
    --steps "${STEPS:-60000}" --cond "$V" --out "$S/skill_${V}_${TAGR}" > "$S/skill_${V}_${TAGR}.log" 2>&1 &
  pids+=($!)
done
st=0; for p in "${pids[@]}"; do wait $p || st=1; done
tail -n 2 "$S"/skill_*_"$TAGR".log
exit $st
