#!/usr/bin/env bash
#SBATCH --job-name=ew_stwm
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# State track: retrain only the world model + cost-to-go of an earlier s_pipeline.sh run with WM_ARGS (e.g.
# --event-pos-only), keep its entities, events and skill, and rerun the closed loop: dev (seed 0), the earlier
# protocol seed 1 (paired comparison with the old model) and the fresh protocol seed 2 (the clean number).
#   OUT=<state run dir> ENV=cube-triple-play-v0 LOOP_ENV=cube-triple-v0 WM_ARGS=--event-pos-only sbatch slurm/s_wm_rerun.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
export OMP_NUM_THREADS=2
OUT=${OUT:?OUT}; ENV=${ENV:?ENV}; LOOP_ENV=${LOOP_ENV:?LOOP_ENV}; CACHE=$OUT/cache/$ENV
W=$OUT/wm_${SLURM_JOB_ID}
mkdir -p "$W"; record_source "$W/src"
echo "=== WM ${WM_ARGS:-} $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/u_wm.py" --events "$OUT/events" --out "$W" ${WM_ARGS:-}
LOOP=("$TORCH_PY" "$PROJECT/scripts/u_closed_loop.py" --env "$LOOP_ENV" --model "$W/u_model.pt" --state-layout "$OUT/front/layout.json"
      --skill "$OUT/skill/u_skill.pt" --events "$OUT/events" --cache "$CACHE" --workers 10)
for s in 0 1 2; do
  E=20; [ "$s" = 0 ] && E=6
  echo "=== closed loop seed $s, $E episodes per task $(date -u +%T)"
  "${LOOP[@]}" --episodes "$E" --seed "$s" --out "$W/loop_seed$s"
done
echo "=== done $(date -u +%T)"
