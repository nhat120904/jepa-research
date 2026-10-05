#!/usr/bin/env bash
#SBATCH --job-name=ew_cubeD
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Cube direction D (PRIVILEGED object state): move events -> event WM + affordance -> pick-and-place skill.
#   PART=train sbatch slurm/cube_d.sh ;  PART=loop OUT=<dir> sbatch --cpus-per-task=16 --mem=96G slurm/cube_d.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=visual-cube-triple-play-v0; CACHE=$RUN_ROOT/cache/$ENV
OUT=${OUT:-$RUN_ROOT/cubeD_${SLURM_JOB_ID}}
record_source "$OUT/src_${SLURM_JOB_ID}"
case ${PART:?PART} in
  train)
    "$TORCH_PY" "$PROJECT/scripts/cube_events.py" --cache "$CACHE" --kind triple --out "$OUT/events"
    "$TORCH_PY" "$PROJECT/scripts/cube_planner.py" --events "$OUT/events" --out "$OUT/planner"
    "$TORCH_PY" "$PROJECT/scripts/train_cube_skill.py" --cache "$CACHE" --events "$OUT/events" --out "$OUT/skill" ${SKILL_ARGS:-}
    ;;
  loop)
    IFS=";" read -ra LIST <<< "${ARMS:-learned}"
    for HIGH in "${LIST[@]}"; do
      "$TORCH_PY" "$PROJECT/scripts/cube_closed_loop.py" --planner "$OUT/planner/cube_planner.pt" --skill "$OUT/skill/cube_skill.pt" \
        --low "$HIGH" --episodes "${EPISODES:-6}" --workers 14 --events "$OUT/events" --cache "$CACHE" --out "$OUT/loop_${HIGH}${TAG:-}" ${LOOP_ARGS:-}
    done
    ;;
esac
