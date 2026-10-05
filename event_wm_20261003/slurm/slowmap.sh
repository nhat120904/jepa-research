#!/usr/bin/env bash
#SBATCH --job-name=ew_slowmap
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Unified-method probe 2: deep spatial slow-feature map, identical hyperparameters per task family.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
OUT=${OUT:-$RUN_ROOT/slowmap_${SLURM_JOB_ID}}
record_source "$OUT/src"
for ENV in ${ENVS:-visual-puzzle-4x5-play-v0 visual-cube-triple-play-v0}; do
  EXTRA=""; [[ "$ENV" == visual-cube-triple-play-v0 ]] && EXTRA="--ref-events $RUN_ROOT/cubeD_57116/events"
  EV_ONLY=""; [[ -f "$OUT/$ENV/slowmap.pt" ]] && EV_ONLY="--eval-only"
  "$TORCH_PY" "$PROJECT/scripts/slowmap.py" --cache "$RUN_ROOT/cache/$ENV" --out "$OUT/$ENV" $EXTRA $EV_ONLY ${SLOW_ARGS:-}
  if [[ -f "$RUN_ROOT/cache/$ENV/val_qpos.npy" ]]; then
    "$TORCH_PY" "$PROJECT/scripts/slowmap_probe.py" --cache "$RUN_ROOT/cache/$ENV" --slowmap "$OUT/$ENV/slowmap.pt" --out "$OUT/$ENV"
  fi
done
