#!/usr/bin/env bash
#SBATCH --job-name=ew_slots
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Unified front end candidate 3: SlotContrast-style slots on frozen DINOv2-reg-S; one configuration for all task families.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
OUT=${OUT:-$RUN_ROOT/slots_${SLURM_JOB_ID}}
record_source "$OUT/src"
for ENV in ${ENVS:-visual-cube-triple-play-v0}; do
  EV_ONLY=""; [[ -f "$OUT/$ENV/slots.pt" ]] && EV_ONLY="--eval-only"
  "$TORCH_PY" "$PROJECT/scripts/slots.py" --cache "$RUN_ROOT/cache/$ENV" --out "$OUT/$ENV" $EV_ONLY ${SLOT_ARGS:-}
done
