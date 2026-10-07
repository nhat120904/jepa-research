#!/usr/bin/env bash
#SBATCH --job-name=ew_sm
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Scene-memory v1: train sm_model on pixel play (same hyperparameters for every family), then export codes for
# val/train and score them on val with PRIVILEGED state (sm_diag.py).
#   FAMILY=cube|puzzle|scene ENV=<visual play env> [STEPS=20000 EPISODES=1000 EXPORT_EPISODES=1000] sbatch slurm/sm_pipeline.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
FAMILY=${FAMILY:?FAMILY}; ENV=${ENV:?ENV}; CACHE=$RUN_ROOT/cache/$ENV
STEPS=${STEPS:-20000}; EPISODES=${EPISODES:-1000}; EXPORT_EPISODES=${EXPORT_EPISODES:-$EPISODES}
OUT=$RUN_ROOT/scene_memory_v1/${FAMILY}_${SLURM_JOB_ID}
mkdir -p "$OUT"; record_source "$OUT/src"
echo "=== train $(date -u +%T) steps=$STEPS episodes=$EPISODES"
"$TORCH_PY" "$PROJECT/scripts/sm_train.py" --cache "$CACHE" --episodes "$EPISODES" --steps "$STEPS" --out "$OUT/train"
echo "=== export + diagnostics $(date -u +%T)"
"$TORCH_PY" "$PROJECT/scripts/sm_diag.py" --model "$OUT/train/sm.pt" --cache "$CACHE" --family "$FAMILY" \
  --episodes "$EXPORT_EPISODES" --out "$OUT/diag"
echo "=== done $(date -u +%T)"
