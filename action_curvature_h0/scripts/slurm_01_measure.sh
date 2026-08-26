#!/usr/bin/env bash
#SBATCH --job-name=acm_measure
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-63%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_measure_%A_%a.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/scripts/_common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
TASK_ID=${SLURM_ARRAY_TASK_ID:?}
SOURCE=${ACM_SOURCE:?set ACM_SOURCE to dataset, cem_fixed or cem_local}

"$PY" "$PROJECT/scripts/measure_curvature.py" \
  --snapshot-index "$TASK_ID" --n-snapshots 64 \
  --manifest "$MANIFEST" --population-dir "$POPULATIONS" \
  --action-source "$SOURCE" \
  --horizons 1,3,5 --sigmas 0.00125,0.0025,0.005,0.01,0.025,0.05,0.10,0.20 \
  --n-directions 8 --repeats 3 \
  --out-dir "$PROJECT/outputs/stage1/$SOURCE/snapshot_$(printf '%03d' "$TASK_ID")"
