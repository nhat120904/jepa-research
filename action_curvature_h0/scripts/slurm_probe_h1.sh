#!/usr/bin/env bash
#SBATCH --job-name=acm_probe_h1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --array=0-7%4
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_probe_h1_%A_%a.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/scripts/_common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
TASK_ID=${SLURM_ARRAY_TASK_ID:?}

# Characterisation probe, not a Stage-1 measurement: one horizon, the full scale
# sweep, two directions, across eight snapshots.  The question is whether the
# observation-path resolution floor found at snapshot 0 (cube grasped) is global
# or specific to snapshots where the cube is kinematically pinned to the
# gripper.  Both action sources are run so the CEM centre-clipping fix is
# exercised at the same time.
for SOURCE in dataset cem_fixed; do
  "$PY" "$PROJECT/scripts/measure_curvature.py" \
    --snapshot-index "$TASK_ID" --n-snapshots 64 \
    --manifest "$MANIFEST" --population-dir "$POPULATIONS" \
    --action-source "$SOURCE" --horizons 1 \
    --sigmas 0.00125,0.0025,0.005,0.01,0.025,0.05,0.10,0.20 \
    --n-directions 2 --repeats 2 \
    --out-dir "$PROJECT/outputs/probe_h1/$SOURCE/snapshot_$(printf '%03d' "$TASK_ID")"
done
echo "PROBE TASK $TASK_ID COMPLETE $(date -u +%FT%TZ)"
