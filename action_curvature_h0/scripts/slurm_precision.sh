#!/usr/bin/env bash
#SBATCH --job-name=acm_prec
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_prec_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/scripts/_common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Same snapshot, same sweep, same directions, same centre: only the model
# forward precision differs.  Tests whether the ||D2 Phi|| floor at ~2e-3 is
# numerical (float64 removes it, alpha -> 2) or a genuine property of the model.
for DT in float32 float64; do
  "$PY" "$PROJECT/scripts/measure_curvature.py" \
    --snapshot-index 0 --n-snapshots 64 \
    --manifest "$MANIFEST" --population-dir "$POPULATIONS" \
    --action-source dataset --horizons 1 \
    --sigmas 0.00125,0.0025,0.005,0.01,0.025,0.05,0.10,0.20 \
    --n-directions 2 --repeats 2 --model-dtype "$DT" \
    --out-dir "$PROJECT/outputs/precision/$DT/snapshot_000"
done
echo "PRECISION SWEEP COMPLETE $(date -u +%FT%TZ)"
