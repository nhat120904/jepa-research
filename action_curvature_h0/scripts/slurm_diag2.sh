#!/usr/bin/env bash
#SBATCH --job-name=acm_diag2
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_diag2_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/scripts/_common.sh
# Diagnostic-only: one snapshot known to still fail centre clipping (job
# 46123_2), one horizon, one sigma, to capture clip_diagnostic without a full
# sweep.
"$PY" "$PROJECT/scripts/measure_curvature.py" \
  --snapshot-index 2 --n-snapshots 64 \
  --manifest "$MANIFEST" --population-dir "$POPULATIONS" \
  --action-source cem_fixed --horizons 5 --sigmas 0.00125 \
  --n-directions 2 --repeats 2 \
  --out-dir "$PROJECT/outputs/diag2/cem_fixed/snapshot_002"
python3 -c "
import json
s = json.load(open('$PROJECT/outputs/diag2/cem_fixed/snapshot_002/summary.json'))
print(json.dumps(s['direction_feasibility'], indent=2))
print(json.dumps(s['centre_clipping'], indent=2))
print(json.dumps(s['clip_diagnostic'], indent=2))
"
