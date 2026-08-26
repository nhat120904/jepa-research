#!/usr/bin/env bash
#SBATCH --job-name=acm_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_smoke_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/scripts/_common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. offline numerics must pass before any GPU measurement is trusted.
# The test file runs standalone: the compute venv has no pytest, and adding a
# dependency to gate a measurement is a worse trade than dropping the runner.
"$PY" "$PROJECT/tests/test_core.py"
"$PY" "$PROJECT/scripts/aggregate.py" --self-test --shard-root /dev/null --out-dir /dev/null

# 2. one snapshot, minimal grid, both action sources
for SOURCE in dataset cem_fixed cem_local; do
  "$PY" "$PROJECT/scripts/measure_curvature.py" \
    --snapshot-index 0 --manifest "$MANIFEST" --population-dir "$POPULATIONS" \
    --action-source "$SOURCE" --horizons 1,5 --sigmas 0.00125,0.0025,0.005,0.01,0.025,0.05,0.10,0.20 \
    --n-directions 2 --repeats 3 \
    --out-dir "$PROJECT/outputs/smoke/$SOURCE/snapshot_000"
done

# 3. aggregate the smoke shards end to end
"$PY" "$PROJECT/scripts/aggregate.py" \
  --shard-root "$PROJECT/outputs/smoke" \
  --out-dir "$PROJECT/outputs/smoke/aggregate" --n-resamples 200

echo "SMOKE COMPLETE $(date -u +%FT%TZ)"
