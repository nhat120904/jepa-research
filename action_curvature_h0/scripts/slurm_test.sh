#!/usr/bin/env bash
#SBATCH --job-name=acm_test
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_test_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/scripts/_common.sh
"$PY" "$PROJECT/tests/test_core.py"
"$PY" "$PROJECT/scripts/aggregate.py" --self-test --shard-root /dev/null --out-dir /dev/null
echo "OFFLINE NUMERICS OK $(date -u +%FT%TZ)"
