#!/usr/bin/env bash
#SBATCH --job-name=ar_three_seed
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/action_response/logs/%x_%j.out
set -euo pipefail

PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/action_response
"$PY" /home/nhatnc129/nhat.nc/jepa-research/action_response_20260927/aggregate.py \
  "$ROOT/run_55392_20260928T014050Z" \
  "$ROOT/run_55395_20260928T014356Z" \
  "$ROOT/run_55398_20260928T014725Z" \
  --output "$ROOT/three_seed_endpoint_summary.json"
