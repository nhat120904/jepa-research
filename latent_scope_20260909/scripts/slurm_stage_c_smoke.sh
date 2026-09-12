#!/usr/bin/env bash
#SBATCH --job-name=lscope_c_smoke
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/stage_c_smoke_%j.out

set -euo pipefail

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
POLICY_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_baseline
STAGE_C_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_c
OUTPUT="$STAGE_C_ROOT/smoke_$SLURM_JOB_ID.json"
mkdir -p "$STAGE_C_ROOT/logs"

env PYTHONPATH="$PROJECT_ROOT/latent_scope_20260909" \
    "$POLICY_ROOT/policy_venv/bin/python" \
    "$PROJECT_ROOT/latent_scope_20260909/scripts/smoke_stage_c_models.py" \
    --output "$OUTPUT"
