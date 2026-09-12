#!/usr/bin/env bash
#SBATCH --job-name=lscope_c_audit
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/stage_c_audit_%j.out

set -euo pipefail

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
STAGE_C_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_c
SIM_PYTHON=/mnt/data/nhatnc129/jepa/latent_scope_stage_a/venv/bin/python
OUTPUT="$STAGE_C_ROOT/readiness_$SLURM_JOB_ID.json"
ARGS=()
if [[ -n "${STAGE_B1_RESULT:-}" ]]; then
    ARGS+=(--stage-b1-result "$STAGE_B1_RESULT")
fi
mkdir -p "$STAGE_C_ROOT"

"$SIM_PYTHON" \
    "$PROJECT_ROOT/latent_scope_20260909/scripts/audit_stage_c_readiness.py" \
    --config "$PROJECT_ROOT/latent_scope_20260909/configs/stage_c_screen.json" \
    --output "$OUTPUT" \
    "${ARGS[@]}"
