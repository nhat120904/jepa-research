#!/usr/bin/env bash
#SBATCH --job-name=lscope_c_train
#SBATCH --partition=main
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/stage_c_train_%A_%a.out

set -euo pipefail

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
POLICY_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_baseline
STAGE_C_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_c
: "${STAGE_B1_RESULT:?Stage C requires a positive Stage-B1 result path}"
: "${STAGE_C_CAMPAIGN_ID:?Set a stable campaign identifier}"

ARMS=(
    endpoint_only
    frame_rollout
    simple_progress
    unstructured_segment
    compositional_segment
    direct_value
)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="$STAGE_C_ROOT/runs/$STAGE_C_CAMPAIGN_ID/$ARM"
mkdir -p "$OUTPUT_DIR" "$STAGE_C_ROOT/logs"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

env PYTHONPATH="$PROJECT_ROOT/latent_scope_20260909" \
    STAGE_B1_RESULT="$STAGE_B1_RESULT" \
    "$POLICY_ROOT/policy_venv/bin/python" \
    "$PROJECT_ROOT/latent_scope_20260909/scripts/run_stage_c_train.py" \
    --config "$PROJECT_ROOT/latent_scope_20260909/configs/stage_c_screen.json" \
    --arm "$ARM" \
    --output-dir "$OUTPUT_DIR"
