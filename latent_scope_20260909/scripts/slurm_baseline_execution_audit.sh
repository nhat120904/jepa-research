#!/usr/bin/env bash
#SBATCH --job-name=lscope_base_audit
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/base_audit_%j.out
set -euo pipefail
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909
RUNTIME=/mnt/data/nhatnc129/jepa/latent_scope_baseline
SHARED=/mnt/data/nhatnc129/jepa/latent_scope_stage_a
OUTPUT="$RUNTIME/outputs/base_audit_$SLURM_JOB_ID"
mkdir -p "$OUTPUT/code"
cp "$PROJECT/scripts/audit_baseline_execution.py" "$PROJECT/scripts/run_baseline_sim_client.py" "$PROJECT/scripts/run_groot_policy_server.py" "$OUTPUT/code/"
cp "$PROJECT/configs/baseline_policy_gate.json" "$OUTPUT/config.json"
export HF_HOME="$RUNTIME/hf_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
PORT=$((20000 + SLURM_JOB_ID % 20000))
env PYTHONPATH="$RUNTIME/src/Isaac-GR00T" "$RUNTIME/policy_venv/bin/python" \
    "$OUTPUT/code/run_groot_policy_server.py" --config "$OUTPUT/config.json" \
    --state-output "$OUTPUT/policy_server_state.json" --port "$PORT" \
    >"$OUTPUT/policy_server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM
env MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    PYTHONPATH="$SHARED/src/robocasa365:$SHARED/src/robosuite" \
    "$RUNTIME/sim_venv/bin/python" "$OUTPUT/code/audit_baseline_execution.py" \
    --config "$OUTPUT/config.json" --output-dir "$OUTPUT" \
    --server-pid "$SERVER_PID" --port "$PORT"
