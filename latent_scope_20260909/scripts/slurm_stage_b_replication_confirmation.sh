#!/usr/bin/env bash
#SBATCH --job-name=lscope_rep_confirm
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/seed_confirm_%j.out
set -euo pipefail
PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
RUNTIME_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_baseline
SHARED_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_a
OUTPUT_DIR="$RUNTIME_ROOT/outputs/seed_confirm_$SLURM_JOB_ID"
mkdir "$OUTPUT_DIR"
# Reuse the exact simulator/policy helpers used in 52524.
cp -a "$RUNTIME_ROOT/outputs/qualification_52524/scripts" "$OUTPUT_DIR/scripts"
cp "$PROJECT_ROOT/latent_scope_20260909/scripts/run_stage_b_replication_confirmation.py" "$OUTPUT_DIR/scripts/"
cp "$RUNTIME_ROOT/outputs/qualification_52524/config.json" "$OUTPUT_DIR/config.json"
sha256sum "$OUTPUT_DIR/scripts/"*.py "$OUTPUT_DIR/config.json" > "$OUTPUT_DIR/code_sha256.txt"
export SEED_CONFIRM_STARTED_UNIX
SEED_CONFIRM_STARTED_UNIX=$(date +%s)
export HF_HOME="$RUNTIME_ROOT/hf_cache" TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MESA_SHADER_CACHE_DIR="$RUNTIME_ROOT/mesa_shader_cache"
PORT=$((20000 + SLURM_JOB_ID % 20000))
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID"
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
env PYTHONPATH="$RUNTIME_ROOT/src/Isaac-GR00T" \
 "$RUNTIME_ROOT/policy_venv/bin/python" "$OUTPUT_DIR/scripts/run_groot_policy_server.py" \
 --config "$OUTPUT_DIR/config.json" --state-output "$OUTPUT_DIR/policy_server_state.json" \
 --port "$PORT" > "$OUTPUT_DIR/policy_server.log" 2>&1 &
SERVER_PID=$!
env MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
 PYTHONPATH="$OUTPUT_DIR/scripts:$SHARED_ROOT/src/robocasa365:$SHARED_ROOT/src/robosuite" \
 "$RUNTIME_ROOT/sim_venv/bin/python" "$OUTPUT_DIR/scripts/run_stage_b_replication_confirmation.py" \
 --output-dir "$OUTPUT_DIR" --port "$PORT" --server-pid "$SERVER_PID" \
 --server-state "$OUTPUT_DIR/policy_server_state.json"
