#!/usr/bin/env bash
#SBATCH --job-name=lscope_q_scrub
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=07:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/qualification_%j.out
set -euo pipefail
PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
RUNTIME_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_baseline
SHARED_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_a
OUTPUT_DIR="$RUNTIME_ROOT/outputs/qualification_$SLURM_JOB_ID"
CONFIG="$PROJECT_ROOT/latent_scope_20260909/configs/stage_b_qualification.json"
PORT=$((20000 + SLURM_JOB_ID % 20000))
: "${QUALIFICATION_TEST_JOB:?Submit only after CPU invariant tests}"
: "${QUALIFICATION_RESTORE_JOB:?Submit only after compiled-snapshot CPU verification}"
RESTORE_PROOF="$RUNTIME_ROOT/outputs/compiled_restore_${QUALIFICATION_RESTORE_JOB}.json"
rg -q '"verdict": "COMPILED_RESTORE_PASS"' "$RESTORE_PROOF"
test -f "$RUNTIME_ROOT/outputs/qualification_tests_${QUALIFICATION_TEST_JOB}.json"
rg -q '"verdict": "QUALIFICATION_INVARIANTS_PASS"' "$RUNTIME_ROOT/outputs/qualification_tests_${QUALIFICATION_TEST_JOB}.json"
mkdir -p "$OUTPUT_DIR"
export QUALIFICATION_STARTED_UNIX
QUALIFICATION_STARTED_UNIX=$(date +%s)
export HF_HOME="$RUNTIME_ROOT/hf_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MESA_SHADER_CACHE_DIR="$RUNTIME_ROOT/mesa_shader_cache"
# Freeze the implementation/config actually used; later workspace edits cannot change this job.
cp -a "$PROJECT_ROOT/latent_scope_20260909/scripts" "$OUTPUT_DIR/scripts"
cp "$CONFIG" "$OUTPUT_DIR/config.json"
CONFIG="$OUTPUT_DIR/config.json"
sha256sum "$OUTPUT_DIR/scripts/"*.py "$CONFIG" > "$OUTPUT_DIR/code_sha256.txt"
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
  --config "$CONFIG" --state-output "$OUTPUT_DIR/policy_server_state.json" \
  --port "$PORT" > "$OUTPUT_DIR/policy_server.log" 2>&1 &
SERVER_PID=$!
env MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  PYTHONPATH="$OUTPUT_DIR/scripts:$SHARED_ROOT/src/robocasa365:$SHARED_ROOT/src/robosuite" \
  "$RUNTIME_ROOT/sim_venv/bin/python" "$OUTPUT_DIR/scripts/run_stage_b_qualification.py" \
  --config "$CONFIG" --output-dir "$OUTPUT_DIR" --server-pid "$SERVER_PID" \
  --server-state "$OUTPUT_DIR/policy_server_state.json" --port "$PORT" \
  --restore-proof "$RESTORE_PROOF"
