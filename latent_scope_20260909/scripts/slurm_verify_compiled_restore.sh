#!/usr/bin/env bash
#SBATCH --job-name=lscope_binary_cpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/compiled_restore_%j.out
set -euo pipefail
PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
RUNTIME_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_baseline
SHARED_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_a
export OMP_NUM_THREADS=8
env MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  PYTHONPATH="$PROJECT_ROOT/latent_scope_20260909/scripts:$SHARED_ROOT/src/robocasa365:$SHARED_ROOT/src/robosuite" \
  "$RUNTIME_ROOT/sim_venv/bin/python" "$PROJECT_ROOT/latent_scope_20260909/scripts/verify_compiled_restore.py" \
  --source "$RUNTIME_ROOT/outputs/qualification_52410/prefix_attempt_00/source_snapshot.pkl" \
  --config "$RUNTIME_ROOT/outputs/qualification_52410/config.json" \
  --output "$RUNTIME_ROOT/outputs/compiled_restore_$SLURM_JOB_ID.json"
