#!/usr/bin/env bash
#SBATCH --job-name=lscope_q_tests
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/qualification_tests_%j.out
set -euo pipefail
PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
RUNTIME_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_baseline
export OMP_NUM_THREADS=2
env PYTHONPATH="$PROJECT_ROOT/latent_scope_20260909" \
  "$RUNTIME_ROOT/policy_venv/bin/python" \
  "$PROJECT_ROOT/latent_scope_20260909/scripts/test_qualification_invariants.py" \
  --output "$RUNTIME_ROOT/outputs/qualification_tests_$SLURM_JOB_ID.json"
env PYTHONPATH="$PROJECT_ROOT/latent_scope_20260909" \
  "$RUNTIME_ROOT/policy_venv/bin/python" \
  "$PROJECT_ROOT/latent_scope_20260909/scripts/probe_stage_c_targets.py" \
  --config "$PROJECT_ROOT/latent_scope_20260909/configs/stage_c_screen.json" \
  --output "$RUNTIME_ROOT/outputs/qualification_target_probe_$SLURM_JOB_ID.json"
