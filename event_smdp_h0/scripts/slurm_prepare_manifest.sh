#!/usr/bin/env bash
#SBATCH --job-name=event_manifest
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/event_manifest_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
export STABLEWM_HOME="$RUNTIME"
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/prepare_manifest.py" "$PROJECT/scripts/slurm_prepare_manifest.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/prepare_manifest.py" \
  --exclude-manifest "$REPO/diagnosis/results/ogb_stage0/audit_locked/manifest.json" \
  --exclude-manifest "$REPO/counterfactual_flow/outputs/ogbench_cube_phase1a/manifest.json" \
  --exclude-manifest "$REPO/counterfactual_flow/outputs/ogbench_cube_phase1av2/manifest.json" \
  --num-snapshots 65 --goal-offset 25 --min-task-distance-m 0.08 \
  --out "$PROJECT/outputs/gate0/manifest.json"
