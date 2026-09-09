#!/usr/bin/env bash
#SBATCH --job-name=erwm_e0_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/erwm_e0_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_resolved_20260908"
RUNTIME=/mnt/data/nhatnc129/jepa/erwm_e0
OUT="$PROJECT/outputs/gate_e0/${SLURM_JOB_ID}"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
mkdir -p "$OUT" /mnt/data/nhatnc129/jepa_runs/logs
cd "$PROJECT"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sha256sum \
  strike_slide_env.py \
  scripts/run_gate_e0.py \
  scripts/slurm_gate_e0_smoke.sh

"$RUNTIME/.venv/bin/python" scripts/run_gate_e0.py \
  --out "$OUT/result.json" \
  --seeds "${E0_SEEDS:-8}" \
  --candidates "${E0_CANDIDATES:-24}"
