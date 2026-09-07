#!/usr/bin/env bash
#SBATCH --job-name=mwm_g1_full
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g1_full_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
OUT="$PROJECT/outputs/gate1_full"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sha256sum "$PROJECT/scripts/gate1_decision_room.py" \
          "$PROJECT/scripts/slurm_gate1_full.sh"
"$PY" "$PROJECT/scripts/gate1_decision_room.py" \
  --out-dir "$OUT" \
  --episodes 128 \
  --num-samples 256 \
  --iterations 6 \
  --restarts 2 \
  --bootstrap 20000 \
  --device cuda
