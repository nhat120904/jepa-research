#!/usr/bin/env bash
#SBATCH --job-name=perd_train_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/perd_train_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/physical_search_distillation"
STAGE0_ROOT=/mnt/data/nhatnc129/jepa/lewm_stage0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/core.py" "$PROJECT/scripts/train_models.py" "$PROJECT/scripts/slurm_02_train_smoke.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$STAGE0_ROOT/.venv/bin/python" "$PROJECT/scripts/train_models.py" \
  --data-dir "$PROJECT/outputs/h0/populations" --out-dir "$PROJECT/outputs/h0/checkpoints_smoke" \
  --arms operator operator_metric --seeds 11 --epochs 1 --patience 1
