#!/usr/bin/env bash
#SBATCH --job-name=jepa_costval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/costval_%A_%a.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
export PATH="$PWD/.venv/bin:$PATH"

MODELS=(dino_wm_metaworld jepa_wm_metaworld)
TAGS=(dino jepa)
I=${SLURM_ARRAY_TASK_ID:?submit as the declared array}
MODEL=${MODELS[$I]}
TAG=${TAGS[$I]}

.venv/bin/python scripts/79_validate_stateprobe_scalar_cost.py \
  --config configs/diagnostic_metaworld.yaml --model "$MODEL" \
  --object-probe "checkpoints/spatial_object_probe_${MODEL}_offpolicy.pt" \
  --ee-probe "checkpoints/ee_probe_${MODEL}_offpolicy.pt" \
  --step 5 --val-frac 0.1 --split-seed 0 --w-hand 0.5 \
  --n-bootstrap 10000 --out-prefix "results/stateprobe_scalar_validation_${TAG}"
