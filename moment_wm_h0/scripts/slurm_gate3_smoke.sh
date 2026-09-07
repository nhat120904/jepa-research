#!/usr/bin/env bash
#SBATCH --job-name=mwm_g3_smoke
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g3_smoke_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
DATA="$PROJECT/outputs/gate3_smoke/control.pt"
FEATURES="$PROJECT/outputs/gate3_smoke/features.pt"
MODELS="$PROJECT/outputs/gate3_smoke/models"
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
sha256sum "$PROJECT/models.py" \
          "$PROJECT/scripts/generate_control_dataset.py" \
          "$PROJECT/scripts/prepare_control_features.py" \
          "$PROJECT/scripts/train_strong_baseline.py" \
          "$PROJECT/scripts/evaluate_strong_baselines.py" \
          "$PROJECT/scripts/slurm_gate3_smoke.sh"
"$PY" "$PROJECT/scripts/generate_control_dataset.py" \
  --out "$DATA" --episodes 128 --queries 4 --batch-size 64 --device cuda
"$PY" "$PROJECT/scripts/prepare_control_features.py" \
  --dataset "$DATA" --out "$FEATURES" --batch-size 48 \
  --feature-dim 32 --pca-samples 2000 --eval-episodes 16 --device cuda
for arm in mse cov_mse cadm; do
  "$PY" "$PROJECT/scripts/train_strong_baseline.py" \
    --features "$FEATURES" --arm "$arm" --seed 0 \
    --out-dir "$MODELS/$arm/seed_0" \
    --epochs 2 --patience 2 --batch-size 64 --device cuda
done
"$PY" "$PROJECT/scripts/evaluate_strong_baselines.py" \
  --features "$FEATURES" --checkpoints-root "$MODELS" \
  --gate1-actions "$PROJECT/outputs/gate1_full/actions.pt" \
  --out-dir "$PROJECT/outputs/gate3_smoke/evaluation" \
  --arms mse cov_mse cadm --seeds 0 --num-samples 64 \
  --iterations 2 --restarts 1 --bootstrap 200 --device cuda
