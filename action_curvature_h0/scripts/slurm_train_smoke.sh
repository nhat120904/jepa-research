#!/usr/bin/env bash
#SBATCH --job-name=acm_trsmoke
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/acm_trsmoke_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/action_curvature_h0/scripts/_common.sh
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# 20 steps only: verify the data window, the triplet-on-S-axis rollout, the
# freeze, and that both arms run the identical path before spending seeds.
for LAM in 0 0.1; do
  "$PY" "$PROJECT/scripts/train_as.py" --lambda-as "$LAM" --seed 0 \
    --steps 20 --batch-size 8 --log-every 5 \
    --out-dir "$PROJECT/outputs/train_smoke/lam${LAM}_seed0"
done
echo "TRAIN SMOKE COMPLETE $(date -u +%FT%TZ)"
