#!/usr/bin/env bash
#SBATCH --job-name=spwm_stage0
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_stage0_%j.out
set -euo pipefail

source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
RUN_ID=${RUN_ID:-stage0_20260904}
sha256sum "$PROJECT/scripts/stage0_plumbing.py" "$PROJECT/scripts/slurm_stage0.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

"$PY" "$PROJECT/scripts/stage0_plumbing.py" \
  --dataset "$DATA_ROOT/visual-scene-play-v0.npz" \
  --num-frames 24 \
  --num-pairs 12 \
  --goal-offset 100 \
  --episodes 2 \
  --render-size 64 \
  --camera front_pixels \
  --seed 90000 \
  --out-dir "$PROJECT/outputs/stage0/diagnostic/$RUN_ID"

echo "DONE $(date -u +%FT%TZ)"
