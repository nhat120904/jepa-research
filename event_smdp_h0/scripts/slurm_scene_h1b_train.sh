#!/usr/bin/env bash
#SBATCH --job-name=scene_h1b_train
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_h1b_train_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
MODEL_SEED=${MODEL_SEED:-0}
OUT="$PROJECT/outputs/scene_h1b/checkpoints/seed${MODEL_SEED}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} MODEL_SEED=$MODEL_SEED $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_abstract_smdp.py" "$PROJECT/scripts/train_scene_h1b.py" \
  "$PROJECT/scripts/slurm_scene_h1b_train.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/train_scene_h1b.py" \
  --data-root "$PROJECT/outputs/scene_h1/data" --seed "$MODEL_SEED" --out-dir "$OUT"

