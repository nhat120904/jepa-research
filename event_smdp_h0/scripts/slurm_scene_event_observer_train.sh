#!/usr/bin/env bash
#SBATCH --job-name=scene_event_observer_train
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_event_observer_train_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
INDEX=${SLURM_ARRAY_TASK_ID:?}
MODEL_SEED=${MODEL_SEED:-0}
if (( INDEX == 0 )); then VIEW=latent; else VIEW=privileged; fi
OUT="$PROJECT/outputs/scene_event_perception/checkpoints/${VIEW}/seed${MODEL_SEED}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} ARRAY=${SLURM_ARRAY_JOB_ID} INDEX=$INDEX VIEW=$VIEW MODEL_SEED=$MODEL_SEED $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_event_perception.py" \
  "$PROJECT/scripts/train_scene_event_observer.py" \
  "$PROJECT/scripts/slurm_scene_event_observer_train.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/train_scene_event_observer.py" \
  --data-root "$PROJECT/outputs/scene_h1/data" --feature-view "$VIEW" \
  --seed "$MODEL_SEED" --out-dir "$OUT"

