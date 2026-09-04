#!/usr/bin/env bash
#SBATCH --job-name=scene_event_ablation_train
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_event_ablation_train_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
SEEDS=${SEEDS:-0 1 2}
export STABLEWM_HOME="$RUNTIME"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} SEEDS='${SEEDS}' $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_event_history.py" "$PROJECT/scene_history_dataset.py" \
  "$PROJECT/scripts/train_scene_event_history_observer.py" \
  "$PROJECT/scripts/slurm_scene_event_ablation_train.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for ABLATION in obs_history action_only; do
  for SEED in $SEEDS; do
    OUT="$PROJECT/outputs/scene_event_history/checkpoints/${ABLATION}_full/seed${SEED}"
    echo "ABLATION=$ABLATION SEED=$SEED START=$(date -u +%FT%TZ)"
    "$RUNTIME/.venv/bin/python" \
      "$PROJECT/scripts/train_scene_event_history_observer.py" \
      --data-root "$PROJECT/outputs/scene_h1/data" \
      --feature-view latent --history full --coverage full \
      --ablation "$ABLATION" --seed "$SEED" --out-dir "$OUT"
  done
done
