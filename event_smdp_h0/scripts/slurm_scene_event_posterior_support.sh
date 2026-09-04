#!/usr/bin/env bash
#SBATCH --job-name=scene_event_posterior_support
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-63%8
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_event_posterior_support_%A_%a.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
INDEX=${SLURM_ARRAY_TASK_ID:?}
RESET_SEED=$((84500 + INDEX))
RUN_ID=${RUN_ID:-${SLURM_ARRAY_JOB_ID:?}}
SOURCE="$PROJECT/outputs/scene_event_perception_replication/eval/confirmatory_20260904/$((64 + INDEX))/result.json"
OUT="$PROJECT/outputs/scene_event_posterior_support/replay/${RUN_ID}/${INDEX}"
export STABLEWM_HOME="$RUNTIME"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} INDEX=$INDEX SEED=$RESET_SEED RUN=$RUN_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_event_perception.py" \
  "$PROJECT/scripts/replay_scene_event_posterior_support.py" \
  "$PROJECT/scripts/slurm_scene_event_posterior_support.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/replay_scene_event_posterior_support.py" \
  --reset-seed "$RESET_SEED" --observer-seed 1 --source-result "$SOURCE" \
  --observer-checkpoint "$PROJECT/outputs/scene_event_perception/checkpoints/latent/seed1/observer.pt" \
  --out-dir "$OUT"
