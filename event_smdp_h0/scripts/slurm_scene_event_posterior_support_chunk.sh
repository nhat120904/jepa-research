#!/usr/bin/env bash
#SBATCH --job-name=scene_event_posterior_support_chunk
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_event_posterior_support_chunk_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
START_INDEX=${START_INDEX:?submit with START_INDEX}
END_INDEX=${END_INDEX:?submit with END_INDEX}
RUN_ID=${RUN_ID:-support_20260904}
if (( START_INDEX < 0 || END_INDEX > 63 || START_INDEX > END_INDEX )); then
  echo "invalid locked index interval ${START_INDEX}--${END_INDEX}" >&2
  exit 2
fi
export STABLEWM_HOME="$RUNTIME"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} RANGE=${START_INDEX}-${END_INDEX} RUN=$RUN_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scene_event_perception.py" \
  "$PROJECT/scripts/replay_scene_event_posterior_support.py" \
  "$PROJECT/scripts/slurm_scene_event_posterior_support_chunk.sh"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for (( INDEX=START_INDEX; INDEX<=END_INDEX; INDEX++ )); do
  RESET_SEED=$((84500 + INDEX))
  SOURCE="$PROJECT/outputs/scene_event_perception_replication/eval/confirmatory_20260904/$((64 + INDEX))/result.json"
  OUT="$PROJECT/outputs/scene_event_posterior_support/replay/${RUN_ID}/${INDEX}"
  echo "INDEX=$INDEX SEED=$RESET_SEED START=$(date -u +%FT%TZ)"
  "$RUNTIME/.venv/bin/python" \
    "$PROJECT/scripts/replay_scene_event_posterior_support.py" \
    --reset-seed "$RESET_SEED" --observer-seed 1 --source-result "$SOURCE" \
    --observer-checkpoint \
      "$PROJECT/outputs/scene_event_perception/checkpoints/latent/seed1/observer.pt" \
    --out-dir "$OUT"
done
