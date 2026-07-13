#!/usr/bin/env bash
#SBATCH --job-name=jepa_confirm
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%2
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/confirmatory_%A_%a.out
#
# Locked confirmatory evaluation on seeds never used by the n=8/16 development
# ladder. One array cell runs exactly one task x arm, so partial failures can be
# resubmitted without invalidating completed cells.
#
# Submit only after the current replication/probe jobs finish, for example:
#   sbatch --dependency=afterany:26481:26482:26485 \
#     scripts/slurm_confirmatory_locked.sh
set -euo pipefail

cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"

PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
EPISODES=${CONFIRM_EPISODES:-64}
SEED0=${CONFIRM_SEED0:-20000}
IDX=${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}

TASKS=(mw-push mw-pick-place mw-push mw-pick-place mw-push mw-pick-place mw-push mw-pick-place mw-push mw-pick-place)
KINDS=(oracle oracle l2 l2 stateprobe stateprobe l2 l2 stateprobe stateprobe)
MODELS=(none none dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld dino_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld jepa_wm_metaworld)

TASK=${TASKS[$IDX]}
KIND=${KINDS[$IDX]}
MODEL=${MODELS[$IDX]}
TAG=${MODEL}_${KIND}_${TASK}_seed${SEED0}_n${EPISODES}
OUT=results/confirmatory_${TAG}.csv

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA}_${IDX} $(date)"
echo "locked confirmatory: task=$TASK kind=$KIND model=$MODEL seed0=$SEED0 episodes=$EPISODES out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [ "$KIND" = oracle ]; then
  "$PY" scripts/29_oracle_ceiling.py --config "$CFG" \
    --tasks "$TASK" --episodes "$EPISODES" --seed0 "$SEED0" \
    --strict-success --out "$OUT"
  exit 0
fi

EXTRA=()
if [ "$KIND" = stateprobe ]; then
  OBJ=checkpoints/spatial_object_probe_${MODEL}_offpolicy.pt
  EEP=checkpoints/ee_probe_${MODEL}_offpolicy.pt
  test -f "$OBJ"
  test -f "$EEP"
  EXTRA+=(--probe "$OBJ" --ee-probe "$EEP")
fi

"$PY" scripts/30_latent_oracle.py --config "$CFG" --model "$MODEL" \
  --cost "$KIND" "${EXTRA[@]}" \
  --tasks "$TASK" --episodes "$EPISODES" --seed0 "$SEED0" \
  --strict-success --out "$OUT"

echo "===== CONFIRMATORY_CELL_DONE ===== $(date)"
