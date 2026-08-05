#!/usr/bin/env bash
#SBATCH --job-name=jepa_taskladder2
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-11%4
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/task_breadth_ladder2_%A_%a.out
#
# Second batch of the task-breadth extension
# (docs/plans/2026-08-04-generality-extension-design.md): exploratory, no
# single geometric hypothesis to test yet (lever-pull/faucet-open = rotation,
# plate-slide/box-close = translation without grasp like drawer-close,
# shelf-place = grasp+place control expected to fail like push/pick-place,
# soccer = free-standing object push without grasp). Only dino_wm_metaworld
# runs the l2 arm: dino_wm_metaworld and jepa_wm_metaworld share a frozen
# dinov2_vits14 encoder, and under exact dynamics the predictor is never
# called, so running both models would duplicate the same experiment.
# 6 tasks x {oracle, l2} = 12 cells.
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"

PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
EPISODES=${LADDER_EPISODES:-16}
SEED0=${LADDER_SEED0:-71000}
IDX=${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}

TASKS=(mw-lever-pull mw-lever-pull mw-faucet-open mw-faucet-open mw-plate-slide mw-plate-slide mw-box-close mw-box-close mw-shelf-place mw-shelf-place mw-soccer mw-soccer)
KINDS=(oracle l2 oracle l2 oracle l2 oracle l2 oracle l2 oracle l2)

TASK=${TASKS[$IDX]}
KIND=${KINDS[$IDX]}
TAG=${TASK}_${KIND}_seed${SEED0}_n${EPISODES}
OUT=results/task_breadth_ladder2_${TAG}.csv

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA}_${IDX} $(date)"
echo "task breadth ladder2: task=$TASK kind=$KIND seed0=$SEED0 episodes=$EPISODES out=$OUT"
echo "commit=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

if [ "$KIND" = oracle ]; then
  "$PY" scripts/29_oracle_ceiling.py --config "$CFG" \
    --tasks "$TASK" --episodes "$EPISODES" --seed0 "$SEED0" \
    --strict-success --out "$OUT"
else
  "$PY" scripts/30_latent_oracle.py --config "$CFG" --model dino_wm_metaworld \
    --cost l2 --tasks "$TASK" --episodes "$EPISODES" --seed0 "$SEED0" \
    --strict-success --out "$OUT"
fi

echo "===== TASK_BREADTH_LADDER2_CELL_DONE ===== $(date)"
