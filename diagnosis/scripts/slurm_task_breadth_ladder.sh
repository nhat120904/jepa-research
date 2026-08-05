#!/usr/bin/env bash
#SBATCH --job-name=jepa_taskladder
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-9%4
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/task_breadth_ladder_%A_%a.out
#
# Gate 2 (positive control, via the no-model oracle arm) + Stage 1 terminal-
# cost ladder for the task-breadth extension
# (docs/plans/2026-08-04-generality-extension-design.md). Only
# dino_wm_metaworld runs the l2 arm: dino_wm_metaworld and jepa_wm_metaworld
# share a frozen dinov2_vits14 encoder (see design doc), and under exact
# dynamics the predictor is never called, so running both models would
# duplicate the same experiment. 5 tasks x {oracle, l2} = 10 cells.
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
SEED0=${LADDER_SEED0:-70000}
IDX=${SLURM_ARRAY_TASK_ID:?submit this script as a Slurm array}

# 5 gate-1-eligible tasks x {oracle, l2}, peg-insert-side excluded
# (INELIGIBLE-BUDGET, results/task_breadth_expert_check.md).
TASKS=(mw-door-open mw-door-open mw-drawer-close mw-drawer-close mw-button-press mw-button-press mw-window-close mw-window-close mw-assembly mw-assembly)
KINDS=(oracle l2 oracle l2 oracle l2 oracle l2 oracle l2)

TASK=${TASKS[$IDX]}
KIND=${KINDS[$IDX]}
TAG=${TASK}_${KIND}_seed${SEED0}_n${EPISODES}
OUT=results/task_breadth_ladder_${TAG}.csv

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA}_${IDX} $(date)"
echo "task breadth ladder: task=$TASK kind=$KIND seed0=$SEED0 episodes=$EPISODES out=$OUT"
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

echo "===== TASK_BREADTH_LADDER_CELL_DONE ===== $(date)"
