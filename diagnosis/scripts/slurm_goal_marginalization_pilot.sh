#!/usr/bin/env bash
#SBATCH --job-name=jepa_goalmarg
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/goal_marg_pilot_%j.out
# Gate 2 locked pilot (docs/plans/2026-08-11-goal-marginalization-design.md).
# 16 episodes x 3 arms x full closed-loop CEM on mw-push, seeds 90000-90015.
# Do not sweep K/n-pert/pert-sigma on these seeds -- they are locked.
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

TASK=${GOALMARG_TASK:-mw-push}
EPISODES=${GOALMARG_EPISODES:-16}
SEED0=${GOALMARG_SEED0:-90000}
OUT=${GOALMARG_OUT:-results/goal_marginalization_mw-push_seed90000_n16.csv}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
sha256sum scripts/82_goal_marginalization_pilot.py
echo "task=$TASK episodes=$EPISODES seed0=$SEED0 out=$OUT"

$PY scripts/82_goal_marginalization_pilot.py --config configs/diagnostic_metaworld.yaml \
    --model dino_wm_metaworld --tasks "$TASK" --episodes "$EPISODES" --seed0 "$SEED0" \
    --strict-success --out "$OUT"
echo "===== GOAL_MARG_PILOT_DONE ====="; date
