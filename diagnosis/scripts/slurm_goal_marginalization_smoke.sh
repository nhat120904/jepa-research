#!/usr/bin/env bash
#SBATCH --job-name=jepa_goalmarg_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/goal_marg_smoke_%j.out
# Gate 2 smoke test (docs/plans/2026-08-11-goal-marginalization-design.md).
# 2 episodes, mw-push, all three arms -- checks the perturbation/encode/CEM
# pipeline and the clean-intervention diagnostic before the locked 16-episode
# array.
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

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
sha256sum scripts/82_goal_marginalization_pilot.py

$PY scripts/82_goal_marginalization_pilot.py --config configs/diagnostic_metaworld.yaml \
    --model dino_wm_metaworld --tasks mw-push --episodes 2 --seed0 90000 \
    --strict-success --out results/goal_marg_smoke.csv
echo "===== GOAL_MARG_SMOKE_DONE ====="; date
cat results/goal_marg_smoke.csv
