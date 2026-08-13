#!/usr/bin/env bash
#SBATCH --job-name=jepa_snr_smoke
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/task_snr_smoke_%j.out
# Gate 1 smoke test (docs/plans/2026-08-11-goal-marginalization-design.md).
# Tiny run (2 episodes x 2 tasks) to check the encode/regime-classify pipeline
# before the full 9-task x 16-episode pilot.
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
sha256sum scripts/80_task_snr_pilot.py

$PY scripts/80_task_snr_pilot.py --config configs/diagnostic_metaworld.yaml \
    --model dino_wm_metaworld --tasks mw-push mw-drawer-close --episodes 2 \
    --seed0 80000 --out results/task_snr_smoke.csv
echo "===== TASK_SNR_SMOKE_DONE ====="; date
cat results/task_snr_smoke.csv
