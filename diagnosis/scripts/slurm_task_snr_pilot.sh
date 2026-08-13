#!/usr/bin/env bash
#SBATCH --job-name=jepa_snr_pilot
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/task_snr_pilot_%j.out
# Gate 1 full pilot (docs/plans/2026-08-11-goal-marginalization-design.md).
# 9 tasks x 16 episodes; only 3 frames encoded per episode (cheap), but the
# expert rollout itself renders every step, so budget real wall time.
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

TASKS="mw-push mw-pick-place mw-reach mw-button-press mw-drawer-close mw-window-close mw-faucet-open mw-plate-slide mw-soccer"
OUT=${SNR_OUT:-results/task_snr_pilot.csv}
EPISODES=${SNR_EPISODES:-16}
SEED0=${SNR_SEED0:-80000}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
sha256sum scripts/80_task_snr_pilot.py
echo "tasks=[$TASKS] episodes=$EPISODES seed0=$SEED0 out=$OUT"

$PY scripts/80_task_snr_pilot.py --config configs/diagnostic_metaworld.yaml \
    --model dino_wm_metaworld --tasks $TASKS --episodes "$EPISODES" \
    --seed0 "$SEED0" --out "$OUT"
echo "===== TASK_SNR_PILOT_DONE ====="; date
