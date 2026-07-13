#!/usr/bin/env bash
#SBATCH --job-name=jepa_infoUB
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/jepa_infoUB_%j.out
# Encoder information upper-bound (scripts/48) — the Reviewer-1 probe study:
# does the frozen dino_wm_metaworld latent CARRY the quantities the winning
# state-oracle cost consumed (gripper aperture, hand-object relative vector,
# the full c*(s) value from [z_t, z_g], contact indicator), or does it lack
# them? On-policy probes train on the cached latents; the off-policy pass
# re-evaluates the single-frame probes on random-action rollout latents (the
# distribution the CEM planner actually scores) — hence the GPU (encoder +
# EGL render). Outputs: results/encoder_info_upperbound.{csv,md} + a probe
# bundle in checkpoints/.
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"
export CAI_JEPA_TORCH_THREADS=${CAI_JEPA_TORCH_THREADS:-10}
PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
M=${INFOUB_MODEL:-dino_wm_metaworld}
EPOCHS=${INFOUB_EPOCHS:-12}

$PY scripts/48_encoder_info_upperbound.py \
    --config "$CFG" --model "$M" --epochs "$EPOCHS" \
    --offpolicy \
    --op-episodes 12 --op-max-steps 60 --op-collect-every 4 \
    --op-tasks mw-push mw-pick-place mw-reach
