#!/usr/bin/env bash
#SBATCH --job-name=jepa_infoUB
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/encoder_info_upperbound_%j.out
# Off-policy pass for the encoder information upper-bound (scripts/48).
# The on-policy probes are trained on CPU (no GPU needed); this job loads that
# probe bundle and re-evaluates the single-frame readouts (gripper aperture,
# hand-object relative vector, object) on RANDOM-action rollout latents — the
# off-policy distribution the CEM planner actually scores. Reports the on/off gap.
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env 2>/dev/null || true; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
M=${INFOUB_MODEL:-dino_wm_metaworld}

$PY scripts/48_encoder_info_upperbound.py \
    --config "$CFG" --model "$M" \
    --offpolicy-only \
    --probes-ckpt "checkpoints/encoder_info_upperbound_${M}.pt" \
    --op-episodes 12 --op-max-steps 60 --op-collect-every 4 \
    --op-tasks mw-push mw-pick-place mw-reach
