#!/usr/bin/env bash
#SBATCH --job-name=jepa_invtr
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/inverse_train_%j.out
# Lever #2 (WAV-style): train the inverse action-proposal h_inv(z_t, Δobj) → raw
# action on the cached expert transitions (scripts/28). Cache-only — the forward
# world model is never called, so no MUJOCO_GL. Read the GATE at the end:
#   * val action-MSE should be well BELOW the constant-mean baseline (it learned an
#     inverse, not the mean action);
#   * contact_action_spread > 0 means the inverse proposes a DIFFERENT action when
#     object motion is requested — the contact-creating signal CEM lacks.
# Only if both hold is the closed-loop spend (slurm_inverse_closed_loop.sh) worth it.
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
M=${INV_MODEL:-dino_wm_metaworld}
EPOCHS=${INV_EPOCHS:-20}; CW=${INV_CONTACT_WEIGHT:-4}; MT=${INV_MOVE_THRESH:-0.005}
CKPT=checkpoints/inverse_proposal_${M}.pt

echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

set -e
echo "===== train inverse proposal (scripts/28) ====="; date
$PY scripts/28_train_inverse_proposal.py --config $CFG --model $M \
    --epochs $EPOCHS --contact-weight $CW --move-thresh $MT
ls -la $CKPT
echo "===== INVERSE_TRAIN_DONE ====="; date
