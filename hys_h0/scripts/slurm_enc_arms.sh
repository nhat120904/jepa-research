#!/usr/bin/env bash
#SBATCH --job-name=hys_enc
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%2
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_enc_%A_%a.out
#
# HyS-JEPA full form: encoder-LoRA + predictor, 4 arms x 3 seeds, mw-push pilot.
#   off    = prediction-only (does straightening add anything?)
#   none   = global straightening
#   switch = mode-gated (the proposal)
#   random = matched random drop -- NON-NEGOTIABLE control; it beat `switch` in the
#            frozen form, so without it a `switch > none` result proves nothing.
# Pre-registered: primary comparison switch vs random; secondary none vs off.
# Stopping rule: if the between-arm gap is smaller than the between-seed spread of the
# same arm, declare null rather than reading signs.
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
S=../hys_h0
GATES=(off none switch random off none switch random off none switch random)
SEEDS=(0   0    0      0      1   1    1      1      2   2    2      2)
I=${SLURM_ARRAY_TASK_ID}; GATE=${GATES[$I]}; SEED=${SEEDS[$I]}
R=${LORA_R:-16}
echo "HOST=$(hostname) gate=$GATE seed=$SEED lora_r=$R $(date)"
.venv/bin/python $S/scripts/04_train_encoder_straightener.py \
  --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
  --tasks mw-push --gate $GATE --seed $SEED --lora-r $R \
  --max-trajs-per-task 60 --epochs 6 --batch-size 4 \
  --out-lora $S/outputs/enc_${GATE}_r${R}_seed${SEED}.pt
echo "ENC_CELL_DONE gate=$GATE seed=$SEED $(date)"
