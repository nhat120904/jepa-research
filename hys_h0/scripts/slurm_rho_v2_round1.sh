#!/usr/bin/env bash
#SBATCH --job-name=hys_r1
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_r1_%A_%a.out
# Round 1 of the trimmed plan: switch vs random only, 3 seeds.
# The single decisive question -- do the contact SEMANTICS matter, or is any matched
# drop of the high-curvature tail equivalent? If these two do not separate beyond seed
# noise, the gating does nothing and rounds 2+ are not worth the GPU time.
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
GATES=(switch random switch random switch random); SEEDS=(0 0 1 1 2 2)
I=${SLURM_ARRAY_TASK_ID}; GATE=${GATES[$I]}; SEED=${SEEDS[$I]}
TAG=dino_push_v2_${GATE}_s${SEED}
echo "HOST=$(hostname) gate=$GATE seed=$SEED $(date)"
.venv/bin/python scripts/51_oracle_coverage_selection.py \
  --config configs/diagnostic_metaworld.yaml --model dino_wm_metaworld \
  --tasks mw-push --cost straight --projector $S/outputs/v2_${GATE}_seed${SEED}.pt \
  --episodes 16 --seed0 41000 --strict-success --dump-candidates \
  --out-prefix results/cem_preselection_${TAG}
echo "HYS_R1_CELL_DONE tag=$TAG $(date)"
