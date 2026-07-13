#!/usr/bin/env bash
#SBATCH --job-name=jepa_h_cfho
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseH_cf_heldout_%A_%a.out
# Phase-H corrective rerun: strict unseen-trajectory Action-Score evaluation.
#
# Every training seed uses the same immutable trajectory manifest (split seed 0):
# 70% train / 15% validation / 15% test.  scripts/40 never materializes test
# tensors.  scripts/08 filters both planning anchors and the hard-negative pool to
# the reserved test trajectories.  Seed 0 also produces the single deterministic
# frozen baseline on that exact test split; all seeds produce a LoRA result.
#
# Submit both model arrays (do not execute the Python commands on a login node):
#   P4H_MODEL=dino_wm_droid sbatch --array=0-3 scripts/slurm_phaseH_cf_heldout.sh
#   P4H_MODEL=jepa_wm_droid sbatch --array=0-3 scripts/slurm_phaseH_cf_heldout.sh
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"

PY=.venv/bin/python
CFG=configs/diagnostic_droid.yaml
M=${P4H_MODEL:-dino_wm_droid}
SEED=${SLURM_ARRAY_TASK_ID:-0}
SPLIT_SEED=${P4H_SPLIT_SEED:-0}
NS=${P4H_CEM_SAMPLES:-64}
IT=${P4H_CEM_ITERS:-15}
MP=${P4H_MAX_PLAN:-40}

case "$M" in
  dino_wm_droid)
    EPOCHS=${P4H_EPOCHS:-8}; RANK=${P4H_RANK:-8}; ALPHA=${P4H_ALPHA:-16}
    ;;
  jepa_wm_droid)
    EPOCHS=${P4H_EPOCHS:-16}; RANK=${P4H_RANK:-16}; ALPHA=${P4H_ALPHA:-32}
    ;;
  *)
    echo "Unsupported P4H_MODEL=$M (expected dino_wm_droid or jepa_wm_droid)" >&2
    exit 2
    ;;
esac

MANIFEST=checkpoints/splits/phaseH_${M}_split${SPLIT_SEED}.json
CK=checkpoints/predictor_cf_${M}_heldout_s${SEED}.pt
LORA_CSV=results/droid_planning_cf_${M}_heldout_s${SEED}.csv
FROZEN_CSV=results/droid_planning_cf_${M}_heldout_frozen.csv

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
echo "model=$M train_seed=$SEED split_seed=$SPLIT_SEED rank=$RANK epochs=$EPOCHS"
echo "manifest=$MANIFEST eval_split=test"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
set -e

echo "### (1/3) train on train split; select checkpoint on validation split ###"
$PY scripts/40_train_predictor_cf.py --config "$CFG" --model "$M" \
    --seed "$SEED" --split-seed "$SPLIT_SEED" \
    --val-frac 0.15 --test-frac 0.15 --split-manifest "$MANIFEST" \
    --epochs "$EPOCHS" --rank "$RANK" --alpha "$ALPHA" \
    --lambda-cf 1.0 --num-neg 4 --out "$CK"

if [ "$SEED" -eq 0 ]; then
  echo "### (2/3) frozen predictor on reserved test trajectories ###"
  $PY scripts/08_planning_probe.py --config "$CFG" --only-model "$M" \
      --eval-split test --split-manifest "$MANIFEST" \
      --cem-num-samples "$NS" --cem-iterations "$IT" \
      --max-planning-transitions "$MP" --out-csv "$FROZEN_CSV"
else
  echo "### (2/3) frozen baseline skipped (deterministic; seed-0 array task owns it) ###"
fi

echo "### (3/3) validation-selected CF-LoRA predictor on reserved test trajectories ###"
$PY scripts/08_planning_probe.py --config "$CFG" --only-model "$M" \
    --eval-split test --split-manifest "$MANIFEST" \
    --cem-num-samples "$NS" --cem-iterations "$IT" \
    --max-planning-transitions "$MP" --predictor-lora "$CK" \
    --out-csv "$LORA_CSV"

echo "===== PHASEH_CF_HELDOUT_${M}_s${SEED}_DONE ====="
date
