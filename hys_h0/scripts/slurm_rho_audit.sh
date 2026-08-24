#!/usr/bin/env bash
#SBATCH --job-name=hys_rho
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/mnt/data/nhatnc129/contactworld/logs/hys_rho_%A_%a.out
#
# THE gate. Not success counts -- rho_init / rho_final on the CEM population.
# Baseline (results/cem_preselection_dino_{push,pick}_l2_*, already on disk):
#   rho_init 0.25 (push) / 0.02 (pick); rho_final CI-clean NEGATIVE in all cells
#   (diagnosis/docs/CURRENT_STATUS.md:71).
# Config below is copied from scripts/slurm_cem_preselection_audit.sh so the new
# arms are directly comparable to those existing l2 dumps.
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
CFG=configs/diagnostic_metaworld.yaml
MODEL=dino_wm_metaworld
EPISODES=${PRESEL_EPISODES:-16}
SEED0=${PRESEL_SEED0:-41000}
S=../hys_h0

TASKS=(mw-push mw-push mw-push mw-pick-place mw-pick-place mw-pick-place)
GATES=(none switch off none switch off)
TAGS=(dino_push_straight_none dino_push_straight_switch dino_push_straight_off
      dino_pick_straight_none dino_pick_straight_switch dino_pick_straight_off)

I=${SLURM_ARRAY_TASK_ID:?submit as the declared array}
TASK=${TASKS[$I]}; GATE=${GATES[$I]}; TAG=${TAGS[$I]}
PROJ=$S/outputs/projector_${GATE}_seed0.pt
PREFIX=results/cem_preselection_${TAG}

echo "HOST=$(hostname) task=$TASK gate=$GATE episodes=$EPISODES seed0=$SEED0"
"$PY" scripts/51_oracle_coverage_selection.py \
  --config "$CFG" --model "$MODEL" --tasks "$TASK" \
  --cost straight --projector "$PROJ" \
  --episodes "$EPISODES" --seed0 "$SEED0" --strict-success --dump-candidates \
  --out-prefix "$PREFIX"
echo "HYS_RHO_CELL_DONE tag=$TAG $(date)"
