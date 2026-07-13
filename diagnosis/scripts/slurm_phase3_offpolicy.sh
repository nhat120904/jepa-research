#!/usr/bin/env bash
#SBATCH --job-name=jepa_p3fix
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phase3_offpolicy_%j.out
# Phase-3 3b (the fix + re-gate). Run AFTER slurm_phase3_diag.sh confirms the readout
# collapses off-policy. (1) retrain the spatial object probe AND the ee probe with
# off-policy random-action frames mixed in (--offpolicy-frac); (2) re-measure 3a on the
# robust object probe; (3) re-run the stateprobe gate (perfect dynamics, exact
# state-oracle cost) with the ROBUST probes.
#   push >> 2/16 (toward state-oracle 16/16) -> fix = off-policy-robust readout (frozen
#                                               encoder, cheap) -> carry into closed-loop.
#   push ~ 2/16                               -> latent loses object info off-policy ->
#                                               escalate to an encoder-level objective.
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
CFG=configs/diagnostic_metaworld.yaml
M=${P3_MODEL:-dino_wm_metaworld}
FRAC=${P3_FRAC:-0.5}
EPISODES=${P3_EPISODES:-16}
OBJ=checkpoints/spatial_object_probe_${M}_offpolicy.pt
EEP=checkpoints/ee_probe_${M}_offpolicy.pt

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)  frac=$FRAC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

set -e
echo "### 3b-1 train off-policy-robust spatial object probe ###"
$PY scripts/22_train_spatial_probe.py --config "$CFG" --model "$M" \
    --offpolicy-frac "$FRAC" --out results/representation_precision_spatial_offpolicy.csv
echo "### 3b-2 train off-policy-robust ee probe ###"
$PY scripts/19_train_ee_probe.py --config "$CFG" --model "$M" --offpolicy-frac "$FRAC"
echo "### 3a re-check on the ROBUST object probe ###"
$PY scripts/34_offpolicy_precision.py --config "$CFG" --model "$M" \
    --probe "$OBJ" --target obj --episodes "$EPISODES" \
    --out results/offpolicy_precision_obj_robust.csv
echo "### re-gate: stateprobe with the ROBUST probes ###"
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost stateprobe --probe "$OBJ" --ee-probe "$EEP" \
    --tasks mw-push mw-pick-place --episodes "$EPISODES" --strict-success \
    --out results/latent_oracle_stateprobe_robust.csv
echo "===== PHASE3_OFFPOLICY_DONE ====="; date
echo "--- robust stateprobe (cf. baseline stateprobe push 2/16, pick 0/16) ---"
cat results/latent_oracle_stateprobe_robust.csv
