#!/usr/bin/env bash
#SBATCH --job-name=jepa_p4c2
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseC_repr_v2_%j.out
# Track-2 RETRAIN (v2). The v1 run (job 23900) trained phi with the default
# lambda_temporal=lambda_cross=0.3 and the temporal-ranking hinge dominated the
# shared transformer trunk — weighted temporal contribution ~82% of total loss at
# epoch 8 — starving the grounding objective (held-out grounding decode collapsed
# to 7.1% <5cm vs the 92% a grounding-ONLY probe of the same architecture reaches,
# Test-1b). The phi gate then re-gated push 1/16 (== the frozen stateprobe wall),
# i.e. phi never got a fair shot: its object estimate was too degraded to plan on.
#
# This v2 cuts lambda_temporal and lambda_cross 10x (0.3 -> 0.03) so the grounding
# gradient dominates the trunk; the temporal terms still shape phi_extra (which the
# --cost phi arm down-weights by beta anyway) but can no longer reshape the trunk
# away from the object signal. Distinct --out / --out csv so the v1 checkpoint and
# results/latent_oracle_phi.csv are preserved for the before/after comparison.
#   push/pick 0 -> >0 -> the representation reshape works once grounding is intact.
#   push/pick stays ~0-2/16 EVEN WITH grounding restored -> the pockets are inherent
#       to what ANY frozen-encoder readout can express -> escalate to encoder
#       fine-tuning (out of scope here).
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
M=${P4_MODEL:-dino_wm_metaworld}
EPISODES=${P4C_EPISODES:-16}
LT=${P4C_LAMBDA_TEMPORAL:-0.03}
LX=${P4C_LAMBDA_CROSS:-0.03}
REPR=checkpoints/action_repr_adapter_${M}_v2.pt

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
set -e

echo "### Phase C v2 — train action_repr_adapter (phi), lambda_temporal=$LT lambda_cross=$LX ###"
$PY scripts/37_train_repr_adapter.py --config "$CFG" --model "$M" \
    --lambda-temporal "$LT" --lambda-cross "$LX" \
    --mine-adv --adv-cost l2 --adv-episodes 4 --adv-tasks mw-push mw-pick-place \
    --out "$REPR"

echo "### Phase C v2 — gate: scripts/30 --cost phi under perfect dynamics ###"
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost phi --repr-adapter "$REPR" \
    --tasks mw-push mw-pick-place --episodes "$EPISODES" --strict-success \
    --out results/latent_oracle_phi_v2.csv

echo "===== PHASEC_REPR_V2_DONE ====="; date
echo "--- phi v2 gate (cf. v1 phi push 1/16, baseline l2 0/16, robust-probe stateprobe 1/16, state-oracle 16/16) ---"
cat results/latent_oracle_phi_v2.csv
