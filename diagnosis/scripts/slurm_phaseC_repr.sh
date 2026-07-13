#!/usr/bin/env bash
#SBATCH --job-name=jepa_p4c
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseC_repr_%j.out
# Track-2 (docs/plans/2026-07-01-adversarial-cost-and-repr-design.md): train the
# action-conditioned representation adapter phi = A(z) (scripts/37, four losses —
# grounding + cf-contrastive margin + temporal ranking + adversarial margin against
# mined CEM-exploited elites), then gate it exactly like every other Phase-0/3 cost
# rung: scripts/30 --cost phi under PERFECT latent dynamics.
#   push/pick 0 -> >0 -> the representation reshape works -> carry phi into the
#       scripts/18 closed loop (a later step; phi-space rollout, not built yet).
#   push/pick stays ~0-2/16 -> the pockets are inherent to what a frozen-encoder
#       *any* readout (linear, transformer, adversarially-hardened) can express ->
#       escalate to fine-tuning the encoder itself (out of scope here).
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
REPR=checkpoints/action_repr_adapter_${M}.pt

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
set -e

echo "### Phase C — train action_repr_adapter (phi), mining adversarial negatives live ###"
$PY scripts/37_train_repr_adapter.py --config "$CFG" --model "$M" \
    --mine-adv --adv-cost l2 --adv-episodes 4 --adv-tasks mw-push mw-pick-place \
    --out "$REPR"

echo "### Phase C — gate: scripts/30 --cost phi under perfect dynamics ###"
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost phi --repr-adapter "$REPR" \
    --tasks mw-push mw-pick-place --episodes "$EPISODES" --strict-success \
    --out results/latent_oracle_phi.csv

echo "===== PHASEC_REPR_DONE ====="; date
echo "--- phi gate (cf. baseline l2 push 0/16, robust-probe stateprobe 1/16, state-oracle 16/16) ---"
cat results/latent_oracle_phi.csv
