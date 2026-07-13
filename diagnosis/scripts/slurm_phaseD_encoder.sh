#!/usr/bin/env bash
#SBATCH --job-name=jepa_p4d
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseD_encoder_%j.out
# Phase D — encoder-level LoRA (docs/plans/2026-07-02-encoder-lora-action-grounding-design.md).
#
# Phase 4 closed the frozen-encoder program EMPIRICALLY: Phase A (job 24020)
# confirmed CEM reward-hacking (robust probe 91.5% <5cm random -> 19-24% on
# CEM-mined elites); Phase C v2 (job 24018) relearned phi WITH healthy grounding
# (MSE 0.0043) and still gated push 1/16 pick 0/16 — identical to v1, object
# ~21cm from goal. Any phi(z) is a function of the frozen z: merged latents stay
# merged. This job moves the same losses INSIDE the encoder (zero-init LoRA on
# the DINOv2 blocks) so the merged latents can be separated at the source.
#
# Read-out (vs the frozen ladder: l2 0/16, stateprobe 2/16, phi v1/v2 1/16;
# ceiling: state-oracle 16/16 push, 11/16 pick):
#   push/pick > 2/16 with reach >= 13/16  -> the geometry lever works; carry to
#       the closed loop (predictor-compat step decided then, scripts/26).
#   still 0-2/16 -> even a LoRA-rank encoder reshape is insufficient; the honest
#       escalation is full encoder finetuning/retraining (different paper scale),
#       and the negative-result chain gains its strongest rung.
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
EPISODES=${P4D_EPISODES:-16}
R=${P4D_LORA_R:-16}
LP=${P4D_LAMBDA_PRESERVE:-0.05}
LORA=checkpoints/encoder_lora_${M}.pt
PHI=checkpoints/phi_enclora_${M}.pt

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
set -e

echo "### Phase D — sanity: LoRA injected-but-DISABLED must reproduce the baseline ###"
# zero-init identity is locked offline by tests/test_lora_encoder.py; this is the
# on-checkpoint equivalent (check_normalization goes through adapter.encode).
$PY - <<EOF
import torch
from models.adapters import build_adapter
from models.heads.lora_encoder import inject_encoder_lora
from models.heads.lora_predictor import set_lora_enabled
dev = "cuda" if torch.cuda.is_available() else "cpu"
ad = build_adapter("$M", device=dev).eval()
vis = torch.rand(2, 1, 3, 224, 224) * 255
prop = torch.zeros(2, 1, 4)
z0 = ad.encode(vis, prop)
inj = inject_encoder_lora(ad, r=$R)
z1 = ad.encode(vis, prop)              # zero-init -> must be identical
set_lora_enabled(inj, False)
z2 = ad.encode(vis, prop)
assert torch.allclose(z0, z1, atol=1e-5) and torch.allclose(z0, z2, atol=1e-5), \
    "LoRA-injected encoder is NOT identity at init — STOP"
print(f"identity check OK ({len(inj)} modules injected)")
EOF

echo "### Phase D — train encoder LoRA + phi (terms 1-3 + preservation, v0: no adv) ###"
$PY scripts/38_train_encoder_lora.py --config "$CFG" --model "$M" \
    --lora-r "$R" --lambda-preserve "$LP" --offpolicy-frac 0.5 \
    --out-lora "$LORA" --out-phi "$PHI"

echo "### Phase D — gate: scripts/30 --cost phi --encoder-lora (incl. mw-reach generality) ###"
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost phi --repr-adapter "$PHI" --encoder-lora "$LORA" \
    --tasks mw-reach mw-push mw-pick-place --episodes "$EPISODES" --strict-success \
    --out results/latent_oracle_phi_enclora.csv

echo "### Phase D — variant: is plain L2 plannable in the reshaped space? ###"
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost l2 --encoder-lora "$LORA" \
    --tasks mw-reach mw-push mw-pick-place --episodes "$EPISODES" --strict-success \
    --out results/latent_oracle_l2_enclora.csv

echo "===== PHASED_ENCODER_DONE ====="; date
echo "--- phi+enclora gate (cf. frozen phi v2 1/16, l2 0/16, state-oracle 16/16) ---"
cat results/latent_oracle_phi_enclora.csv
echo "--- l2+enclora gate ---"
cat results/latent_oracle_l2_enclora.csv
