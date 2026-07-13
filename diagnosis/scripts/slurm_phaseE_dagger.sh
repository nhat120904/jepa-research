#!/usr/bin/env bash
#SBATCH --job-name=jepa_p4e
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseE_dagger_%j.out
# Phase E — encoder-level adversarial DAgger, round 1 (follows Phase D job 24128).
#
# Phase D v0 (no adversarial term) was the FIRST crossing of the contact wall under
# perfect dynamics: φ+encoder-LoRA push 5/16 (every frozen lever ≤2/16). But 11 push
# episodes still fail with the object 9-30cm from goal — the encoder reshape closed
# SOME residual-error pockets, not all. This round mines exactly those surviving
# pockets (CEM elites the round-0 φ+encLoRA cost is exploited on) and retrains with
# the adversarial term (term 4, OFF in v0) pushing them apart — encoder-level DAgger.
#
# Read-out (vs Phase D round-0 φ+encLoRA: push 5/16, pick 1/16, reach 2/16):
#   push climbs toward 16/16  -> DAgger closes the pockets; iterate / carry to closed loop.
#   push plateaus ~5/16       -> the surviving pockets are not fixable by re-supervising
#       against them at this LoRA rank; escalate (higher rank / more blocks / hybrid cost).
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
EPISODES=${P4E_EPISODES:-16}
MINE_EPISODES=${P4E_MINE_EPISODES:-8}    # 8×2 tasks ≈ 6-7k elites; keep runtime sane
R=${P4E_LORA_R:-16}
LP=${P4E_LAMBDA_PRESERVE:-0.05}
LADV=${P4E_LAMBDA_ADV:-1.0}
# round-0 (Phase D) inputs
LORA0=checkpoints/encoder_lora_${M}.pt
PHI0=checkpoints/phi_enclora_${M}.pt
# round-1 outputs
BUF=/mnt/data/nhatnc129/jepa_runs/cem_exploit_buffer_phi_enclora_r0.pt
LORA1=checkpoints/encoder_lora_${M}_r1.pt
PHI1=checkpoints/phi_enclora_${M}_r1.pt

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
if [ ! -f "$LORA0" ] || [ ! -f "$PHI0" ]; then
  echo "MISSING round-0 checkpoints ($LORA0 / $PHI0) — run Phase D first. STOP." >&2
  exit 1
fi
set -e

echo "### Phase E — STAGE 1: mine round-0 φ+encLoRA exploited elites (with frames) ###"
$PY scripts/35_cem_exploit_precision.py --config "$CFG" --model "$M" \
    --cost phi --repr-adapter "$PHI0" --encoder-lora "$LORA0" \
    --tasks mw-push mw-pick-place --episodes "$MINE_EPISODES" --strict-success \
    --keep-frames --save-buffer "$BUF" \
    --out results/cem_exploit_precision_phi_enclora_r0.csv

echo "### Phase E — STAGE 2: retrain encoder LoRA + phi WITH adversarial term ###"
$PY scripts/38_train_encoder_lora.py --config "$CFG" --model "$M" \
    --lora-r "$R" --lambda-preserve "$LP" --offpolicy-frac 0.5 \
    --adv-buffer "$BUF" --lambda-adv "$LADV" \
    --out-lora "$LORA1" --out-phi "$PHI1"

echo "### Phase E — STAGE 3: gate round-1 (scripts/30 --cost phi, incl. mw-reach) ###"
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost phi --repr-adapter "$PHI1" --encoder-lora "$LORA1" \
    --tasks mw-reach mw-push mw-pick-place --episodes "$EPISODES" --strict-success \
    --out results/latent_oracle_phi_enclora_r1.csv

echo "===== PHASEE_DAGGER_DONE ====="; date
echo "--- round-1 gate (cf. round-0 phi+encLoRA push 5/16 pick 1/16 reach 2/16; state-oracle 16/16) ---"
cat results/latent_oracle_phi_enclora_r1.csv