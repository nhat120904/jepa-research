#!/usr/bin/env bash
#SBATCH --job-name=jepa_phil2
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phil2_gate_%j.out
# Phase D follow-up — HYBRID cost gate (no training; re-gates the round-0 Phase D
# checkpoints with the new `phil2` cost arm = γ·L2 + β·φ_obj).
#
# Motivation (Phase D job 24128, perfect-dynamics gate):
#   φ+encLoRA : push 5/16  pick 1/16  reach 2/16   (φ's object-centric cost is
#               degenerate on the no-object reach task -> reach regresses)
#   L2+encLoRA: push 0/8   —          reach 16/16  (raw L2 solves reach, misses push)
# phil2 is ONE general cost, no task branching: the L2 backbone drives the hand
# (reach), the β·φ_obj term injects the grounded object signal (push), and on reach
# φ_obj≈const so it should not break the L2 solution. GOAL: reach≈16/16 AND push>0
# in a SINGLE cost — the generality φ alone lost.
#
# β sweeps the L2-vs-object balance. β too small -> pure L2 (push 0); β too big ->
# pure φ_obj (reach breaks like the φ arm). Submit a few (P4_BETA=0.02,0.1,0.5).
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
EPISODES=${P4_EPISODES:-16}
BETA=${P4_BETA:-0.1}
GAMMA=${P4_GAMMA:-1.0}
LORA=checkpoints/encoder_lora_${M}.pt
PHI=checkpoints/phi_enclora_${M}.pt
TAG=$(echo "$BETA" | tr '.' 'p')
OUT=results/latent_oracle_phil2_b${TAG}.csv

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date) beta=$BETA gamma=$GAMMA"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
if [ ! -f "$LORA" ] || [ ! -f "$PHI" ]; then
  echo "MISSING round-0 checkpoints ($LORA / $PHI) — run Phase D first. STOP." >&2
  exit 1
fi
set -e

echo "### phil2 gate: γ·L2 + β·φ_obj on round-0 encoder-LoRA (reach/push/pick) ###"
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost phil2 --repr-adapter "$PHI" --encoder-lora "$LORA" \
    --beta "$BETA" --gamma-l2 "$GAMMA" \
    --tasks mw-reach mw-push mw-pick-place --episodes "$EPISODES" --strict-success \
    --out "$OUT"

echo "===== PHIL2_GATE_DONE (beta=$BETA) ====="; date
echo "--- phil2 gate (cf. φ 5/16 push,2/16 reach; L2 0/8 push,16/16 reach; state-oracle 16/16) ---"
cat "$OUT"