#!/usr/bin/env bash
#SBATCH --job-name=jepa_p4f
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseF_seed%a_%j.out
# Phase F — training-seed sweep of the encoder-LoRA + φ recipe (Phase D core, no adv).
#
# WHY: round-0 (Phase D, seed 0) gave push 5/16; a near-identical retrain (Phase E,
# seed 0 + RNG shift, adv≈0) gave push 0/16 with BETTER grounding. So the push metric
# has enormous training-seed variance and 5/16 sits within the frozen baseline noise
# (0–2/16). This sweep runs the CLEAN recipe across independent training seeds and
# gates each on the SAME fixed 16 push episodes (seed0=10000), to estimate push
# mean±spread vs frozen. Only then can any "crossing" be claimed (or refuted).
#
# Submit as an array: sbatch --array=1-3 scripts/slurm_phaseF_seedsweep.sh
# (seed 0 already characterised: round-0=5/16, round-1(rng-shift)=0/16.)
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
SEED=${SLURM_ARRAY_TASK_ID:-${P4F_SEED:-1}}
EPISODES=${P4F_EPISODES:-16}
R=${P4F_LORA_R:-16}
LP=${P4F_LAMBDA_PRESERVE:-0.05}
LORA=checkpoints/encoder_lora_${M}_s${SEED}.pt
PHI=checkpoints/phi_enclora_${M}_s${SEED}.pt

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date) SEED=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
set -e

echo "### Phase F seed=$SEED — train encoder LoRA + phi (clean recipe, no adv) ###"
$PY scripts/38_train_encoder_lora.py --config "$CFG" --model "$M" \
    --seed "$SEED" --lora-r "$R" --lambda-preserve "$LP" --offpolicy-frac 0.5 \
    --out-lora "$LORA" --out-phi "$PHI"

echo "### Phase F seed=$SEED — gate PUSH only (fixed episodes, cf. seed0 round-0 5/16) ###"
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
    --cost phi --repr-adapter "$PHI" --encoder-lora "$LORA" \
    --tasks mw-push --episodes "$EPISODES" --strict-success \
    --out results/latent_oracle_phi_enclora_s${SEED}.csv

echo "===== PHASEF_SEED${SEED}_DONE ====="; date
awk -F, 'NR>1{a++; s+=$4; e+=$5} END{printf "seed='$SEED' push any=%d/%d held=%d/%d\n",s,a,e,a}' \
    results/latent_oracle_phi_enclora_s${SEED}.csv