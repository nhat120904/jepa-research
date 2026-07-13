#!/usr/bin/env bash
#SBATCH --job-name=jepa_cfpl_mw
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/cf_planning_mw_s%a_%j.out
# ======================================================================
# THE KILLER TEST for paper weakness W3: does the predictor-side CF fix
# (paper Sec.~cf / C6) actually buy CLOSED-LOOP PLANNING SUCCESS (k/16),
# or only the grounding proxies the paper itself discredits?
#
# C6 currently shows the CF-InfoNCE predictor-LoRA (scripts/40) lifts DROID
# Action-Error / Action-Score / CRA — but DROID has NO simulator, so it can
# never report push k/16. MetaWorld can. So we apply the SAME objective on
# dino_wm_metaworld and run the SAME upstream CEM closed loop that scores
# push 0/16 for the frozen baseline.
#
# Two paired arms in scripts/18 (identical env init + CEM noise per episode):
#   l2      — FROZEN predictor, L2 cost      (the paper's 0/16 push baseline)
#   l2lora  — CF-trained predictor-LoRA, SAME L2 cost  (predictor axis isolated)
# Any k/16 delta l2 -> l2lora is attributable to the predictor axis alone
# (cost, planner, budget all held fixed). Interpretation either way strengthens
# the paper:
#   * push crosses 2/16  -> C6 becomes a genuine PLANNING fix, not a proxy win.
#   * push stays <=2/16   -> direct closed-loop proof of "grounding necessary,
#                            not sufficient" (Sec.~e1), replacing an inference.
#
# Recipe matches the CANONICAL (non-overfit) DROID recipe: rank 8, 8 epochs,
# num-neg 4, lambda_cf 1.0 (the rank16/24ep variant OVERFIT RoboCasa — do NOT
# use it here). CEM config = scripts/18 defaults = upstream L2_cem parity
# (H=6, 300 samples, 15 iters, 3 stepped, 100 max steps), strict episode-end
# success (matches the oracle ladder).
#
# Single killer A/B:   sbatch scripts/slurm_cf_planning_metaworld.sh
# CI seed sweep (W1):  sbatch --array=0-4 scripts/slurm_cf_planning_metaworld.sh
#   (for array seeds >0, set CF_ARMS=l2lora to skip the seed-independent frozen
#    arm and save the expensive CEM re-run:  CF_ARMS=l2lora sbatch --array=1-4 ...)
# ======================================================================
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
M=${CF_MODEL:-dino_wm_metaworld}
SEED=${SLURM_ARRAY_TASK_ID:-${CF_SEED:-0}}

# --- CF training recipe (canonical DROID rank8/8ep; do not switch to the overfit one) ---
EPOCHS=${CF_EPOCHS:-8}; RANK=${CF_RANK:-8}; NUMNEG=${CF_NUMNEG:-4}; LAMBDA=${CF_LAMBDA:-1.0}
CK=checkpoints/predictor_cf_${M}_s${SEED}.pt

# --- closed-loop eval (upstream CEM parity via scripts/18 defaults) ---
TASKS=${CF_TASKS:-"mw-reach mw-push mw-pick-place"}
ARMS=${CF_ARMS:-"l2 l2lora"}
EPISODES=${CF_EPISODES:-16}
OUT=${CF_OUT:-results/metaworld_cf_planning_s${SEED}.csv}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
echo "model=$M seed=$SEED  recipe: rank=$RANK epochs=$EPOCHS num_neg=$NUMNEG lambda_cf=$LAMBDA"
echo "arms=[$ARMS] tasks=[$TASKS] episodes=$EPISODES out=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
# training reads the metaworld latent cache + regimes (scripts/03 + 04 must have run)
ls -la data/precomputed_latents/*${M}* 2>/dev/null || echo "WARN: no metaworld latent cache found for $M"
set -e

echo "### seed $SEED (1/2) — train CF predictor-LoRA on MetaWorld (scripts/40) ###"
$PY scripts/40_train_predictor_cf.py --config "$CFG" --model "$M" \
    --seed "$SEED" --epochs "$EPOCHS" --rank "$RANK" \
    --num-neg "$NUMNEG" --lambda-cf "$LAMBDA" --out "$CK"
# scripts/40 prints the GATE (cf_rank_acc frozen->corrected, recon ratio) to the log above.
ls -la "$CK"

echo "### seed $SEED (2/2) — closed-loop push k/16, frozen vs CF-LoRA (scripts/18) ###"
$PY scripts/18_closed_loop_eval.py \
    --config "$CFG" --model "$M" \
    --predictor-lora "$CK" \
    --tasks $TASKS --arms $ARMS --episodes "$EPISODES" \
    --strict-success --out "$OUT"

echo "===== CF_PLANNING_MW_s${SEED}_DONE ====="; date
echo "--- $OUT ---"; cat "$OUT"
