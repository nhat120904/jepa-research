#!/usr/bin/env bash
#SBATCH --job-name=jepa_cf_rcasa
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseH_cf_robocasa_%j.out
# Phase H on RoboCasa — the counterfactual predictor objective as a SECOND
# beat-baseline arena (RoboCasa Pick / PnPCounterTop) beyond DROID Action-Score.
# Same locked A/B as slurm_phaseH_cf_droid.sh: train predictor-CF (scripts/40, latent
# InfoNCE-over-actions), then Action-Score (scripts/08) frozen vs cf-LoRA with
# IDENTICAL CEM settings so only the predictor differs.
#
# DEPENDS on the RoboCasa diagnostic pipeline (slurm_robocasa_pipeline.sh) having
# produced the dino_wm_droid latent cache + regimes; chain with
# --dependency=afterok:<pipeline_jobid>.
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
# The RoboCasa loader (scripts/08 plans on real transitions) walks $JEPAWM_DSET/robocasa/.
export JEPAWM_DSET="$PWD/data"
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
CFG=configs/diagnostic_robocasa.yaml
M=${P4H_MODEL:-dino_wm_droid}
EPOCHS=${P4H_EPOCHS:-8}
RANK=${P4H_RANK:-8}
ALPHA=${P4H_ALPHA:-16}
LCF=${P4H_LAMBDA_CF:-1.0}
NNEG=${P4H_NUM_NEG:-4}
NS=${P4H_CEM_SAMPLES:-64}
IT=${P4H_CEM_ITERS:-15}
MP=${P4H_MAX_PLAN:-40}
CK=checkpoints/predictor_cf_${M}_robocasa.pt
SPLIT=checkpoints/splits/phaseH_robocasa_${M}_split0.json
FROZEN_CSV=results/robocasa_planning_cf_${M}_frozen.csv
LORA_CSV=results/robocasa_planning_cf_${M}_lora.csv

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date) M=$M lambda_cf=$LCF JEPAWM_DSET=$JEPAWM_DSET"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
set -e

echo "### Phase H RoboCasa (1/3) — train predictor-CF (latent InfoNCE-over-actions) ###"
$PY scripts/40_train_predictor_cf.py --config "$CFG" --model "$M" \
    --split-seed 0 --split-manifest "$SPLIT" --val-frac 0.15 --test-frac 0.15 \
    --epochs "$EPOCHS" --rank "$RANK" --alpha "$ALPHA" \
    --lambda-cf "$LCF" --num-neg "$NNEG" --out "$CK"

echo "### Phase H RoboCasa (2/3) — Action-Score: FROZEN baseline (no LoRA) ###"
$PY scripts/08_planning_probe.py --config "$CFG" --only-model "$M" \
    --eval-split test --split-manifest "$SPLIT" \
    --cem-num-samples "$NS" --cem-iterations "$IT" --max-planning-transitions "$MP" \
    --out-csv "$FROZEN_CSV"

echo "### Phase H RoboCasa (3/3) — Action-Score: cf-LoRA predictor ###"
$PY scripts/08_planning_probe.py --config "$CFG" --only-model "$M" \
    --eval-split test --split-manifest "$SPLIT" \
    --cem-num-samples "$NS" --cem-iterations "$IT" --max-planning-transitions "$MP" \
    --predictor-lora "$CK" \
    --out-csv "$LORA_CSV"

echo "===== PHASEH_CF_ROBOCASA_DONE ====="; date
$PY - "$FROZEN_CSV" "$LORA_CSV" <<'PYEOF'
import sys, pandas as pd
f = pd.read_csv(sys.argv[1]); l = pd.read_csv(sys.argv[2])
key = ["regime", "horizon"]
m = f.merge(l, on=key, suffixes=("_frozen", "_lora"))
m["delta_err"] = m["action_error_lora"] - m["action_error_frozen"]
print(m[key + ["n_planned_frozen", "action_error_frozen", "action_error_lora",
               "cra_eff_frozen", "cra_eff_lora", "delta_err"]].to_string(index=False))
print(f"\nmean Action-Error  frozen={f['action_error'].mean():.4f}  cf-LoRA={l['action_error'].mean():.4f}")
print(f"mean CRA_eff       frozen={f['cra_eff'].mean():.4f}  cf-LoRA={l['cra_eff'].mean():.4f}")
print(f"mean Action-Score  frozen={f['action_score'].mean():.4f}  cf-LoRA={l['action_score'].mean():.4f}")
PYEOF
