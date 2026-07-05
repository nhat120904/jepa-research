#!/usr/bin/env bash
#SBATCH --job-name=jepa_p4h
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseH_cf_%j.out
# Phase H — the PROPOSAL's counterfactual objective on the PREDICTOR, evaluated on
# DROID Action-Score (the beat-baseline arena). This is the axis Phase 0-G never
# tested: every prior phase swapped the planning COST under PERFECT dynamics (F
# removed) and got reward-hacked. Here we instead fine-tune F itself with the
# latent-space counterfactual InfoNCE objective (scripts/40) and measure whether the
# L2-CEM planner recovers better actions.
#
# WHY it can beat baseline where the cost-side program didn't: DROID Action-Score
# measures action DISCRIMINATION (net planned vs expert delta), which is exactly what
# L_cf optimises, and today's baseline is action-blind there (CRA_eff ≈ chance,
# droid_planning_safe.csv). It is a different, winnable evaluation from the MetaWorld
# contact-oracle — and compared to the actual published-weak baseline, not the oracle
# ceiling.
#
# A/B is locked: frozen vs cf-LoRA run back-to-back with IDENTICAL CEM settings
# (64 samples x 15 iters, 40 tx/cell — matches droid_planning_safe.csv), so the only
# difference is the predictor. Compare Action-Error (d_ref-independent; lower better).
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
CFG=configs/diagnostic_droid.yaml
M=${P4H_MODEL:-dino_wm_droid}
EPOCHS=${P4H_EPOCHS:-8}
RANK=${P4H_RANK:-8}
ALPHA=${P4H_ALPHA:-16}
LCF=${P4H_LAMBDA_CF:-1.0}
NNEG=${P4H_NUM_NEG:-4}
NS=${P4H_CEM_SAMPLES:-64}
IT=${P4H_CEM_ITERS:-15}
MP=${P4H_MAX_PLAN:-40}
CK=checkpoints/predictor_cf_${M}.pt
# Model-tagged outputs so a 2nd model (jepa_wm_droid) does NOT clobber dino's results.
FROZEN_CSV=results/droid_planning_cf_${M}_frozen.csv
LORA_CSV=results/droid_planning_cf_${M}_lora.csv

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date) M=$M lambda_cf=$LCF num_neg=$NNEG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
set -e

echo "### Phase H (1/3) — train predictor-CF (latent InfoNCE-over-actions) ###"
$PY scripts/40_train_predictor_cf.py --config "$CFG" --model "$M" \
    --epochs "$EPOCHS" --rank "$RANK" --alpha "$ALPHA" \
    --lambda-cf "$LCF" --num-neg "$NNEG" --out "$CK"

echo "### Phase H (2/3) — Action-Score: FROZEN baseline (no LoRA) ###"
$PY scripts/08_planning_probe.py --config "$CFG" --only-model "$M" \
    --cem-num-samples "$NS" --cem-iterations "$IT" --max-planning-transitions "$MP" \
    --out-csv "$FROZEN_CSV"

echo "### Phase H (3/3) — Action-Score: cf-LoRA predictor ###"
$PY scripts/08_planning_probe.py --config "$CFG" --only-model "$M" \
    --cem-num-samples "$NS" --cem-iterations "$IT" --max-planning-transitions "$MP" \
    --predictor-lora "$CK" \
    --out-csv "$LORA_CSV"

echo "===== PHASEH_CF_DONE ====="; date
echo "--- Action-Error by (regime,horizon): FROZEN vs cf-LoRA (lower = better) ---"
$PY - "$FROZEN_CSV" "$LORA_CSV" <<'PYEOF'
import sys, pandas as pd
f = pd.read_csv(sys.argv[1])
l = pd.read_csv(sys.argv[2])
key = ["regime", "horizon"]
m = f.merge(l, on=key, suffixes=("_frozen", "_lora"))
cols = key + ["n_planned_frozen", "action_error_frozen", "action_error_lora",
             "cra_eff_frozen", "cra_eff_lora"]
m = m[cols].copy()
m["delta_err"] = m["action_error_lora"] - m["action_error_frozen"]   # negative = LoRA better
print(m.to_string(index=False))
print(f"\nmean Action-Error  frozen={f['action_error'].mean():.4f}  "
      f"cf-LoRA={l['action_error'].mean():.4f}")
print(f"mean CRA_eff       frozen={f['cra_eff'].mean():.4f}  cf-LoRA={l['cra_eff'].mean():.4f}")
print(f"mean Action-Score  frozen={f['action_score'].mean():.4f}  "
      f"cf-LoRA={l['action_score'].mean():.4f}  (note: different d_ref per run)")
PYEOF
