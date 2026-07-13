#!/usr/bin/env bash
#SBATCH --job-name=jepa_p4hs
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseH_cfseed_s%a_%j.out
# Phase H hardening — TRAINING-SEED sweep of the counterfactual predictor (scripts/40)
# on dino_wm_droid, to put a CI on the beat-baseline Action-Score result.
#
# Phase H seed 0 (job 24363) gave pooled Action-Error 1.468->1.086 (-26%), 3/6 cells
# CI-clean. This sweep runs seeds 1-3 of the SAME recipe and evaluates each with the
# SAME scripts/08 protocol (CEM 64x15, 40 tx/cell, fixed rng(42) subsample), so the 4
# training draws {0,1,2,3} are directly comparable and give a per-cell mean/spread. The
# FROZEN baseline is seed-independent (no training) -> reuse results/droid_planning_cf_frozen.csv.
#
# Submit: sbatch --array=1-3 scripts/slurm_phaseH_cf_seedsweep.sh
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
SEED=${SLURM_ARRAY_TASK_ID:-1}
EPOCHS=${P4H_EPOCHS:-8}
NS=${P4H_CEM_SAMPLES:-64}; IT=${P4H_CEM_ITERS:-15}; MP=${P4H_MAX_PLAN:-40}
CK=checkpoints/predictor_cf_${M}_s${SEED}.pt
OUT=results/droid_planning_cf_${M}_s${SEED}.csv

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date) M=$M seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
set -e

echo "### seed $SEED (1/2) — train predictor-CF ###"
$PY scripts/40_train_predictor_cf.py --config "$CFG" --model "$M" \
    --seed "$SEED" --epochs "$EPOCHS" --lambda-cf 1.0 --num-neg 4 --out "$CK"

echo "### seed $SEED (2/2) — Action-Score with cf-LoRA ###"
$PY scripts/08_planning_probe.py --config "$CFG" --only-model "$M" \
    --cem-num-samples "$NS" --cem-iterations "$IT" --max-planning-transitions "$MP" \
    --predictor-lora "$CK" --out-csv "$OUT"

echo "===== PHASEH_CFSEED_${SEED}_DONE ====="; date
$PY - "$SEED" "$OUT" <<'PYEOF'
import sys, pandas as pd
seed, out = sys.argv[1], sys.argv[2]
f = pd.read_csv("results/droid_planning_cf_frozen.csv")
l = pd.read_csv(out)
print(f"seed={seed}  mean Action-Error frozen={f['action_error'].mean():.4f} "
      f"cf-LoRA={l['action_error'].mean():.4f}  "
      f"mean CRA_eff frozen={f['cra_eff'].mean():.4f} cf-LoRA={l['cra_eff'].mean():.4f}")
PYEOF
