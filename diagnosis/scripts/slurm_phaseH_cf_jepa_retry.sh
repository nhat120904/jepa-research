#!/usr/bin/env bash
#SBATCH --job-name=jepa_p4hj
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseH_jepa_l%a_%j.out
# Phase H cross-model retry — STRONGER cf training on jepa_wm_droid (~300M DINOv3).
#
# The rank8/8ep recipe that beats baseline on dino_wm_droid (22M) left jepa_wm_droid
# FLAT (Action-Error 1.407->1.423) though its CRA_eff rose 0.078->0.246. That partial
# CRA gain suggests under-training on the bigger predictor, not necessarily a genuine
# non-transfer. This retry gives it more capacity + epochs (rank 16, alpha 32, 16 ep)
# and sweeps lambda_cf to see whether jepa CRA reaches ~0.5 AND Action-Error drops.
# The FROZEN baseline is reused (results/droid_planning_cf_jepa_wm_droid_frozen.csv).
#
# Submit: sbatch --array=0-1 scripts/slurm_phaseH_cf_jepa_retry.sh   (lambda 1.0, 3.0)
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
M=jepa_wm_droid
RANK=${P4HJ_RANK:-16}; ALPHA=${P4HJ_ALPHA:-32}; EPOCHS=${P4HJ_EPOCHS:-16}
NS=${P4H_CEM_SAMPLES:-64}; IT=${P4H_CEM_ITERS:-15}; MP=${P4H_MAX_PLAN:-40}
LAMBDAS=(1.0 3.0)
IDX=${SLURM_ARRAY_TASK_ID:-0}
LCF=${P4HJ_LAMBDA:-${LAMBDAS[$IDX]}}
TAG=$(echo "$LCF" | tr '.' 'p')
CK=checkpoints/predictor_cf_${M}_r${RANK}_l${TAG}.pt
SPLIT=checkpoints/splits/phaseH_${M}_split0.json
OUT=results/droid_planning_cf_${M}_r${RANK}_l${TAG}.csv
FROZEN=results/droid_planning_cf_${M}_frozen.csv

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date) M=$M rank=$RANK ep=$EPOCHS lambda_cf=$LCF"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
[ -f "$FROZEN" ] || { echo "MISSING jepa frozen baseline $FROZEN — run the base Phase H jepa job first. STOP." >&2; exit 1; }
set -e

echo "### jepa retry lambda=$LCF (1/2) — train predictor-CF (rank $RANK, $EPOCHS ep) ###"
$PY scripts/40_train_predictor_cf.py --config "$CFG" --model "$M" \
    --split-seed 0 --split-manifest "$SPLIT" --val-frac 0.15 --test-frac 0.15 \
    --rank "$RANK" --alpha "$ALPHA" --epochs "$EPOCHS" \
    --lambda-cf "$LCF" --num-neg 4 --out "$CK"

echo "### jepa retry lambda=$LCF (2/2) — Action-Score with cf-LoRA ###"
$PY scripts/08_planning_probe.py --config "$CFG" --only-model "$M" \
    --eval-split test --split-manifest "$SPLIT" \
    --cem-num-samples "$NS" --cem-iterations "$IT" --max-planning-transitions "$MP" \
    --predictor-lora "$CK" --out-csv "$OUT"

echo "===== PHASEH_JEPA_L${TAG}_DONE (lambda=$LCF) ====="; date
$PY - "$LCF" "$FROZEN" "$OUT" <<'PYEOF'
import sys, pandas as pd
lcf, fp, lp = sys.argv[1], sys.argv[2], sys.argv[3]
f = pd.read_csv(fp); l = pd.read_csv(lp)
print(f"lambda_cf={lcf}  mean Action-Error frozen={f['action_error'].mean():.4f} "
      f"cf-LoRA={l['action_error'].mean():.4f}  "
      f"mean CRA_eff frozen={f['cra_eff'].mean():.4f} cf-LoRA={l['cra_eff'].mean():.4f}")
PYEOF
