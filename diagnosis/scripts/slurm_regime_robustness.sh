#!/usr/bin/env bash
#SBATCH --job-name=jepa_regime_robust
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/regime_robust_%j.out
# Threshold-robustness for the Metaworld regime split (reviewer gap: the four
# regimes that stratify every headline table are defined by a fixed-threshold
# cascade — 5mm object move / 10cm pre-grasp / 0.10 gripper-delta — and a reviewer
# will ask whether the pre-grasp/contact chance-floor is an artifact of that cut).
# Re-runs scripts/04 (classify, CPU) + scripts/05 (diagnostic, hard_nn only, GPU)
# under perturbed thresholds and collates effect-CRA per regime (scripts/46).
#   effect-CRA at pre-grasp/contact stays near the 1/17 floor across all configs
#       -> the split choice is not load-bearing; the collapse is real. (expected)
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

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

# Always restore the canonical (baseline-threshold) regime sidecars on exit, so a
# perturbed sweep never leaves downstream runs reading the wrong regime labels.
restore_baseline () {
  echo "===== restoring baseline regime sidecars ====="
  env -u CAI_JEPA_GRIP_DELTA -u CAI_JEPA_OBJ_MOVE -u CAI_JEPA_PRE_GRASP \
      $PY scripts/04_classify_regimes.py --config "$CFG" || true
}
trap restore_baseline EXIT

run_cfg () {  # tag  OBJ_MOVE  PRE_GRASP  GRIP_DELTA
  local tag=$1 obj=$2 pg=$3 gd=$4
  echo ""; echo "########## CONFIG $tag: obj_move=$obj pre_grasp=$pg grip_delta=$gd ##########"
  CAI_JEPA_OBJ_MOVE=$obj CAI_JEPA_PRE_GRASP=$pg CAI_JEPA_GRIP_DELTA=$gd \
      $PY scripts/04_classify_regimes.py --config "$CFG"
  CAI_JEPA_ONLY_STRATEGY=hard_nn CAI_JEPA_OUTPUT_CSV=results/regime_robust_${tag}.csv \
      $PY scripts/05_run_diagnostic.py --config "$CFG"
}

set -e
#        tag         obj_move  pre_grasp  grip_delta      (baseline = 0.005 / 0.10 / 0.10)
run_cfg  base        0.005     0.10       0.10
run_cfg  objmove2p5  0.0025    0.10       0.10
run_cfg  objmove10   0.010     0.10       0.10
run_cfg  pregrasp8   0.005     0.08       0.10
run_cfg  pregrasp12  0.005     0.12       0.10
run_cfg  gripdelta05 0.005     0.10       0.05
run_cfg  gripdelta20 0.005     0.10       0.20

echo ""; echo "########## COLLATING ##########"
$PY scripts/46_analyze_regime_robustness.py \
    --glob 'results/regime_robust_*.csv' \
    --out results/regime_robustness_summary.csv

echo "===== REGIME_ROBUST_DONE ====="; date
