#!/usr/bin/env bash
#SBATCH --job-name=jepa_recon
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/reconcile_%j.out
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export JEPAWM_DSET=/mnt/data/nhatnc129/jepa/datasets      # robocasa custom_teleop reads this
export BOTO_CONFIG=/dev/null
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"; nvidia-smi --query-gpu=name --format=csv,noheader

# 1) DROID — reuse 03 cache; re-classify + re-score on gripper-primary regimes
echo "########## DROID (re-04/05/12 on gripper-primary) ##########"; date
$PY scripts/04_classify_regimes.py --config configs/diagnostic_droid.yaml && echo OK04_droid || echo SKIP04_droid
$PY scripts/05_run_diagnostic.py   --config configs/diagnostic_droid.yaml && echo OK05_droid || echo SKIP05_droid
$PY scripts/12_boundary_diagnostic.py --config configs/diagnostic_droid.yaml && echo OK12_droid || echo SKIP12_droid

# 2) franka_custom — reuse cache; re-04/05 (droid-style regimes)
echo "########## franka_custom (re-04/05) ##########"; date
$PY scripts/04_classify_regimes.py --config configs/diagnostic_franka_custom.yaml && echo OK04_franka || echo SKIP04_franka
$PY scripts/05_run_diagnostic.py   --config configs/diagnostic_franka_custom.yaml && echo OK05_franka || echo SKIP05_franka

# 3) robocasa — fresh 03 (JEPAWM_DSET fix) per model, then 04/05
echo "########## robocasa (03 fix + 04/05) ##########"; date
for M in dino_wm_droid vjepa2_ac_droid; do
  echo "--- robocasa 03 $M ---"; date
  CAI_JEPA_ONLY_MODEL=$M $PY scripts/03_extract_latents.py --config configs/diagnostic_robocasa.yaml \
    && echo "OK extract $M" || echo "SKIP extract $M"
done
$PY scripts/04_classify_regimes.py --config configs/diagnostic_robocasa.yaml && echo OK04_robocasa || echo SKIP04_robocasa
$PY scripts/05_run_diagnostic.py   --config configs/diagnostic_robocasa.yaml && echo OK05_robocasa || echo SKIP05_robocasa

echo "===== RECONCILE_DONE ====="; date
