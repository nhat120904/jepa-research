#!/usr/bin/env bash
#SBATCH --job-name=jepa_p3diag
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phase3_diag_%j.out
# Phase-3 3a (cheap, decisive): measure the object + ee probe decode error on the
# OFF-POLICY distribution (random-action sim frames) the planner actually scores, vs
# Test-1b's 92% <5cm on EXPERT frames. Encode-only, no planning → fast.
#   off-policy <5cm COLLAPSES vs 92%  -> root cause confirmed; run slurm_phase3_offpolicy.sh
#   off-policy <5cm stays high        -> readout fine off-policy; re-open planner/cost-noise
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
M=${P3_MODEL:-dino_wm_metaworld}
EPISODES=${P3_EPISODES:-16}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null

set -e
echo "### 3a object readout off-policy ###"
$PY scripts/34_offpolicy_precision.py --config "$CFG" --model "$M" \
    --probe checkpoints/spatial_object_probe_${M}.pt --target obj --episodes "$EPISODES" \
    --out results/offpolicy_precision_obj.csv
echo "### 3a ee readout off-policy ###"
$PY scripts/34_offpolicy_precision.py --config "$CFG" --model "$M" \
    --probe checkpoints/ee_probe_${M}.pt --target ee --episodes "$EPISODES" \
    --out results/offpolicy_precision_ee.csv
echo "===== PHASE3_DIAG_DONE ====="; date
