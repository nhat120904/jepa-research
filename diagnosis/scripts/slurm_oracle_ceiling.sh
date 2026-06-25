#!/usr/bin/env bash
#SBATCH --job-name=jepa_oracle
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/oracle_ceiling_%j.out
# Oracle planning ceiling (scripts/29) — the go/no-go for the model-side fix.
# Pure goal-conditioned sim-MPC: perfect dynamics (MuJoCo) + perfect object state,
# same env / horizon / num_act_stepped / strict success as the WM eval. A GPU is
# requested only because make_env builds an EGL offscreen renderer (the planner
# itself never touches the encoder or any checkpoint).
#   oracle ≫ 0 on contact  -> task solvable here; the gap is the MODEL -> pursue
#                             the GT-supervised predictor fix (scripts/26 λ_msk>0)
#   oracle  = 0 on contact  -> the PLANNING PROTOCOL is the bottleneck, not the
#                             predictor; the paper's solution must target planning
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
OUT=${ORACLE_OUT:-results/metaworld_oracle_ceiling.csv}
EPISODES=${ORACLE_EPISODES:-16}
TASKS=${ORACLE_TASKS:-"mw-reach mw-push mw-pick-place"}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "tasks=[$TASKS] episodes=$EPISODES out=$OUT"

set -e
$PY scripts/29_oracle_ceiling.py --config "$CFG" \
    --tasks $TASKS --episodes "$EPISODES" --out "$OUT" --strict-success
echo "===== ORACLE_CEILING_DONE ====="; date
cat "$OUT"
