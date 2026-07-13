#!/usr/bin/env bash
#SBATCH --job-name=jepa_latoracle
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/latent_oracle_%j.out
# Latent oracle (scripts/30) — separates PREDICTOR from ENCODER. Run AFTER the
# state oracle (scripts/29) returns "model is the gap" on contact. Perfect latent
# dynamics through the REAL frozen encoder (sim-step -> render -> encode), same
# upstream L2-in-latent cost / horizon / strict success as scripts/18; only F is
# swapped out.
#   latent-oracle PASS + WM-l2 FAIL -> PREDICTOR is the bottleneck (license the
#                                       GT-supervised predictor fix, scripts/26)
#   latent-oracle FAIL              -> ENCODER representation / L2-latent cost is
#                                       the bottleneck (predictor fix can't help)
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
M=${LATORACLE_MODEL:-dino_wm_metaworld}
OUT=${LATORACLE_OUT:-results/metaworld_latent_oracle.csv}
EPISODES=${LATORACLE_EPISODES:-16}
TASKS=${LATORACLE_TASKS:-"mw-reach mw-push mw-pick-place"}
COST=${LATORACLE_COST:-l2}
PROBE=${LATORACLE_PROBE:-}
EEPROBE=${LATORACLE_EEPROBE:-}
PLANNER=${LATORACLE_PLANNER:-cem}          # cem | mppi | shooting (planner-generality ablation)
MPPI_BETA=${LATORACLE_MPPI_BETA:-5.0}
EXTRA_ARGS=(--planner "$PLANNER" --mppi-beta "$MPPI_BETA")
[ -n "$PROBE" ] && EXTRA_ARGS+=(--probe "$PROBE")
[ -n "$EEPROBE" ] && EXTRA_ARGS+=(--ee-probe "$EEPROBE")

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "model=$M cost=$COST planner=$PLANNER tasks=[$TASKS] episodes=$EPISODES out=$OUT probe=${PROBE:-<none>} ee=${EEPROBE:-<none>}"

set -e
$PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" --cost "$COST" \
    --tasks $TASKS --episodes "$EPISODES" --out "$OUT" --strict-success "${EXTRA_ARGS[@]}"
echo "===== LATENT_ORACLE_DONE ====="; date
cat "$OUT"
