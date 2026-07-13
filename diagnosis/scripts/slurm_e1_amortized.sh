#!/usr/bin/env bash
#SBATCH --job-name=jepa_e1_gcidm
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/e1_amortized_%j.out
# E1 — amortized GC-IDM control + search-dose response
# (docs/plans/2026-07-06-e1-amortized-control-design.md). Arms: pure gcidm (no cost,
# no search) + cemseed_it{0,2,6,12,24} (CEM/stateprobe seeded with the gcidm proposal,
# increasing search dose). Crown measurement: per-replan seed-vs-chosen corruption rate.
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
M=${E1_MODEL:-dino_wm_metaworld}
TASKS=${E1_TASKS:-"mw-push"}
ITERS=${E1_ITERS:-"0 2 6 12 24"}
EPISODES=${E1_EPISODES:-8}
OBJ=checkpoints/spatial_object_probe_${M}_offpolicy.pt
EEP=checkpoints/ee_probe_${M}_offpolicy.pt
INV=checkpoints/inverse_proposal_${M}.pt

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "model=$M tasks=[$TASKS] iters=[$ITERS] episodes=$EPISODES"

set -e
# Smoke: 1 episode, gcidm + it0 + it2 only — fail fast before the real sweep.
$PY scripts/43_e1_amortized_control.py --config "$CFG" --model "$M" \
    --tasks $TASKS --iters-grid 0 2 --episodes 1 \
    --probe "$OBJ" --ee-probe "$EEP" --inverse-head "$INV" --strict-success \
    --out-episodes results/e1_smoke_episodes.csv \
    --out-curves results/e1_smoke_curves.csv \
    --out-seedvs results/e1_smoke_seedvs.csv
echo "===== E1_SMOKE_OK ====="

$PY scripts/43_e1_amortized_control.py --config "$CFG" --model "$M" \
    --tasks $TASKS --iters-grid $ITERS --episodes "$EPISODES" \
    --probe "$OBJ" --ee-probe "$EEP" --inverse-head "$INV" --strict-success \
    --out-episodes results/e1_episodes.csv \
    --out-curves results/e1_curves.csv \
    --out-seedvs results/e1_seed_vs_chosen.csv
echo "===== E1_DONE ====="; date
