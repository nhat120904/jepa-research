#!/usr/bin/env bash
#SBATCH --job-name=jepa_plangen
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/planner_generality_%j.out
# Planner-generality ablation (reviewer point #2: "the planner is the adversary" is
# only shown for CEM). Re-runs the DECISIVE latent-oracle rungs under three test-time
# optimizers — cem (elite-refit), mppi (softmax-weighted), shooting (fixed proposal,
# argmin over samples x iters) — holding dynamics perfect, cost, budget, seeds fixed.
# All three minimise the SAME exploitable stateprobe cost.
#   Expected (thesis holds): push x stateprobe stays 0-2/16 under ALL THREE planners,
#     while the honest control reach x l2 still converts search->success under all three
#     (proves the harness is healthy, not that these planners are simply weaker).
# Costs use the OFF-POLICY-HARDENED probes (same as the exploitation table / overopt sweep).
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
M=${PLANGEN_MODEL:-dino_wm_metaworld}
EPISODES=${PLANGEN_EPISODES:-16}
PLANNERS=${PLANGEN_PLANNERS:-"cem mppi shooting"}
MPPI_BETA=${PLANGEN_MPPI_BETA:-5.0}
OBJ=checkpoints/spatial_object_probe_${M}_offpolicy.pt
EEP=checkpoints/ee_probe_${M}_offpolicy.pt

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "model=$M planners=[$PLANNERS] episodes=$EPISODES mppi_beta=$MPPI_BETA"
echo "probes: obj=$OBJ ee=$EEP"

set -e
for PL in $PLANNERS; do
  echo "===== PLANNER=$PL : exploitable rung (push x stateprobe) ====="
  $PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
      --cost stateprobe --probe "$OBJ" --ee-probe "$EEP" \
      --tasks mw-push --episodes "$EPISODES" --strict-success \
      --planner "$PL" --mppi-beta "$MPPI_BETA" \
      --out "results/latent_oracle_stateprobe_${PL}.csv"

  echo "===== PLANNER=$PL : honest control (reach x l2) ====="
  $PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
      --cost l2 \
      --tasks mw-reach --episodes "$EPISODES" --strict-success \
      --planner "$PL" --mppi-beta "$MPPI_BETA" \
      --out "results/latent_oracle_l2reach_${PL}.csv"
done

echo "===== PLANNER_GENERALITY_DONE ====="; date
echo "--- push x stateprobe (exploitable) success_end by planner ---"
for PL in $PLANNERS; do
  F="results/latent_oracle_stateprobe_${PL}.csv"
  [ -f "$F" ] && awk -F, 'NR>1{s+=$5;n++} END{printf "  %-9s push  %d/%d\n", "'"$PL"'", s, n}' "$F"
done
echo "--- reach x l2 (honest control) success_end by planner ---"
for PL in $PLANNERS; do
  F="results/latent_oracle_l2reach_${PL}.csv"
  [ -f "$F" ] && awk -F, 'NR>1{s+=$5;n++} END{printf "  %-9s reach %d/%d\n", "'"$PL"'", s, n}' "$F"
done
