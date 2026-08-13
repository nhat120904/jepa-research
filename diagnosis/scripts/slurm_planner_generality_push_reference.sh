#!/usr/bin/env bash
#SBATCH --job-name=jepa_plangen_ref
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/planner_generality_ref_%j.out
# Closes the reviewer gap in Appendix E: the existing planner-robustness control
# (stateprobe near-failure under CEM/MPPI/shooting) has no physical-reference
# arm for MPPI/shooting, so a reviewer can say "maybe MPPI/shooting simply can't
# solve push under this budget regardless of cost." This adds the missing
# physical-reference (script 29, --planner) and latent-L2 (script 30, --cost l2)
# push arms under mppi and shooting, at the SAME seed0=10000/episodes=16 already
# used for results/latent_oracle_stateprobe_{cem,mppi,shooting}.csv and
# results/metaworld_oracle_ceiling.csv (cem push already 16/16 there).
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
SEED0=${PLANGEN_SEED0:-10000}
PLANNERS=${PLANGEN_PLANNERS:-"mppi shooting"}
MPPI_BETA=${PLANGEN_MPPI_BETA:-5.0}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
echo "model=$M planners=[$PLANNERS] episodes=$EPISODES seed0=$SEED0"

set -e
for PL in $PLANNERS; do
  echo "===== PLANNER=$PL : physical reference (push, no model) ====="
  $PY scripts/29_oracle_ceiling.py --config "$CFG" \
      --tasks mw-push --episodes "$EPISODES" --seed0 "$SEED0" --strict-success \
      --planner "$PL" --mppi-beta "$MPPI_BETA" \
      --out "results/oracle_ceiling_push_${PL}.csv"

  echo "===== PLANNER=$PL : latent L2 (push) ====="
  $PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
      --cost l2 \
      --tasks mw-push --episodes "$EPISODES" --seed0 "$SEED0" --strict-success \
      --planner "$PL" --mppi-beta "$MPPI_BETA" \
      --out "results/latent_oracle_l2push_${PL}.csv"
done

echo "===== PLANNER_GENERALITY_REFERENCE_DONE ====="; date

echo "--- push physical reference success_end by planner ---"
for PL in cem $PLANNERS; do
  F="results/oracle_ceiling_push_${PL}.csv"
  if [ "$PL" = "cem" ]; then F="results/metaworld_oracle_ceiling.csv"; fi
  [ -f "$F" ] && awk -F, -v task=mw-push 'NR==1{for(i=1;i<=NF;i++){if($i=="task")tcol=i;if($i=="success_end")scol=i}} NR>1 && $tcol==task{s+=$scol;n++} END{printf "  %-9s push %d/%d\n", "'"$PL"'", s, n}' "$F"
done

echo "--- push latent L2 success_end by planner ---"
for PL in $PLANNERS; do
  F="results/latent_oracle_l2push_${PL}.csv"
  [ -f "$F" ] && awk -F, 'NR>1{s+=$5;n++} END{printf "  %-9s push %d/%d\n", "'"$PL"'", s, n}' "$F"
done

echo "--- push stateprobe success_end by planner (already present) ---"
for PL in cem $PLANNERS; do
  F="results/latent_oracle_stateprobe_${PL}.csv"
  [ -f "$F" ] && awk -F, 'NR>1{s+=$5;n++} END{printf "  %-9s push %d/%d\n", "'"$PL"'", s, n}' "$F"
done
