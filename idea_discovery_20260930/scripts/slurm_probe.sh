#!/usr/bin/env bash
#SBATCH --job-name=idea_probe
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/idea_discovery_20260930/logs/probe_%j.out
set -euo pipefail
ROOT=/home/nhatnc129/nhat.nc/jepa-research/idea_discovery_20260930
RUNROOT=/mnt/data/nhatnc129/jepa/idea_discovery_20260930
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
CKPT=/mnt/data/nhatnc129/jepa/ossckpt/vjepa2_opensource/vjepa2_vit_giant.pth
export OMP_NUM_THREADS=4
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export PYTHONPATH="$RUNROOT/extra${PYTHONPATH:+:$PYTHONPATH}"
TASK=${TASK:?TASK must be physics, tracking, or failure}
OUT="$RUNROOT/runs/${TASK}_$SLURM_JOB_ID"
mkdir -p "$OUT"
"$PY" "$ROOT/scripts/probe_${TASK}.py" \
    --data-root "$RUNROOT/assets" --out "$OUT" \
    --vjepa-source "$ROOT/external/vjepa2" --checkpoint "$CKPT" \
    ${PROBE_ARGS:-}
