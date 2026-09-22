#!/bin/bash
#SBATCH --job-name=mikasa_gate0_probe
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/mikasa_gate0/logs/gate0_probe_%j.out
#SBATCH --error=/mnt/data/nhatnc129/jepa/mikasa_gate0/logs/gate0_probe_%j.out

# Gate 0 probe: Vulkan/renderer availability and simulation throughput.
# Hard 30 minute cap; every probe also has its own subprocess timeout.

set -uo pipefail

ROOT=/mnt/data/nhatnc129/jepa/mikasa_gate0
VENV="$ROOT/.venv"
OUT="$ROOT/outputs/probe_${SLURM_JOB_ID}"
SCRIPTS=/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/scripts

mkdir -p "$OUT"

echo "[probe] host=$(hostname) job=${SLURM_JOB_ID} $(date -Is)"
echo "[probe] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Keep everything offline-friendly and quiet; ManiSkill downloads its own assets
# for PickCube on first use, which is small.
export MS_ASSET_DIR="$ROOT/assets"
mkdir -p "$MS_ASSET_DIR"

"$VENV/bin/python" "$SCRIPTS/mikasa_gate0_driver.py" \
  --out-dir "$OUT" \
  --python "$VENV/bin/python"
rc=$?

echo "[probe] driver exit=$rc $(date -Is)"
exit $rc
