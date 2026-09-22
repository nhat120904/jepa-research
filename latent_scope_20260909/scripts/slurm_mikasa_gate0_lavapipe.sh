#!/bin/bash
#SBATCH --job-name=mikasa_gate0_lvp
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/mikasa_gate0/logs/gate0_lavapipe_%j.out
#SBATCH --error=/mnt/data/nhatnc129/jepa/mikasa_gate0/logs/gate0_lavapipe_%j.out

# Last Gate 0 attempt. Job 53287 showed the node carries a COMPUTE-ONLY NVIDIA
# driver: libnvidia-gpucomp.so.570.133.20 exists but libGLX_nvidia.so.0 does not,
# so every NVIDIA Vulkan ICD is skipped by the loader.
#
# SAPIEN's _ensure_vulkan_icd() returns early when VK_ICD_FILENAMES is already set,
# so pointing it at the system lavapipe ICD is respected. lavapipe is a CPU software
# Vulkan implementation and was never exercised by the earlier probes.
#
# Question: does lavapipe let ManiSkill run at all, and is state-observation
# throughput usable? If not, MIKASA-Robo is out on this cluster. No further attempts.

set -uo pipefail

ROOT=/mnt/data/nhatnc129/jepa/mikasa_gate0
VENV="$ROOT/.venv"
OUT="$ROOT/outputs/lavapipe_${SLURM_JOB_ID}"
SCRIPTS=/home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/scripts
mkdir -p "$OUT"

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.x86_64.json
export MS_ASSET_DIR="$ROOT/assets"
mkdir -p "$MS_ASSET_DIR"

echo "[lvp] host=$(hostname) job=${SLURM_JOB_ID} $(date -Is)"
echo "[lvp] VK_ICD_FILENAMES=$VK_ICD_FILENAMES"

echo "=== does SAPIEN create a Vulkan instance under lavapipe? ==="
"$VENV/bin/python" - 2>&1 <<'PY' | tail -20
import sapien
try:
    sapien.Scene()
    print("SAPIEN_SCENE_OK")
except BaseException as exc:
    print(f"SAPIEN_SCENE_FAILED {exc!r}")
PY

echo "=== throughput probes under lavapipe ==="
for spec in "state 1 physx_cpu" "state 1 gpu" "state 16 gpu" "rgb 1 gpu"; do
  set -- $spec
  mode=$1; n=$2; backend=$3
  name="lvp_${mode}_n${n}_${backend}"
  echo "--- $name"
  timeout 300 "$VENV/bin/python" "$SCRIPTS/mikasa_gate0_worker.py" \
    --obs-mode "$mode" --num-envs "$n" --sim-backend "$backend" \
    --out "$OUT/${name}.json" >/dev/null 2>&1
  "$VENV/bin/python" -c "
import json
try:
    w = json.load(open('$OUT/${name}.json'))
    print('  ok=%s stage=%s fps=%s err=%s' % (
        w.get('ok'), w.get('stage_reached'), w.get('fps_env_steps'),
        str(w.get('error'))[:160]))
except FileNotFoundError:
    print('  no JSON produced (hard abort or timeout)')
"
done

echo "[lvp] artifacts in $OUT"
echo "[lvp] done $(date -Is)"
