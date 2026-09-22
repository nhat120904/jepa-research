#!/bin/bash
#SBATCH --job-name=mikasa_gate0_vkdiag
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/mikasa_gate0/logs/gate0_vkdiag_%j.out
#SBATCH --error=/mnt/data/nhatnc129/jepa/mikasa_gate0/logs/gate0_vkdiag_%j.out

# Final bounded Gate 0 diagnostic after job 53286 returned
# vk::createInstanceUnique: ErrorIncompatibleDriver.
#
# Two questions only:
#   Q1 Is a usable NVIDIA Vulkan driver library present at all on this node?
#      (i.e. is this "one config file away", or is the driver's Vulkan side absent?)
#   Q2 Does ManiSkill's CPU physics backend avoid Vulkan entirely?
#
# If both answers are negative, MIKASA-Robo is out on this cluster. No further
# infrastructure attempts after this job.

set -uo pipefail

ROOT=/mnt/data/nhatnc129/jepa/mikasa_gate0
VENV="$ROOT/.venv"
OUT="$ROOT/outputs/vkdiag_${SLURM_JOB_ID}"
mkdir -p "$OUT"

echo "[vkdiag] host=$(hostname) job=${SLURM_JOB_ID} $(date -Is)"
nvidia-smi --query-gpu=name,driver_version --format=csv

echo "=== Q1a: NVIDIA / Vulkan libraries present on this node ==="
for pat in 'libGLX_nvidia.so*' 'libnvidia-glvkspirv*' 'libvulkan.so*' 'libvulkan_lvp*' 'libnvidia-gpucomp*'; do
  echo "--- $pat"
  find /usr/lib /usr/lib64 /usr/local/lib -maxdepth 3 -name "$pat" 2>/dev/null | head -5
done | tee "$OUT/libraries.txt"

echo "=== Q1b: any nvidia ICD manifest anywhere standard ==="
find /usr/share/vulkan /etc/vulkan /usr/local/share/vulkan /opt -maxdepth 4 \
  -name '*icd*.json' 2>/dev/null | head -20 | tee "$OUT/icd_manifests.txt"

echo "=== Q1c: Vulkan loader trace while SAPIEN creates an instance ==="
# VK_LOADER_DEBUG names every ICD the loader tried and why it was rejected.
VK_LOADER_DEBUG=all "$VENV/bin/python" - > "$OUT/loader_trace.txt" 2>&1 <<'PY'
import sapien
try:
    sapien.Scene()
    print("SAPIEN_SCENE_OK")
except BaseException as exc:
    print(f"SAPIEN_SCENE_FAILED {exc!r}")
PY
echo "--- loader trace (tail) ---"
tail -40 "$OUT/loader_trace.txt"
grep -ciE 'nvidia' "$OUT/loader_trace.txt" | sed 's/^/nvidia mentions in trace: /'

echo "=== Q2: does the CPU physics backend avoid Vulkan? ==="
"$VENV/bin/python" - > "$OUT/cpu_backend.txt" 2>&1 <<'PY'
import json
import time

result = {"ok": False, "error": None}
try:
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    env = gym.make(
        "PickCube-v1", num_envs=1, obs_mode="state", sim_backend="physx_cpu"
    )
    env.reset(seed=0)
    for _ in range(5):
        env.step(env.action_space.sample())
    t0 = time.perf_counter()
    for _ in range(50):
        env.step(env.action_space.sample())
    dt = time.perf_counter() - t0
    result = {"ok": True, "fps": round(50 / dt, 2)}
    env.close()
except BaseException as exc:
    result["error"] = repr(exc)
print(json.dumps(result, indent=2))
PY
cat "$OUT/cpu_backend.txt"

echo "[vkdiag] artifacts in $OUT"
echo "[vkdiag] done $(date -Is)"
