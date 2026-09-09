#!/usr/bin/env bash
#SBATCH --job-name=erwm_e0_vkprobe
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/erwm_e0_vkprobe_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_resolved_20260908"
VENV=/mnt/data/nhatnc129/jepa/erwm_e0/.venv

export PYTHONPATH="$PROJECT:${PYTHONPATH:-}"
cd "$PROJECT"

echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} $(date -u +%FT%TZ)"
nvidia-smi
env | sort | rg '^(CUDA|LD_LIBRARY_PATH|VK_|DISPLAY|EGL)' || true
find /etc/vulkan /usr/share/vulkan /usr/local/share/vulkan -maxdepth 3 -type f -name '*nvidia*.json' -print 2>/dev/null || true
ldconfig -p | rg 'lib(GLX|EGL)_nvidia|libvulkan' || true

probe() {
  local tag=$1
  local icd=${2:-}
  echo "PROBE=$tag ICD=${icd:-unset}"
  if [[ -n "$icd" ]]; then
    VK_ICD_FILENAMES="$icd" timeout 45 "$VENV/bin/python" - <<'PY'
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import strike_slide_env  # noqa: F401

env = gym.make(
    "ERStrikeSlide-v0", num_envs=1, obs_mode="state_dict", reward_mode="none",
    control_mode="pd_ee_target_delta_pos", render_mode=None, sim_backend="cpu",
    physics_hz=500,
)
env.reset(seed=61000)
print("GYM_MAKE_OK")
env.close()
PY
  else
    timeout 45 "$VENV/bin/python" - <<'PY'
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import strike_slide_env  # noqa: F401

env = gym.make(
    "ERStrikeSlide-v0", num_envs=1, obs_mode="state_dict", reward_mode="none",
    control_mode="pd_ee_target_delta_pos", render_mode=None, sim_backend="cpu",
    physics_hz=500,
)
env.reset(seed=61000)
print("GYM_MAKE_OK")
env.close()
PY
  fi
}

probe default || true
for icd in /etc/vulkan/icd.d/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json; do
  [[ -f "$icd" ]] && probe "$(basename "$(dirname "$icd")")" "$icd" || true
done
