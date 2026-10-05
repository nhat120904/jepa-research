#!/usr/bin/env bash
#SBATCH --job-name=ew_cubestate
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Cube step 1: slow features -> ICA sources -> co-change object groups, scored against privileged qpos.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=visual-cube-triple-play-v0; BASE=${BASE:?BASE}
OUT=$RUN_ROOT/cube_state_${SLURM_JOB_ID}
record_source "$OUT"
"$TORCH_PY" "$PROJECT/scripts/cube_state.py" --cache "$RUN_ROOT/cache/$ENV" --kind triple --base "$BASE" --out "$OUT"
