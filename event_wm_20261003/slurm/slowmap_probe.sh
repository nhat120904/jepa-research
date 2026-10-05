#!/usr/bin/env bash
#SBATCH --job-name=ew_smprobe
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# PRIVILEGED probes of a slow-feature map: cube positions (keypoint probe) and arm joints (MLP), vs raw pixels.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
SM=${SM:?SM}; ENV=${ENV:-visual-cube-triple-play-v0}
"$TORCH_PY" "$PROJECT/scripts/slowmap_probe.py" --cache "$RUN_ROOT/cache/$ENV" --slowmap "$SM/$ENV/slowmap.pt" --out "$SM/$ENV"
