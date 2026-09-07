#!/usr/bin/env bash
#SBATCH --job-name=order_src
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_src_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/order-jepa/scripts/_common.sh

PIN=0a9492fa12044b852ae9e001cc74604b79c8bb0c
if [[ ! -d "$DINO_WM_ROOT/.git" ]]; then
  git clone --no-checkout https://github.com/gaoyuezhou/dino_wm.git "$DINO_WM_ROOT"
  git -C "$DINO_WM_ROOT" checkout --detach "$PIN"
fi
test "$(git -C "$DINO_WM_ROOT" rev-parse HEAD)" = "$PIN"
test "$(git -C "$DINO_WM_ROOT" remote get-url origin)" = "https://github.com/gaoyuezhou/dino_wm.git"
echo "ORIGINAL SOURCE VERIFIED commit=$PIN"

