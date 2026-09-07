#!/usr/bin/env bash
#SBATCH --job-name=order_osf_ls
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_osf_ls_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/order-jepa/scripts/_common.sh
"$PY" "$PROJECT/scripts/discover_osf_release.py" \
  --out /mnt/data/nhatnc129/jepa/dino_wm_official/osf_release_tree.json
