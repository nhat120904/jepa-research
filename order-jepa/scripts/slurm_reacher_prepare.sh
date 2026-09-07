#!/usr/bin/env bash
#SBATCH --job-name=order_rprep
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/order_rprep_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/order-jepa/scripts/_reacher_common.sh
sha256sum "$PROJECT/scripts/prepare_lewm_reacher.py" "$PROJECT/order_jepa/lewm_reacher.py"
"$PY" "$PROJECT/scripts/prepare_lewm_reacher.py" \
  --stable-worldmodel-source "$SWM_SOURCE" \
  --stablewm-home "$STAGE0_ROOT" \
  --out "$PROVENANCE"
