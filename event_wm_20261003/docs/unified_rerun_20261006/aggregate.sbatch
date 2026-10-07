#!/usr/bin/env bash
#SBATCH --job-name=ew_unagg
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
set -euo pipefail
export PYTHONUNBUFFERED=1 PYTHONPATH="$RUN_DIR/source:$RUN_DIR"
/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python "$RUN_DIR/aggregate.py" --root "$RUN_DIR"
