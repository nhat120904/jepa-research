#!/usr/bin/env bash
#SBATCH --job-name=ar_paired_analysis
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/action_response/logs/%x_%j.out
set -euo pipefail

PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
RUN_DIR=${RUN_DIR:-/mnt/data/nhatnc129/jepa/action_response/run_55387_20260928T013708Z}
"$PY" /home/nhatnc129/nhat.nc/jepa-research/action_response_20260927/analyze.py "$RUN_DIR"
