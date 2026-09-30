#!/usr/bin/env bash
#SBATCH --job-name=ar_cube_bank_diag
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/action_response_cube/logs/%x_%j.out
set -euo pipefail

PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
RUN=/mnt/data/nhatnc129/jepa/action_response_cube/run_55413_20260928T023327Z
"$PY" /home/nhatnc129/nhat.nc/jepa-research/action_response_cube_20260928/diagnose_banks.py \
  "$RUN/cem_branches.npz" --output "$RUN/bank_diagnostics.json"
