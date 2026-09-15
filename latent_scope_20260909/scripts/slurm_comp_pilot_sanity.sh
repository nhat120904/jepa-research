#!/usr/bin/env bash
#SBATCH --job-name=lscope_pilot_sanity
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/comp_pilot_sanity_%j.out
set -euo pipefail
PY=/mnt/data/nhatnc129/jepa/latent_scope_baseline/policy_venv/bin/python
"$PY" /home/nhatnc129/nhat.nc/jepa-research/latent_scope_20260909/comp_pilot/analyze_pilot.py "$RUN"
