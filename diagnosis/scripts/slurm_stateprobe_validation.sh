#!/usr/bin/env bash
#SBATCH --job-name=jepa_probeval
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/stateprobe_validation_%j.out

set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
export PATH="$PWD/.venv/bin:$PATH"

.venv/bin/python scripts/63_analyze_stateprobe_validation.py \
  --candidates \
    results/cem_preselection_dino_push_stateprobe_candidates.csv.gz \
    results/cem_preselection_dino_pick_stateprobe_candidates.csv.gz \
    results/cem_preselection_jepa_push_stateprobe_candidates.csv.gz \
    results/cem_preselection_jepa_pick_stateprobe_candidates.csv.gz \
  --object-probes \
    DINO=checkpoints/spatial_object_probe_dino_wm_metaworld_offpolicy.pt \
    JEPA=checkpoints/spatial_object_probe_jepa_wm_metaworld_offpolicy.pt \
  --hand-probes \
    DINO=checkpoints/ee_probe_dino_wm_metaworld_offpolicy.pt \
    JEPA=checkpoints/ee_probe_jepa_wm_metaworld_offpolicy.pt \
  --n-bootstrap 5000 \
  --out-prefix results/stateprobe_cem_validation
