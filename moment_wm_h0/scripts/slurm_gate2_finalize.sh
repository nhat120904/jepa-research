#!/usr/bin/env bash
#SBATCH --job-name=mwm_g2_final
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/mwm_g2_final_%j.out
set -euo pipefail

REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/moment_wm_h0"
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID:-NA} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/finalize_anchor_certificate.py" \
          "$PROJECT/scripts/slurm_gate2_finalize.sh"
"$PY" "$PROJECT/scripts/finalize_anchor_certificate.py" \
  --dataset "$PROJECT/outputs/gate2_full/glides.pt" \
  --features "$PROJECT/outputs/gate2_full/anchors.pt" \
  --results-dir "$PROJECT/outputs/gate2_full/certificates" \
  --parent-executed-sha256 8b779f73b9a945cdb2fd3cf788985ad6aa7cfb92f7cfed7764d3df339594051a \
  --bootstrap 10000
