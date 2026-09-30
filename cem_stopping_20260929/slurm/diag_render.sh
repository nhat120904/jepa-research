#!/usr/bin/env bash
#SBATCH --job-name=cemstop_diagrender
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/cem_stopping/logs/%x_%j.out
# Dataset frames versus our renders of the same start/goal states (pixels + latent).
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/cem_stopping_20260929/slurm/env.sh
OUT="$RUN_ROOT/diag_render_${SLURM_JOB_ID}"
record_source "$OUT"
"$PY" "$PROJECT/scripts/diag_render.py" --tree-dir "$RUN_ROOT/dev_v1" --out-dir "$OUT"
