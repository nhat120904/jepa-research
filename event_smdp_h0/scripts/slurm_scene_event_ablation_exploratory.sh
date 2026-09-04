#!/usr/bin/env bash
#SBATCH --job-name=scene_event_ablation_expl
#SBATCH --partition=mig
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_event_ablation_expl_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
RUN_ID=${RUN_ID:?submit with RUN_ID}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} RUN=$RUN_ID $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/contrast_scene_event_ablation_exploratory.py"
"$RUNTIME/.venv/bin/python" \
  "$PROJECT/scripts/contrast_scene_event_ablation_exploratory.py" \
  --eval-root "$PROJECT/outputs/scene_event_ablation/eval/${RUN_ID}" \
  --out-dir "$PROJECT/outputs/scene_event_ablation/aggregate/${RUN_ID}"
