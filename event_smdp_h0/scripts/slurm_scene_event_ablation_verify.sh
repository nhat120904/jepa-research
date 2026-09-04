#!/usr/bin/env bash
#SBATCH --job-name=scene_event_ablation_verify
#SBATCH --partition=mig
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/scene_event_ablation_verify_%j.out
set -euo pipefail
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/event_smdp_h0"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
cd "$REPO"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} $(date -u +%FT%TZ)"
sha256sum "$PROJECT/scripts/verify_scene_event_ablation_independent.py"
"$RUNTIME/.venv/bin/python" \
  "$PROJECT/scripts/verify_scene_event_ablation_independent.py" \
  --ablation-root "$PROJECT/outputs/scene_event_ablation/eval/ablation_20260904" \
  --factorial-root "$PROJECT/outputs/scene_event_history/eval/confirmatory_20260904" \
  --out "$PROJECT/outputs/scene_event_ablation/aggregate/ablation_20260904/independent_verification.json"
