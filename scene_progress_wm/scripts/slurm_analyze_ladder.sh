#!/usr/bin/env bash
#SBATCH --job-name=spwm_ana
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/spwm_ana_%j.out
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/scene_progress_wm/scripts/_common.sh
RUN_ID=${RUN_ID:-ladder_20260904}
GATE_RUN=${GATE_RUN:-gate_screened_20260904}
EXPECTED_SHARDS=${EXPECTED_SHARDS:?submit with EXPECTED_SHARDS}
sha256sum "$PROJECT/scripts/analyze_scene_ladder.py" "$PROJECT/scripts/slurm_analyze_ladder.sh"
"$PY" "$PROJECT/scripts/analyze_scene_ladder.py" \
  --results-root "$PROJECT/outputs/ladder/eval/$RUN_ID" \
  --gate "$PROJECT/outputs/harness_gate/diagnostic/$GATE_RUN/harness_gate.json" \
  --out-dir "$PROJECT/outputs/ladder/aggregate/$RUN_ID" \
  --expected-shards "$EXPECTED_SHARDS"
echo "DONE $(date -u +%FT%TZ)"
