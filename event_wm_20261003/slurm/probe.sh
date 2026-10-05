#!/usr/bin/env bash
#SBATCH --job-name=ew_probe
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Unified-method probe: agent-free scene image + scene events with ONE rule set on every task family.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
OUT=$RUN_ROOT/probe_${SLURM_JOB_ID}
record_source "$OUT/src"
for W in ${WS:-15}; do
  "$TORCH_PY" "$PROJECT/scripts/agentfree_probe.py" --cache "$RUN_ROOT/cache/visual-puzzle-4x5-play-v0" --W "$W" --out "$OUT"
  "$TORCH_PY" "$PROJECT/scripts/agentfree_probe.py" --cache "$RUN_ROOT/cache/visual-cube-triple-play-v0" --W "$W" \
    --ref-events "$RUN_ROOT/cubeD_57116/events" --out "$OUT"
done
"$TORCH_PY" "$PROJECT/scripts/agentfree_probe.py" --cache "$RUN_ROOT/cache/visual-puzzle-4x6-play-v0" --W 15 --out "$OUT"
