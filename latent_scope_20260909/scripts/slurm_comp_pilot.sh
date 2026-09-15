#!/usr/bin/env bash
#SBATCH --job-name=lscope_pilot
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/comp_pilot_%j.out

# GPU: offline composition pilot (docs/COMP_PILOT_PROTOCOL.md). MODE=profile|full via --export.
# The walltime above is the profile cap; the full run is submitted with an explicit --time override.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo "must run under sbatch" >&2; exit 1; fi
MODE="${MODE:-profile}"
PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
PY=/mnt/data/nhatnc129/jepa/latent_scope_baseline/policy_venv/bin/python
RUN=/mnt/data/nhatnc129/jepa/latent_scope_comp_pilot/runs/${MODE}_$SLURM_JOB_ID
mkdir -p "$RUN/code"
cp -r "$PROJECT_ROOT/latent_scope_20260909/comp_pilot" "$RUN/code/"
cp "$PROJECT_ROOT/latent_scope_20260909/configs/comp_pilot.json" "$RUN/code/"
find "$RUN/code" -type f -name '*.py' -o -name '*.json' | sort | xargs sha256sum > "$RUN/code/SHA256SUMS"

FLAG=""
if [[ "$MODE" == "profile" ]]; then FLAG="--profile"; fi
cd "$RUN/code"
"$PY" -m comp_pilot.train --config "$RUN/code/comp_pilot.json" --run-dir "$RUN" \
    --arms segment_nocomp,segment_comp,frame_rollout,map_union $FLAG
