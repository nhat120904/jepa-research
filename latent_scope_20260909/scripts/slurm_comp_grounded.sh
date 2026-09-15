#!/usr/bin/env bash
#SBATCH --job-name=lscope_grounded
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/comp_grounded_%j.out

# GPU: grounded shared-teacher prototype (docs/COMP_GROUNDED_PROTOCOL.md).
# MODE=profile|full via --export; the full run is submitted with an explicit --time override.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo "must run under sbatch" >&2; exit 1; fi
MODE="${MODE:-profile}"
PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
PY=/mnt/data/nhatnc129/jepa/latent_scope_baseline/policy_venv/bin/python
RUN=/mnt/data/nhatnc129/jepa/latent_scope_comp_pilot/runs/grounded_${MODE}_$SLURM_JOB_ID
mkdir -p "$RUN/code"
cp -r "$PROJECT_ROOT/latent_scope_20260909/comp_pilot" "$RUN/code/"
cp "$PROJECT_ROOT/latent_scope_20260909/configs/comp_pilot.json" "$RUN/code/"
find "$RUN/code" -type f \( -name '*.py' -o -name '*.json' \) | sort | xargs sha256sum > "$RUN/code/SHA256SUMS"
FLAG=""
if [[ "$MODE" == "profile" ]]; then FLAG="--profile"; fi
cd "$RUN/code"
"$PY" -m comp_pilot.grounded --config "$RUN/code/comp_pilot.json" --run-dir "$RUN" $FLAG
