#!/usr/bin/env bash
#SBATCH --job-name=lscope_reeval
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/comp_pilot_reeval_%j.out

# GPU: re-evaluate frozen 52634 checkpoints with the corrected readout + raw-input probes.
# MODE=profile|full via --export; the full run is submitted with an explicit --time override.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo "must run under sbatch" >&2; exit 1; fi
MODE="${MODE:-profile}"
PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
PY=/mnt/data/nhatnc129/jepa/latent_scope_baseline/policy_venv/bin/python
SOURCE=/mnt/data/nhatnc129/jepa/latent_scope_comp_pilot/runs/full_52634
OUT=/mnt/data/nhatnc129/jepa/latent_scope_comp_pilot/runs/reeval_${MODE}_$SLURM_JOB_ID
mkdir -p "$OUT/code"
cp -r "$PROJECT_ROOT/latent_scope_20260909/comp_pilot" "$OUT/code/"
cp "$PROJECT_ROOT/latent_scope_20260909/configs/comp_pilot.json" "$OUT/code/"
find "$OUT/code" -type f \( -name '*.py' -o -name '*.json' \) | sort | xargs sha256sum > "$OUT/code/SHA256SUMS"
FLAG=""
if [[ "$MODE" == "profile" ]]; then FLAG="--profile"; fi
cd "$OUT/code"
"$PY" -m comp_pilot.reeval --config "$OUT/code/comp_pilot.json" --source-run "$SOURCE" --out-dir "$OUT" $FLAG
