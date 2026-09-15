#!/usr/bin/env bash
#SBATCH --job-name=lscope_pilot_enc
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/comp_pilot_encode_%j.out

# GPU: frozen DINOv2-S encoding of all Scrub demos (4x4 patch pool, fp16, 20 Hz). Resumable.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo "must run under sbatch" >&2; exit 1; fi

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
PY=/mnt/data/nhatnc129/jepa/latent_scope_baseline/policy_venv/bin/python
ROOT=/mnt/data/nhatnc129/jepa/latent_scope_comp_pilot
mkdir -p "$ROOT/features" "$ROOT/code_encode_$SLURM_JOB_ID"
cp "$PROJECT_ROOT/latent_scope_20260909/scripts/encode_scrub_pilot.py" \
   "$PROJECT_ROOT/latent_scope_20260909/configs/comp_pilot.json" "$ROOT/code_encode_$SLURM_JOB_ID/"
sha256sum "$ROOT/code_encode_$SLURM_JOB_ID"/* > "$ROOT/code_encode_$SLURM_JOB_ID/SHA256SUMS"

export HF_HOME="$ROOT/hf_cache"
export TOKENIZERS_PARALLELISM=false
"$PY" "$ROOT/code_encode_$SLURM_JOB_ID/encode_scrub_pilot.py" --config "$ROOT/code_encode_$SLURM_JOB_ID/comp_pilot.json"
