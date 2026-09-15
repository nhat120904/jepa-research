#!/usr/bin/env bash
#SBATCH --job-name=lscope_scrub_comp
#SBATCH --partition=main
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/scrub_composition_audit_%j.out

# CPU only: download all Scrub demos, verify videos, label by state playback with the native
# monitor, then measure segment composition for native count and coverage-area targets.
set -euo pipefail

PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
STAGE_ROOT=/mnt/data/nhatnc129/jepa/latent_scope_stage_a
VENV_PATH="$STAGE_ROOT/venv"
ROBOCASA_SRC="$STAGE_ROOT/src/robocasa365"
ROBOSUITE_SRC="$STAGE_ROOT/src/robosuite"
OUT=/mnt/data/nhatnc129/jepa/latent_scope_baseline/outputs/scrub_composition_audit_$SLURM_JOB_ID

if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo "must run under sbatch" >&2; exit 1; fi
mkdir -p "$OUT/code" /mnt/data/nhatnc129/jepa/latent_scope_data/hf_mirror
cp "$PROJECT_ROOT/latent_scope_20260909/scripts/analyze_scrub_composition.py" \
   "$PROJECT_ROOT/latent_scope_20260909/scripts/check_scrub_demo_data.py" \
   "$PROJECT_ROOT/latent_scope_20260909/configs/scrub_composition_audit.json" "$OUT/code/"
sha256sum "$OUT"/code/* > "$OUT/code/SHA256SUMS"

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$ROBOCASA_SRC:$ROBOSUITE_SRC:${PYTHONPATH:-}"

"$VENV_PATH/bin/python" "$OUT/code/analyze_scrub_composition.py" \
    --config "$OUT/code/scrub_composition_audit.json" \
    --output-dir "$OUT"
