#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_collect
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-17
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_collect_%A_%a.out
# CTA data (docs/CTA_E2E_PROTOCOL.md): follow P0, branch all 8 seeded candidates per decision, keep segments + chunks.
# Tasks 0-15: training roots 30250-31049; tasks 16-17: dev roots 2000-2099. Usage: sbatch slurm_cta_collect.sh
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
RUN="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_collect_${SLURM_ARRAY_JOB_ID}"
CODE="$RUN/code_${SLURM_ARRAY_TASK_ID}"
[[ ! -e "$CODE" ]] || exit 2
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS_${SLURM_ARRAY_TASK_ID}"
unit_tests
T=$SLURM_ARRAY_TASK_ID
if (( T < 16 )); then START=$((30250 + T * 50)); else START=$((2000 + (T - 16) * 50)); fi
"$PY" "$CODE/scripts/d_collect.py" --out "$RUN" --prep "$PREP" --smoke "$SMOKE" --first "$START" --count 50
