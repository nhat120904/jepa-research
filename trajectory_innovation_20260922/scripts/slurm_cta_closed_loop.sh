#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_cl
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:45:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_cl_%A_%a.out
# Usage: CTA_TAG=r0 sbatch [--array=0-9] slurm_cta_closed_loop.sh <cta_train_dir> <collect_dir> <first_root> <roots_per_task> [arms] [skip-done run dirs...]
# ~3.6 min per root with 5 arms (54668), so keep roots_per_task <= 25 under the 1:45 limit.
# Dev roots 2100-2299 only while debugging; sealed roots 3000-3399 only for the locked model (protocol §2).
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
TRAIN=${1:?train run}; COLLECT=${2:?collect run}; FIRST=${3:?first root}; COUNT=${4:?roots per task}
ARMS=${5:-P0,FULL8,CTA8,DIRECT8}
SKIP=("${@:6}")
PROJECT=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
RUN="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_cl_${SLURM_ARRAY_JOB_ID}"
CODE="$RUN/code_${SLURM_ARRAY_TASK_ID}"
[[ ! -e "$CODE" ]] || exit 2
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS_${SLURM_ARRAY_TASK_ID}"
unit_tests
START=$((FIRST + SLURM_ARRAY_TASK_ID * COUNT))
"$PY" "$CODE/scripts/cta_closed_loop.py" --run "$RUN" --prep "$PREP" --smoke "$SMOKE" --train-run "$TRAIN" \
  --dev-shard "$COLLECT/shard_2000_2049.npz" --first "$START" --count "$COUNT" --arms "$ARMS" --skip-done "${SKIP[@]}"
