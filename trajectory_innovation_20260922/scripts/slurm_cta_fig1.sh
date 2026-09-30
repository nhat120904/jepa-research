#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_fig1
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_fig1_%j.out
# Paper Fig. 1 assets (scripts/cta_fig1_data.py): real PushT decisions, native renders, CTA codes and scores.
# Usage: sbatch slurm_cta_fig1.sh <source tree> [roots...]
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
PROJECT=${1:?source tree}
shift
ROOTS=("${@:-}")
[[ -n "${ROOTS[0]}" ]] || ROOTS=(2100 2101 2102 2103 2104 2105 2106 2107)
TI=/mnt/data/nhatnc129/jepa/trajectory_innovation
RUN="$TI/cta_fig1_$SLURM_JOB_ID"
CODE="$RUN/code"
[[ ! -e "$CODE" ]] || exit 2
export CTA_TAG=fig1 WANDB_MODE=disabled
source "$PROJECT/scripts/cta_env.sh"
snapshot "$RUN/CODE_SHA256SUMS"
"$PY" "$CODE/scripts/cta_fig1_data.py" --run "$RUN" --parent "$ROOT/cta_geometry_e2e_55018/train" \
  --r4 "$ROOT/cta_plus_train_55149/train" --prep "$PREP" --smoke "$SMOKE" --roots "${ROOTS[@]}"
echo FIG1_DONE
