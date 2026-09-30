#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_task_premise
#SBATCH --partition=main
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_task_premise_%j.out
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
RELEASE=${1:?immutable release directory}
OUT="/mnt/data/nhatnc129/jepa/trajectory_innovation/cta_task_premise_$SLURM_JOB_ID"
mkdir "$OUT"
cp -r "$RELEASE" "$OUT/code"
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3 "$OUT/code/cta_task_premise_audit.py" \
    --config "$OUT/code/cta_task_premise_audit.json" \
    --native-source "$OUT/code/rinse_bowls.py" --out "$OUT/report.json"
