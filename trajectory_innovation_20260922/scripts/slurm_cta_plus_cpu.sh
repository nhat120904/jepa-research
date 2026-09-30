#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_plus_cpu
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_plus_%x_%j.out
# CPU-only modes of slurm_cta_plus.sh (diag_agg, r4_agg, r5_agg). Same arguments: <immutable release> <mode> [ids].
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
case "${2:?mode}" in diag_agg|r4_agg|r5_agg|r6_agg|r5pen_agg) ;; *) echo "GPU mode $2 must use slurm_cta_plus.sh" >&2; exit 2 ;; esac
exec bash "${1:?immutable release}/scripts/slurm_cta_plus.sh" "$@"
