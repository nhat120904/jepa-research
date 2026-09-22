#!/usr/bin/env bash
#SBATCH --job-name=lscope_ssl_step2
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/trajectory_ssl_step2_%j.out

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo "must run under sbatch" >&2; exit 1; fi
PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
PY=/mnt/data/nhatnc129/jepa/latent_scope_baseline/policy_venv/bin/python
RUN=/mnt/data/nhatnc129/jepa/latent_scope_trajectory_ssl/step2_${SLURM_JOB_ID}
mkdir -p "$RUN/code"
cp -r "$PROJECT_ROOT/latent_scope_20260909/trajectory_ssl" "$RUN/code/"
cp "$PROJECT_ROOT/latent_scope_20260909/configs/trajectory_ssl_step2.json" "$RUN/code/"
cp "$PROJECT_ROOT/latent_scope_20260909/scripts/slurm_trajectory_ssl_step2.sh" "$RUN/code/"
sha256sum "$RUN/code/trajectory_ssl/"*.py "$RUN/code/trajectory_ssl_step2.json" "$RUN/code/slurm_trajectory_ssl_step2.sh" > "$RUN/code/SHA256SUMS"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$RUN/code"
cd "$RUN/code"
"$PY" -m trajectory_ssl.tests --output "$RUN/tests.json"
"$PY" -m trajectory_ssl.build_targets --config "$RUN/code/trajectory_ssl_step2.json" --output "$RUN"
