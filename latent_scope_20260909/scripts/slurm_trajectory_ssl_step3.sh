#!/usr/bin/env bash
#SBATCH --job-name=lscope_ssl_s3
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/latent_scope_baseline/logs/trajectory_ssl_step3_%j.out

set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then echo "must run under sbatch" >&2; exit 1; fi
MODE="${MODE:-profile}"
if [[ "$MODE" != "profile" && "$MODE" != "full" ]]; then echo "MODE must be profile or full" >&2; exit 2; fi
PROJECT_ROOT=/home/nhatnc129/nhat.nc/jepa-research
PY=/mnt/data/nhatnc129/jepa/latent_scope_baseline/policy_venv/bin/python
RUN=/mnt/data/nhatnc129/jepa/latent_scope_trajectory_ssl/step3_${MODE}_${SLURM_JOB_ID}
mkdir -p "$RUN/code"
cp -r "$PROJECT_ROOT/latent_scope_20260909/trajectory_ssl" "$RUN/code/"
cp "$PROJECT_ROOT/latent_scope_20260909/configs/trajectory_ssl_step3.json" "$RUN/code/"
cp "$PROJECT_ROOT/latent_scope_20260909/docs/TRAJECTORY_SSL_STEP3_PROTOCOL.md" "$RUN/code/"
cp "$PROJECT_ROOT/latent_scope_20260909/scripts/slurm_trajectory_ssl_step3.sh" "$RUN/code/"
sha256sum "$RUN/code/trajectory_ssl/"*.py "$RUN/code/trajectory_ssl_step3.json" \
  "$RUN/code/TRAJECTORY_SSL_STEP3_PROTOCOL.md" "$RUN/code/slurm_trajectory_ssl_step3.sh" > "$RUN/code/SHA256SUMS"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONPATH="$RUN/code"
FLAG=""
if [[ "$MODE" == "profile" ]]; then FLAG="--profile"; fi
cd "$RUN/code"
"$PY" -m trajectory_ssl.pilot --config "$RUN/code/trajectory_ssl_step3.json" --run-dir "$RUN" $FLAG
