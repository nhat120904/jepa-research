#!/usr/bin/env bash
#SBATCH --job-name=ew_basecrl
#SBATCH --partition=mig,main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Two arms of the same OGBench agent on one GPU (small MLPs): raw input (published protocol) and adapter-entity input.
# Usage: AGENT=crl ARGS="--agent.alpha=3.0" sbatch baseline.sbatch   (hyperparameters from impls/hyperparameters.sh)
set -euo pipefail
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40 WANDB_MODE=offline WANDB_SILENT=true
export OGBENCH_IMPLS=/mnt/data/nhatnc129/jepa/ogbench/ogbench/impls
export EVENT_WM_SOURCE=$RUN/source
PY=/mnt/data/nhatnc129/jepa/ogbench/venv/bin/python
# jax 0.4.35 resolves CUDA via nvidia.cuda_nvcc.__file__, which is None for this namespace package; point it explicitly
export CUDA_ROOT=/mnt/data/nhatnc129/jepa/ogbench/venv/lib/python3.11/site-packages/nvidia/cuda_nvcc
OUT=$RUN/baseline_${SLURM_JOB_ID}
mkdir -p "$OUT"
sha256sum "$RUN/baseline/run_baseline.py" "$OGBENCH_IMPLS/main.py" "$OGBENCH_IMPLS/agents/$AGENT.py" "$EVENT_WM_SOURCE/s_entities.py" "$LAYOUT" > "$OUT/SOURCE_SHA256SUMS"
cd "$OGBENCH_IMPLS"
for MODE in raw entity; do
  $PY "$RUN/baseline/run_baseline.py" --env_name=cube-triple-play-v0 --dataset_dir=/mnt/data/nhatnc129/jepa/ogbench/data \
      --agent=agents/$AGENT.py $ARGS --input_mode=$MODE --layout="$LAYOUT" --seed=0 \
      --eval_episodes=50 --video_episodes=0 --eval_interval=100000 --save_interval=1000000 \
      --run_group=${AGENT}_${MODE} --save_dir="$OUT/$MODE" ${EXTRA:-} > "$OUT/$MODE.log" 2>&1 &
done
wait
echo BASELINE_DONE
