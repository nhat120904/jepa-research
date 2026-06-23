#!/usr/bin/env bash
#SBATCH --job-name=jepa_extra
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/extra_%j.out
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export BOTO_CONFIG=/dev/null
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
echo "HOST $(hostname) GPU=$CUDA_VISIBLE_DEVICES $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

CONFIGS="diagnostic_robocasa diagnostic_franka_custom diagnostic_pusht diagnostic_point_maze diagnostic_wall"
for name in $CONFIGS; do
  CFG=configs/$name.yaml
  echo "###################### DATASET $name ######################"; date
  MODELS=$($PY -c "import yaml;print(' '.join(yaml.safe_load(open('$CFG'))['models']))")
  echo "models: $MODELS"
  for M in $MODELS; do
    echo "----- 03 extract [$name / $M] -----"; date
    CAI_JEPA_ONLY_MODEL=$M $PY scripts/03_extract_latents.py --config $CFG \
      && echo "OK extract $M" || echo "SKIP extract $M (load/data failure)"
  done
  echo "----- 04 classify [$name] -----"; date
  $PY scripts/04_classify_regimes.py --config $CFG && echo "OK 04 $name" || echo "SKIP 04 $name"
  echo "----- 05 diagnostic [$name] -----"; date
  $PY scripts/05_run_diagnostic.py --config $CFG && echo "OK 05 $name" || echo "SKIP 05 $name"
  echo "====== DONE $name ======"; date
done
echo "===== EXTRA_DATASETS_PIPELINE_DONE ====="; date
