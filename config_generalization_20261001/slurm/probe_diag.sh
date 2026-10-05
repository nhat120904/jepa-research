#!/usr/bin/env bash
#SBATCH --job-name=cg_probe
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=00:35:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/config_generalization/logs/%x_%j.out
# Where is button information lost? Probe pixels / patch tokens / CLS / latent of a trained WM.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/config_generalization_20261001/slurm/env.sh
export PYTHONPATH="/home/nhatnc129/nhat.nc/jepa-research/diagnosis/external/stable-worldmodel:/home/nhatnc129/nhat.nc/jepa-research/diagnosis/external/stable-pretraining:$PYTHONPATH"
OUT=$RUN_ROOT/probe_${SLURM_JOB_ID}_${ENV:?ENV}
record_source "$OUT"
"$TORCH_PY" "$PROJECT/scripts/probe_diag.py" --data "$DATA" --env "$ENV" --wm "${WM:?WM}" --out "$OUT/probe.json"
