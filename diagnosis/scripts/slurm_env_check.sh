#!/usr/bin/env bash
#SBATCH --job-name=jepa_envchk
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/envchk_%j.out
set -euo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
echo "HOST: $(hostname)"; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
PY=.venv/bin/python
$PY - <<'PYEOF'
import torch
print("torch", torch.__version__, "cuda_avail", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.randn(2048,2048,device="cuda"); print("matmul ok", (x@x).sum().item() != 0)
# upstream + diagnosis imports
import sys
try:
    from data.loaders import add_upstream_to_path
    add_upstream_to_path()
    print("add_upstream_to_path ok")
except Exception as e:
    print("UPSTREAM PATH ERR:", repr(e))
for m in ["metrics","stratification"]:
    try:
        __import__(m); print("import", m, "ok")
    except Exception as e:
        print("import", m, "ERR", repr(e))
PYEOF
echo "ENVCHK_DONE"
