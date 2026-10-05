#!/usr/bin/env bash
#SBATCH --job-name=ew_fetch
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Download OGBench visual-scene play data (HF mirror) and cache it as memory-mappable .npy (third task family:
# generality test of the unified method; cube + drawer + window + two lock buttons).
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENV=${ENV:-visual-scene-play-v0}
"$OGB_PY" - <<PY
from huggingface_hub import hf_hub_download
import numpy as np
for name in ("$ENV.npz", "$ENV-val.npz"):
    p = hf_hub_download("ryanhoangt/ogbench_data", name, repo_type="dataset", local_dir="$DATA")
    z = np.load(p, mmap_mode="r")
    print(name, {k: z[k].shape for k in z.files}, flush=True)
PY
"$OGB_PY" "$PROJECT/scripts/cache_data.py" --data "$DATA" --env "$ENV" --train-episodes "${TEP:-1500}" --out "$RUN_ROOT/cache"
ls -la "$RUN_ROOT/cache/$ENV"
