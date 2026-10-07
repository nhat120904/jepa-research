#!/usr/bin/env bash
#SBATCH --job-name=ew_fetchst
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Download the official OGBench STATE play datasets (HF mirror ryanhoangt/ogbench_data) for the state track: the
# same tasks as the visual ones, compared with published state-based baselines on the same data.
#   ENVS="cube-triple-play-v0 puzzle-4x5-play-v0" sbatch slurm/fetch_state.sh
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
ENVS=${ENVS:-"cube-triple-play-v0 puzzle-4x5-play-v0"}
for ENV in $ENVS; do
"$OGB_PY" - <<PY
from huggingface_hub import hf_hub_download
import numpy as np
for name in ("$ENV.npz", "$ENV-val.npz"):
    p = hf_hub_download("ryanhoangt/ogbench_data", name, repo_type="dataset", local_dir="$DATA")
    z = np.load(p)
    print(name, {k: (z[k].shape, str(z[k].dtype)) for k in z.files}, "episodes", int(z["terminals"].sum()), flush=True)
PY
done
ls -la "$DATA" | grep -v visual
