#!/usr/bin/env bash
#SBATCH --job-name=cg_stats
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/config_generalization/logs/%x_%j.out
# Download OGBench visual puzzle / cube-triple play data (HF mirror) and measure held-out configuration novelty.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/config_generalization_20261001/slurm/env.sh
ENVS=${ENVS:-"visual-puzzle-3x3-play-v0 visual-puzzle-4x4-play-v0 visual-puzzle-4x5-play-v0 visual-puzzle-4x6-play-v0 visual-cube-triple-play-v0"}
"$OGB_PY" - <<PY
import hashlib, json, os
from huggingface_hub import hf_hub_download
man = {}
for env in "$ENVS".split():
    for name in (f"{env}.npz", f"{env}-val.npz"):
        p = hf_hub_download("ryanhoangt/ogbench_data", name, repo_type="dataset", local_dir="$DATA")
        man[name] = {"bytes": os.path.getsize(p)}
        print(name, man[name], flush=True)
json.dump(man, open("$RUN_ROOT/data_manifest.json", "w"), indent=1)
PY
OUT=$RUN_ROOT/stats_${SLURM_JOB_ID}
record_source "$OUT"
"$OGB_PY" "$PROJECT/scripts/config_stats.py" --data "$DATA" --envs $ENVS --out "$OUT/config_stats.json"
