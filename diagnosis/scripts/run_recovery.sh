#!/usr/bin/env bash
# Full Metaworld diagnostic pipeline, restart-safe.
#
# Run detached so it survives the harness killing foreground/background jobs
# between turns:
#   setsid bash scripts/run_recovery.sh > logs/recovery.log 2>&1 < /dev/null &
#
# Each numbered step is cache-aware (03 skips encoded models, 04 rewrites the
# regime sidecar, 05/06 re-read), so re-running after a crash resumes cheaply.
# Touches results/.pipeline_done on success.
set -euo pipefail

cd "$(dirname "$0")/.."   # -> diagnosis/

# CUDA fragmentation guard (helps the K=16-negatives CRA batches) + HDF5 lock
# disable (NFS / repeated-open friendliness).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HDF5_USE_FILE_LOCKING=FALSE

# Use the project venv if present (upstream ships .venv; our deps install into it).
if [[ -f external/jepa-wms/.venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source external/jepa-wms/.venv/bin/activate
fi

CONFIG="${1:-configs/diagnostic_metaworld.yaml}"
mkdir -p logs results
rm -f results/.pipeline_done

echo "[run_recovery] config=$CONFIG  $(date -Is)"
python scripts/03_extract_latents.py  --config "$CONFIG"
python scripts/04_classify_regimes.py --config "$CONFIG"
python scripts/05_run_diagnostic.py   --config "$CONFIG"
python scripts/06_analyze_results.py  --metaworld_csv results/metaworld_diagnostic.csv

touch results/.pipeline_done
echo "[run_recovery] DONE  $(date -Is)"
