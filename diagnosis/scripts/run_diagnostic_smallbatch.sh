#!/usr/bin/env bash
# Re-run ONLY the diagnostic + analysis (05 + 06) once the latent cache and
# regime sidecars already exist. Use this to retune batch_size / metrics
# without re-encoding (the expensive 03 step).
#
#   setsid bash scripts/run_diagnostic_smallbatch.sh > logs/diagnostic_smallbatch.log 2>&1 < /dev/null &
#
# Touches results/.diagnostic_done on success.
set -euo pipefail

cd "$(dirname "$0")/.."   # -> diagnosis/

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HDF5_USE_FILE_LOCKING=FALSE

if [[ -f external/jepa-wms/.venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source external/jepa-wms/.venv/bin/activate
fi

CONFIG="${1:-configs/diagnostic_metaworld.yaml}"
mkdir -p logs results
rm -f results/.diagnostic_done

echo "[run_diagnostic_smallbatch] config=$CONFIG  $(date -Is)"
python scripts/05_run_diagnostic.py  --config "$CONFIG"
python scripts/06_analyze_results.py --metaworld_csv results/metaworld_diagnostic.csv

touch results/.diagnostic_done
echo "[run_diagnostic_smallbatch] DONE  $(date -Is)"
