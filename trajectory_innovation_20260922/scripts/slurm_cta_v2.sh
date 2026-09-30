#!/usr/bin/env bash
#SBATCH --job-name=ti_cta_v2
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/trajectory_innovation/logs/cta_v2_%x_%j.out
# CTA v2 on PushT (scripts/cta_train_v2.py). Modes:
#   encode  full-token features of the Round-4 banks (scripts/cta_encode_r4_full.py) -> $TI/cta_v2_r4feat_<job>
#   smoke   unit tests; encode 1 Round-4 shard; v2 training on a small slice (code path only, not a result)
#   train   full v2 training + offline ladder; needs R4FEAT=<encode dir>; SEED (default 0); EXTRA_ARGS
set -euo pipefail
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use sbatch' >&2; exit 1; }
MODE=${1:?mode}
PY=/mnt/data/nhatnc129/jepa/lewm_stage0/.venv/bin/python
PROJ=/home/nhatnc129/nhat.nc/jepa-research/trajectory_innovation_20260922
TI=/mnt/data/nhatnc129/jepa/trajectory_innovation
PCA=$TI/cta_feat_54490/pca.pt
COLLECT=$TI/cta_plus_collect_55147
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch PYTHONUNBUFFERED=1
echo "HOST=$(hostname) JOB=$SLURM_JOB_ID MODE=$MODE $(date -u +%FT%TZ)"
OUT="$TI/cta_v2_${MODE}_${SLURM_JOB_ID}"
mkdir -p "$OUT/code"
cp "$PROJ"/scripts/cta_train_v2.py "$PROJ"/scripts/cta_encode_r4_full.py "$PROJ"/scripts/cta_encode.py "$OUT/code"/
cp "$PROJ"/ti_wm/cta.py "$PROJ"/ti_wm/cta_parallel.py "$PROJ"/ti_wm/cta_eval.py "$PROJ"/ti_wm/cta_geometry.py "$OUT/code"/
sha256sum "$OUT"/code/* > "$OUT/code/SHA256SUMS"
cd "$PROJ/scripts"
case "$MODE" in
  encode)
    "$PY" cta_encode_r4_full.py --collect "$COLLECT" --pca "$PCA" --out "$OUT"
    ;;
  smoke)
    (cd "$PROJ" && "$PY" -m unittest discover -s cta_tests -p "test_cta_ogb_v2.py")
    "$PY" cta_encode_r4_full.py --collect "$COLLECT" --pca "$PCA" --out "$OUT/r4" --max-shards 1
    "$PY" cta_train_v2.py --run "$OUT/train" --r4 "$OUT/r4" --with-perturbed --limit 300 --steps1 40 --steps2 30 \
      --eval-every 20 --select-limit 40 --final-limit 40
    ;;
  train)
    "$PY" cta_train_v2.py --run "$OUT/train_s${SEED:-0}" --r4 "${R4FEAT:?R4FEAT}" --seed "${SEED:-0}" ${EXTRA_ARGS:-}
    ;;
  *) echo "unknown mode $MODE" >&2; exit 2 ;;
esac
echo "DONE $(date -u +%FT%TZ)"
