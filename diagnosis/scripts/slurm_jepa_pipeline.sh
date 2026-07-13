#!/usr/bin/env bash
#SBATCH --job-name=jepa_pipeline
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/jepa_pipeline_%j.out
# Consolidated jepa_wm_metaworld replication of the oracle-ladder + overopt-sweep
# mechanism (originally 4 separate jobs: 25402/25403/25404/25405). Merged into ONE
# job so it only ever requests 1 GPU slot instead of 4 competing for a QOS cap of
# gres/gpu=2 (the dino push+reach and pick-place sweeps already occupy both slots).
# Stages run sequentially in one job:
#   1. train off-policy-hardened spatial+ee probes for jepa_wm_metaworld
#   2. re-check off-policy precision on the robust probe (sanity, matches Sec 5 protocol)
#   3. oracle-ladder rung: latent L2 (cost=l2, no probe)
#   4. oracle-ladder rung: object-readout (cost=gobj, plain non-offpolicy probe)
#   5. oracle-ladder rung: stateprobe re-gate (cost=stateprobe, robust probes from step 1)
#   6. overopt sweep (Fig.1 mechanism), push+reach, n=16 — the long stage; if this
#      times out, resubmitting this SAME script redoes the (cheap) stages 1-5 and
#      resumes stage 6 from its own per-episode CSV checkpoint (scripts/41 logic).
set -uo pipefail
cd /home/nhatnc129/nhat.nc/jepa-research/diagnosis
set -a; source .env; set +a
export TORCH_HOME=/mnt/data/nhatnc129/jepa/cache/torch
export HF_HOME=/mnt/data/nhatnc129/jepa/cache/hf
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN:-}"
export JEPAWM_OSSCKPT=/mnt/data/nhatnc129/jepa/ossckpt
export HF_HUB_OFFLINE=0
export BOTO_CONFIG=/dev/null
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PATH="$PWD/.venv/bin:$PATH"
PY=.venv/bin/python
CFG=configs/diagnostic_metaworld.yaml
M=jepa_wm_metaworld
FRAC=0.5
EPISODES=16
OBJ=checkpoints/spatial_object_probe_${M}_offpolicy.pt
EEP=checkpoints/ee_probe_${M}_offpolicy.pt
PLAINOBJ=checkpoints/object_probe_${M}.pt

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
set -e

# Skip an oracle-ladder rung that already finished on a previous (timed-out) run.
# scripts/30 writes its CSV in truncate ("w") mode and flushes per row, so a full
# expected row count means the rung completed; a partial file (interrupted mid-rung)
# has fewer rows and is redone. Lets a resubmit go straight to the long stage 6
# instead of burning ~5h re-running the cheap-ish rungs 4/5/5b from scratch.
rung_done () {  # $1 = csv path, $2 = expected data-row count
  [ -f "$1" ] && [ "$(($(wc -l < "$1") - 1))" -ge "$2" ]
}

echo "### 0/6 ensure latent cache exists for $M (skips if already extracted) ###"
CAI_JEPA_ONLY_MODEL=$M $PY scripts/03_extract_latents.py --config "$CFG"

if [ ! -f "$OBJ" ]; then
  echo "### 1/6 train off-policy-robust spatial object probe (jepa) ###"
  $PY scripts/22_train_spatial_probe.py --config "$CFG" --model "$M" \
      --offpolicy-frac "$FRAC" --out results/representation_precision_spatial_offpolicy_jepa.csv
else
  echo "### 1/6 skip (already trained): $OBJ ###"
fi
if [ ! -f "$EEP" ]; then
  echo "### 2/6 train off-policy-robust ee probe (jepa) ###"
  $PY scripts/19_train_ee_probe.py --config "$CFG" --model "$M" --offpolicy-frac "$FRAC"
else
  echo "### 2/6 skip (already trained): $EEP ###"
fi

echo "### 3/6 offpolicy precision re-check (robust probe) ###"
$PY scripts/34_offpolicy_precision.py --config "$CFG" --model "$M" \
    --probe "$OBJ" --target obj --episodes "$EPISODES" \
    --out results/offpolicy_precision_obj_robust_jepa.csv || echo "[warn] step 3 failed, continuing"

# rungs 4/5 sweep 3 tasks × 16 ep = 48 rows; rung 5b sweeps 2 tasks × 16 = 32 rows.
echo "### 4/6 oracle-ladder rung: latent L2 (jepa) ###"
if rung_done results/metaworld_latent_oracle_jepa_l2.csv 48; then
  echo "### 4/6 skip (already complete) ###"
else
  $PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" --cost l2 \
      --tasks mw-reach mw-push mw-pick-place --episodes "$EPISODES" --strict-success \
      --out results/metaworld_latent_oracle_jepa_l2.csv
fi

echo "### 5/6 oracle-ladder rung: object-readout (jepa) ###"
if rung_done results/metaworld_latent_oracle_jepa_gobj.csv 48; then
  echo "### 5/6 skip (already complete) ###"
else
  $PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" --cost gobj \
      --probe "$PLAINOBJ" \
      --tasks mw-reach mw-push mw-pick-place --episodes "$EPISODES" --strict-success \
      --out results/metaworld_latent_oracle_jepa_gobj.csv
fi

echo "### 5b/6 oracle-ladder rung: stateprobe re-gate (jepa, robust probes) ###"
if rung_done results/latent_oracle_stateprobe_robust_jepa.csv 32; then
  echo "### 5b/6 skip (already complete) ###"
else
  $PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" --cost stateprobe \
      --probe "$OBJ" --ee-probe "$EEP" \
      --tasks mw-push mw-pick-place --episodes "$EPISODES" --strict-success \
      --out results/latent_oracle_stateprobe_robust_jepa.csv
fi

echo "### 6/6 overopt sweep (Fig.1 mechanism), push+reach, n=16 (jepa) ###"
$PY scripts/41_overoptimization_sweep.py --config "$CFG" --model "$M" \
    --tasks mw-push mw-reach --costs stateprobe l2 --iters-grid 2 6 12 24 --samples-grid 100 \
    --episodes "$EPISODES" \
    --probe "$OBJ" --ee-probe "$EEP" --strict-success \
    --out-episodes "results/overopt_episodes_mwpush_mwreach_n16_jepa.csv" \
    --out-curves "results/overopt_curves_mwpush_mwreach_n16_jepa.csv"

echo "===== JEPA_PIPELINE_DONE ====="; date
