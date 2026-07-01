#!/usr/bin/env bash
#SBATCH --job-name=jepa_p4b
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseB_adversarial_%j.out
# Track-1 Phase B (docs/plans/2026-07-01-adversarial-cost-and-repr-design.md): a
# DAgger-style loop that hardens the object/ee probes against the frames CEM
# actually EXPLOITS (Phase A), not just a random off-policy sample (Phase-3 3b).
# Each round: (1) mine CEM-exploited elites with the CURRENT probes
# (scripts/35 --save-buffer); (2) retrain both probes mixing that buffer in
# (scripts/22/19 --extra-buffer); (3) re-gate (scripts/30 --cost stateprobe) with
# the newly-hardened probes. Repeats P4B_ROUNDS times, each round starting from the
# PREVIOUS round's hardened checkpoint (compounding hardening).
#   push climbs toward 16/16 over rounds, reach stays intact -> Track 1 wins ->
#       carry the final round's probes into scripts/18 closed loop.
#   push plateaus near 1-2/16 -> pockets are inherent to the frozen z geometry, not
#       fixable by re-supervising a readout -> Track 2 (slurm_phaseC_repr.sh) is
#       the answer, not this.
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
M=${P4_MODEL:-dino_wm_metaworld}
ROUNDS=${P4B_ROUNDS:-2}
EPISODES=${P4B_EPISODES:-16}

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date)  rounds=$ROUNDS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
set -e

OBJ=checkpoints/spatial_object_probe_${M}_offpolicy.pt
EEP=checkpoints/ee_probe_${M}_offpolicy.pt

for R in $(seq 1 "$ROUNDS"); do
  echo "### Phase B round $R/$ROUNDS — mine with current probes ###"
  BUF="results/cem_exploit_buffer_round${R}.pt"
  $PY scripts/35_cem_exploit_precision.py --config "$CFG" --model "$M" \
      --probe "$OBJ" --ee-probe "$EEP" --tasks mw-push mw-pick-place \
      --episodes "$EPISODES" --strict-success \
      --out "results/cem_exploit_precision_round${R}.csv" --save-buffer "$BUF"

  echo "### Phase B round $R/$ROUNDS — harden object probe ###"
  $PY scripts/22_train_spatial_probe.py --config "$CFG" --model "$M" \
      --offpolicy-frac 0.5 --extra-buffer "$BUF" --extra-frac 0.3 \
      --out results/representation_precision_spatial_round${R}.csv
  echo "### Phase B round $R/$ROUNDS — harden ee probe ###"
  $PY scripts/19_train_ee_probe.py --config "$CFG" --model "$M" \
      --offpolicy-frac 0.5 --extra-buffer "$BUF" --extra-frac 0.3

  # scripts/22/19 write *_offpolicy_adv.pt (both --offpolicy-frac>0 and
  # --extra-buffer set); point the next round + the re-gate at that checkpoint.
  OBJ="checkpoints/spatial_object_probe_${M}_offpolicy_adv.pt"
  EEP="checkpoints/ee_probe_${M}_offpolicy_adv.pt"
  cp "$OBJ" "checkpoints/spatial_object_probe_${M}_offpolicy_adv_round${R}.pt"
  cp "$EEP" "checkpoints/ee_probe_${M}_offpolicy_adv_round${R}.pt"

  echo "### Phase B round $R/$ROUNDS — re-gate stateprobe ###"
  $PY scripts/30_latent_oracle.py --config "$CFG" --model "$M" \
      --cost stateprobe --probe "$OBJ" --ee-probe "$EEP" \
      --tasks mw-push mw-pick-place --episodes "$EPISODES" --strict-success \
      --out "results/latent_oracle_stateprobe_phaseB_round${R}.csv"
  echo "--- round $R re-gate (cf. baseline stateprobe push 2/16, robust-probe 1/16) ---"
  cat "results/latent_oracle_stateprobe_phaseB_round${R}.csv"
done

echo "===== PHASEB_ADVERSARIAL_DONE ====="; date
