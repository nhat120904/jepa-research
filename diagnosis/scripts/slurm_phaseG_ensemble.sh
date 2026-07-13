#!/usr/bin/env bash
#SBATCH --job-name=jepa_p4g
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/phaseG_l%a_%j.out
# Phase G — ENSEMBLE / disagreement-penalised cost (scripts/39).
#
# WHY: Phase 0-F showed every SINGLE post-hoc cost on the frozen (or LoRA-reshaped)
# latent is reward-hacked by CEM — the seed sweep gated push {5,0,2,1,1}/16 (mean 1.8,
# inside frozen noise 0-2). This arm attacks reward-hacking directly: ensemble the 5
# encoder-LoRA+phi seeds we already trained and add a DISAGREEMENT penalty. An exploit
# pocket fools ONE seed's readout while the others read the object far -> the per-seed
# goal distances scatter -> the penalty inflates the cost -> CEM avoids it. Works ONLY
# if the aliasing is seed-specific; if all 5 share the frozen-DINOv2 blind spot the
# disagreement stays flat and nothing changes. Either result is informative.
#
# lambda sweeps the penalty weight. l0 = pure consensus mean (does averaging 5 seeds
# alone help?); l>0 = adds the disagreement guard. Gated on the SAME fixed 16 push
# episodes (seed0=10000) as the seed sweep, so numbers are directly comparable.
#
# Submit as an array over lambda index: sbatch --array=0-3 scripts/slurm_phaseG_ensemble.sh
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
EPISODES=${P4G_EPISODES:-16}

# lambda grid indexed by SLURM_ARRAY_TASK_ID (0..3); override with P4G_LAMBDA.
LAMBDAS=(0.0 0.5 1.0 2.0)
IDX=${SLURM_ARRAY_TASK_ID:-2}
LAM=${P4G_LAMBDA:-${LAMBDAS[$IDX]}}
TAG=$(echo "$LAM" | tr '.' 'p')

# The 5 seeds trained in Phase D/E/F (each encoder-LoRA paired with its own phi head).
LORAS=(
  checkpoints/encoder_lora_${M}.pt
  checkpoints/encoder_lora_${M}_r1.pt
  checkpoints/encoder_lora_${M}_s1.pt
  checkpoints/encoder_lora_${M}_s2.pt
  checkpoints/encoder_lora_${M}_s3.pt
)
PHIS=(
  checkpoints/phi_enclora_${M}.pt
  checkpoints/phi_enclora_${M}_r1.pt
  checkpoints/phi_enclora_${M}_s1.pt
  checkpoints/phi_enclora_${M}_s2.pt
  checkpoints/phi_enclora_${M}_s3.pt
)
OUT=results/latent_oracle_phiens_l${TAG}.csv

echo "HOST $(hostname) GPU=${CUDA_VISIBLE_DEVICES:-NA} $(date) lambda=$LAM K=${#LORAS[@]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null
for f in "${LORAS[@]}" "${PHIS[@]}"; do
  [ -f "$f" ] || { echo "MISSING checkpoint $f — run Phase D/F first. STOP." >&2; exit 1; }
done
set -e

echo "### Phase G ensemble gate: PUSH, lambda=$LAM (cf. seed sweep {5,0,2,1,1}/16) ###"
$PY scripts/39_latent_oracle_ensemble.py --config "$CFG" --model "$M" \
    --encoder-loras "${LORAS[@]}" --phi-adapters "${PHIS[@]}" \
    --lambda-dis "$LAM" --tasks mw-push --episodes "$EPISODES" --strict-success \
    --out "$OUT"

echo "===== PHASEG_L${TAG}_DONE (lambda=$LAM) ====="; date
awk -F, 'NR>1{a++; s+=$4; e+=$5} END{printf "lambda='$LAM' push any=%d/%d held=%d/%d\n",s,a,e,a}' "$OUT"
