#!/usr/bin/env bash
#SBATCH --job-name=handoff_a2
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/data/nhatnc129/jepa_runs/logs/handoff_a2_%j.out
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: sbatch $0 /absolute/path/to/best.pt" >&2
  exit 2
fi
CHECKPOINT=$1
REPO=/home/nhatnc129/nhat.nc/jepa-research
PROJECT="$REPO/skill_handoff_wm"
RUNTIME=/mnt/data/nhatnc129/jepa/lewm_stage0
DATA=/mnt/data/vhoangth2/datasets/ogbench_data
OUT="$PROJECT/outputs/a2_screen/${SLURM_JOB_ID}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

cd "$REPO"
mkdir -p "$OUT"
echo "HOST=$(hostname) JOB=${SLURM_JOB_ID} $(date -u +%FT%TZ)"
sha256sum \
  "$PROJECT/maze.py" \
  "$PROJECT/policy.py" \
  "$PROJECT/sim.py" \
  "$PROJECT/switching.py" \
  "$PROJECT/termination.py" \
  "$PROJECT/scripts/eval_a1.py" \
  "$PROJECT/scripts/train_termination.py" \
  "$PROJECT/scripts/analyze_a1.py" \
  "$PROJECT/scripts/slurm_a2_pipeline.sh" \
  "$CHECKPOINT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Collection split: 50 oracle episodes, disjoint from both A1 and A2 evaluation seeds.
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/eval_a1.py" \
  --dataset-name antmaze-medium-navigate-v0 --dataset-dir "$DATA" \
  --checkpoint "$CHECKPOINT" --out-dir "$OUT/collection" \
  --task-ids 1,2,3,4,5 \
  --seeds 94001,94002,94003,94004,94005,94006,94007,94008,94009,94010 \
  --arms oracle --durations 10,20,40,80 --probe-steps 40 --primitive-budget 1000

"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/train_termination.py" \
  --episodes "$OUT/collection/episodes.jsonl" \
  --out-dir "$OUT/termination" --seed 0 --steps 10000

# Held-out A2 screen: same task/seed pairs for every deployable baseline and the oracle diagnostic.
"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/eval_a1.py" \
  --dataset-name antmaze-medium-navigate-v0 --dataset-dir "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --termination-checkpoint "$OUT/termination/termination.pt" \
  --out-dir "$OUT/eval" \
  --task-ids 1,2,3,4,5 \
  --seeds 95001,95002,95003,95004,95005,95006,95007,95008,95009,95010,95011,95012,95013,95014,95015,95016,95017,95018,95019,95020 \
  --arms fixed_10,fixed_20,fixed_40,fixed_80,heuristic,learned,oracle \
  --durations 10,20,40,80 --probe-steps 40 --primitive-budget 1000

"$RUNTIME/.venv/bin/python" "$PROJECT/scripts/analyze_a1.py" \
  --episodes "$OUT/eval/episodes.jsonl" --out "$OUT/gate.json" --gate-points 10
