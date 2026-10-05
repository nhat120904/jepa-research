#!/usr/bin/env bash
#SBATCH --job-name=ew_rgoals
#SBATCH --partition=mig
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Overfitting check (user request 2026-10-04): the frozen models and arguments of the official-protocol runs
# (4x5: 57025, 99/100; 4x6: 100/100) on N fresh random start/goal problems instead of the 5 fixed official tasks.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
N=${N:-100}; SEED=${SEED:-3}
OUT=$RUN_ROOT/random_goals_${SLURM_JOB_ID}
record_source "$OUT/src"
"$TORCH_PY" "$PROJECT/scripts/closed_loop.py" --env visual-puzzle-4x5-play-v0 \
  --planner "$RUN_ROOT/round2_56980/planner/planner.pt" --skill "$RUN_ROOT/skill3_56999/tau20_rel3/skill.pt" \
  --reader "$RUN_ROOT/reader_57020_visual-puzzle-4x5-play-v0/reader.pt" --low skill --high learned \
  --random-goals "$N" --seed "$SEED" --workers 14 --out "$OUT/4x5"
"$TORCH_PY" "$PROJECT/scripts/closed_loop.py" --env visual-puzzle-4x6-play-v0 \
  --planner "$RUN_ROOT/size_4x6/planner/planner.pt" --skill "$RUN_ROOT/size_4x6/skill3/skill.pt" \
  --reader "$RUN_ROOT/size_4x6/reader/reader.pt" --low skill --high learned --max-expansions 200000 \
  --random-goals "$N" --seed "$SEED" --workers 14 --out "$OUT/4x6"
