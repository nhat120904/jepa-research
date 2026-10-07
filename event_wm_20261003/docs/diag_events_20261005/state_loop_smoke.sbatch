#!/usr/bin/env bash
#SBATCH --job-name=ew_stsmoke
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/data/nhatnc129/jepa/event_wm/logs/%x_%j.out
# Smoke test of the u_closed_loop.py state path (code path only, untrained smoke models; numbers meaningless):
# one episode of cube-triple-v0 task 1 on CPU.
set -euo pipefail
source /home/nhatnc129/nhat.nc/jepa-research/event_wm_20261003/slurm/env.sh
D=$RUN_ROOT/state_loop_smoke
"$TORCH_PY" "$PROJECT/scripts/u_closed_loop.py" --env cube-triple-v0 --model "$D/u_model.pt" --state-layout "$D/layout.json" \
  --skill "$D/u_skill.pt" --events "$D/events" --cache "$D/cache/cube-triple-play-v0" --episodes 1 --tasks 1 --workers 1 \
  --timeout 60 --device cpu --out "$D/loop"
"$TORCH_PY" - <<PY
import json
d = json.load(open("$D/loop/u_closed_loop.json"))
ep = d["episodes"][0]
print({k: ep[k] for k in ("task", "first_plan", "steps", "replans", "timeouts", "success")})
print("sim_start", ep["sim_start"], "sim_goal", ep["sim_goal"])
print("read_start", ep["read_start"])
print("read_goal", ep["read_goal"])
print("events", ep["events"][:2])
PY
