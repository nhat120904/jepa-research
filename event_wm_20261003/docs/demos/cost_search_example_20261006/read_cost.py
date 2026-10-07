"""Read saved demo states through its frozen cost-to-go checkpoint; no training or simulator."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from u_wm import Model

if "SLURM_JOB_ID" not in os.environ:
    raise RuntimeError("Use sbatch")
torch.set_num_threads(1)
root = Path(os.environ["EXAMPLE_DIR"])
trace = json.loads((root / "plan_task2.json").read_text())
model_path = Path("/mnt/data/nhatnc129/jepa/event_wm/uident_57366_cf/visual-cube-triple-play-v0/self1000/wm/u_model.pt")
expected = "7eb88a9dd077195f809c98afaefa505d90364faf584d26a03f9e707df3f4bcd8"
assert hashlib.sha256(model_path.read_bytes()).hexdigest() == expected
model = Model(torch.load(model_path, map_location="cpu", weights_only=False), "cpu")
states = np.array(trace["states"], np.float32)
states[:, :, 5] = states[:, :, 5] > 0.5
goal = np.array(trace["goal"], np.float32)
goal[:, 5] = goal[:, 5] > 0.5
hs = model.heuristic(states, goal)
rows = [dict(depth=i, h=float(h), priority_0_6g_plus_h=0.6 * i + float(h),
             goal_test=bool(model.at_goal(s, goal)), state=s.tolist())
        for i, (s, h) in enumerate(zip(states, hs))]
result = dict(job_id=os.environ["SLURM_JOB_ID"], checkpoint_sha256=expected,
              source_trace="report_20261005/job57580/plan_task2.json",
              note="CPU recomputation on saved observed/predicted states, not original logged search priorities; final goal successors return before heuristic evaluation in search.",
              rows=rows, goal=goal.tolist(), commands=trace["commands"],
              search_record=dict(expanded=trace["expanded"], full_plan_found=trace["full_plan_found"]))
output = root / ("result_" + os.environ["SLURM_JOB_ID"] + ".json")
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
