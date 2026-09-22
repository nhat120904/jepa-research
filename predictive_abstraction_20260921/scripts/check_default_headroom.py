"""Read-only replay diagnostic against the immutable 53529 code and candidate bank."""
import json
import os
from pathlib import Path

if not os.environ.get("SLURM_JOB_ID"):
    raise RuntimeError("Use sbatch on a compute node")

import numpy as np
import torch
from pa_wm.preflight import physical_answers, sample_task
from pa_wm.proposals import visual_goal_bank
from pa_wm.wall_adapter import WallRGB

torch.set_num_threads(1)
source = Path("/mnt/data/nhatnc129/jepa/predictive_abstraction/preflight_53529")
config = json.loads((source / "code/configs/headroom_v2.json").read_text())
rows = [json.loads(line) for line in (source / "prefixes.jsonl").read_text().splitlines()]
output = Path(os.environ["PA_REVIEW_OUTPUT"])
output.mkdir(exist_ok=False)
results = []
for row in rows:
    env = WallRGB(source / "upstream", *row["layout"], seed=config["seed"])
    start, a, b = sample_task(env, np.random.default_rng(row["seed"]))
    np.testing.assert_array_equal(start, row["start"])
    np.testing.assert_array_equal(a, row["goal_a"])
    np.testing.assert_array_equal(b, row["goal_b"])
    image = env.reset(start)
    snapshot = env.snapshot()
    bank = visual_goal_bank(np.random.default_rng(row["seed"] + 1), image,
                            env.goal_image(a), env.goal_image(b), config["candidates"],
                            config["horizon"], config["knots"], config["action_max_norm"])
    if row["prefix_id"] < config["save_example_banks"]:
        with np.load(source / f"bank_{row['prefix_id']:03d}_observations.npz") as saved:
            np.testing.assert_array_equal(bank, saved["actions"])
    _, states = env.rollout(snapshot, bank[1])
    default = physical_answers(states[1:], a, b, config["eval_radius"])["a_then_b"]
    selected_index = row["tasks"]["a_then_b"]["image_selected_index"]
    _, states = env.rollout(snapshot, bank[selected_index])
    selected = physical_answers(states[1:], a, b, config["eval_radius"])["a_then_b"]
    assert selected == row["tasks"]["a_then_b"]["image_selected_success"]
    oracle = row["tasks"]["a_then_b"]["oracle_bank_success"]
    assert not default or oracle
    results.append({"prefix_id": row["prefix_id"], "layout": row["layout"],
                    "default_a_then_b": default, "oracle": oracle,
                    "old_selected_reproduced": selected,
                    "rescuable_over_default": oracle and not default})
summary = {
    "job_id": os.environ["SLURM_JOB_ID"], "prefixes": len(results),
    "default_successes": sum(r["default_a_then_b"] for r in results),
    "oracle_successes": sum(r["oracle"] for r in results),
    "rescuable_over_default": sum(r["rescuable_over_default"] for r in results),
    "all_prior_selected_outcomes_reproduced": True,
    "scope": "fixed 48-step bank development bound, not closed-loop success or learned result",
    "rows": results,
}
(output / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))
