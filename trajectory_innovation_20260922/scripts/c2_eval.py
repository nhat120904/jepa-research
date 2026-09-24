"""Gate C2 stage 2: closed-loop P0 / PHYS8 / PROG8 on qualification roots 1200-1299 (one shard)."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ti_wm.arms import K_SELECT, run_arm
from ti_wm.contract import require_compute
from ti_wm.progress import ProgressReader, ProgressScorer
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, VisualScorer


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run, prep, smoke, trained, first, count):
    require_compute()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    smoke_report = json.loads((smoke / "smoke.json").read_text())
    assert smoke_report["status"] == "SMOKE_PASS"
    assert sha256(smoke / "goal_map.pt") == smoke_report["goal_map_sha256"]
    train_report = json.loads((trained / "train_report.json").read_text())
    assert train_report["status"] == "TRAINED", train_report["status"]
    cloner = CLONERS[smoke_report["clone_method"]]
    runner = PolicyRunner(prep / "checkpoint")
    visual = VisualScorer()
    visual.set_goal(torch.load(smoke / "goal_map.pt"))
    reader = ProgressReader().to(visual.device)
    reader.load_state_dict(torch.load(trained / "reader.pt")["state_dict"])
    goal_frames = np.load(smoke / "goal_frames.npz")["frames"]
    prog = ProgressScorer(reader, visual, goal_frames)
    output = run / f"roots_{first}_{first + count - 1}.jsonl"
    with open(output, "w") as stream:
        for root in range(first, first + count):
            t0 = time.perf_counter()
            record = {"root": root, "clone_method": smoke_report["clone_method"],
                      "reader_sha256": sha256(trained / "reader.pt")}
            record["P0"] = run_arm(root, "P0", runner, visual, cloner, prog=prog, p0_branches=K_SELECT, diag=True)
            record["PHYS8"] = run_arm(root, "PHYS8", runner, None, cloner, prog=prog)
            record["PROG8"] = run_arm(root, "PROG8", runner, None, cloner, prog=prog)
            record["seconds"] = time.perf_counter() - t0
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(root, {a: record[a]["success"] for a in ("P0", "PHYS8", "PROG8")},
                  f"{record['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run", "prep", "smoke", "trained"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    a = parser.parse_args()
    main(a.run, a.prep, a.smoke, a.trained, a.first, a.count)
