"""Gate C2-confirm: closed-loop P0 / PROG8_r0 / PROG8_r1 on fresh roots (docs/GATE_C2_CONFIRM_PROTOCOL.md)."""

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

ARMS = ("P0", "PROG8_r0", "PROG8_r1")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_scorer(trained, visual, goal_frames):
    assert json.loads((trained / "train_report.json").read_text())["status"] == "TRAINED"
    reader = ProgressReader().to(visual.device)
    reader.load_state_dict(torch.load(trained / "reader.pt")["state_dict"])
    return ProgressScorer(reader, visual, goal_frames)


def main(run, prep, smoke, r0, r1, first, count):
    require_compute()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    smoke_report = json.loads((smoke / "smoke.json").read_text())
    assert smoke_report["status"] == "SMOKE_PASS"
    cloner = CLONERS[smoke_report["clone_method"]]
    runner = PolicyRunner(prep / "checkpoint")
    visual = VisualScorer()
    goal_frames = np.load(smoke / "goal_frames.npz")["frames"]
    scorers = {"PROG8_r0": load_scorer(r0, visual, goal_frames), "PROG8_r1": load_scorer(r1, visual, goal_frames)}
    hashes = {"r0": sha256(r0 / "reader.pt"), "r1": sha256(r1 / "reader.pt")}
    output = run / f"roots_{first}_{first + count - 1}.jsonl"
    with open(output, "w") as stream:
        for root in range(first, first + count):
            t0 = time.perf_counter()
            record = {"root": root, "clone_method": smoke_report["clone_method"], "reader_sha256": hashes}
            record["P0"] = run_arm(root, "P0", runner, None, cloner, p0_branches=1, bank_size=1)
            for arm, prog in scorers.items():
                record[arm] = run_arm(root, "PROG8", runner, None, cloner, prog=prog, bank_size=K_SELECT)
            record["seconds"] = time.perf_counter() - t0
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(root, {a: record[a]["success"] for a in ARMS}, f"{record['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run", "prep", "smoke", "r0", "r1"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    a = parser.parse_args()
    main(a.run, a.prep, a.smoke, a.r0, a.r1, a.first, a.count)
