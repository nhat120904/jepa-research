"""Gates A-C main run for one shard of qualification roots (docs/GATE_ABC_PROTOCOL.md)."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from ti_wm.arms import run_arm, run_hrep, run_official
from ti_wm.contract import require_compute
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, VisualScorer


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run, prep, smoke, first, count):
    require_compute()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    smoke_report = json.loads((smoke / "smoke.json").read_text())
    assert smoke_report["status"] == "SMOKE_PASS", smoke_report["status"]
    assert sha256(smoke / "goal_map.pt") == smoke_report["goal_map_sha256"], "goal map drift"
    cloner = CLONERS[smoke_report["clone_method"]]
    runner = PolicyRunner(prep / "checkpoint")
    scorer = VisualScorer()
    scorer.set_goal(torch.load(smoke / "goal_map.pt"))
    output = run / f"roots_{first}_{first + count - 1}.jsonl"
    with open(output, "w") as stream:
        for root in range(first, first + count):
            t0 = time.perf_counter()
            record = {"root": root, "clone_method": smoke_report["clone_method"]}
            record["OFFICIAL"] = run_official(root, runner)
            p0 = run_arm(root, "P0", runner, scorer, cloner, keep_states=True)
            record["hrep"] = run_hrep(root, runner, p0.pop("_states"), p0.pop("_banks"), cloner)
            p0.pop("_final")
            record["P0"] = p0
            for arm in ("PHYS8", "VIS8", "MEDOID8"):
                record[arm] = run_arm(root, arm, runner, scorer, cloner)
            record["seconds"] = time.perf_counter() - t0
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(root, {a: record[a]["success"] for a in ("OFFICIAL", "P0", "PHYS8", "VIS8", "MEDOID8")},
                  f"{record['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    a = parser.parse_args()
    main(a.run, a.prep, a.smoke, a.first, a.count)
