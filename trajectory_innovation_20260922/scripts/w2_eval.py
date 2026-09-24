"""Week 2: closed-loop P0 / PHYS8 / RANK8 with the frozen S1-b reader (docs/W2_CLOSED_LOOP_PROTOCOL.md)."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from ti_wm.arms import K_SELECT, run_arm
from ti_wm.contract import require_compute
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, VisualScorer
from ti_wm.sibling import SiblingScorer

ARMS = ("P0", "PHYS8", "RANK8")
PREFLIGHT_DECISIONS, PREFLIGHT_REL_TOL, PREFLIGHT_ARGMAX = 32, 0.10, 0.90


class _Stored:
    def __init__(self, frame, pos, prev_pos):
        self.hist = [{"pixels": frame, "agent_pos": prev_pos}, {"pixels": frame, "agent_pos": pos}]


def preflight(rank, collect, s1):
    """The closed-loop scorer must reproduce the S1 held-out scores it was qualified on."""
    d = np.load(collect / "shard_1500_1549.npz")
    saved = np.load(s1 / "heldout_scores_s1b.npy")
    now = []
    for n in range(PREFLIGHT_DECISIONS):
        state = _Stored(d["ctx"][n], d["ctx_pos"][n][0], d["ctx_pos"][n][1])
        branches = [_Stored(d["end"][n, k], d["end_pos"][n, k], d["end_prev_pos"][n, k]) for k in range(K_SELECT)]
        now.append(rank.scores(state, branches))
    now, ref = np.array(now), saved[:PREFLIGHT_DECISIONS]
    rel = float(np.abs(now - ref).max() / max(ref.std(), 1e-6))
    argmax = float(np.mean(now.argmax(1) == ref.argmax(1)))
    if rel > PREFLIGHT_REL_TOL or argmax < PREFLIGHT_ARGMAX:
        raise RuntimeError(f"closed-loop scorer disagrees with S1 held-out scores: rel {rel:.3f}, argmax {argmax:.2f}")
    return {"rel_max_diff": rel, "argmax_agreement": argmax}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(run, prep, smoke, s1, collect, first, count):
    require_compute()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    smoke_report = json.loads((smoke / "smoke.json").read_text())
    assert smoke_report["status"] == "SMOKE_PASS"
    cloner = CLONERS[smoke_report["clone_method"]]
    runner = PolicyRunner(prep / "checkpoint")
    visual = VisualScorer()
    rank = SiblingScorer(s1 / "reader_s1b.pt", visual, np.load(smoke / "goal_frames.npz")["frames"])
    worst = preflight(rank, collect, s1)
    print(f"preflight vs S1 held-out: {worst}", flush=True)
    reader_hash = sha256(s1 / "reader_s1b.pt")
    output = run / f"roots_{first}_{first + count - 1}.jsonl"
    with open(output, "w") as stream:
        for root in range(first, first + count):
            t0 = time.perf_counter()
            record = {"root": root, "clone_method": smoke_report["clone_method"], "reader_sha256": reader_hash,
                      "preflight": worst}
            record["P0"] = run_arm(root, "P0", runner, None, cloner, p0_branches=1, bank_size=1)
            record["PHYS8"] = run_arm(root, "PHYS8", runner, None, cloner, bank_size=K_SELECT)
            record["RANK8"] = run_arm(root, "RANK8", runner, None, cloner, bank_size=K_SELECT, rank=rank)
            record["seconds"] = time.perf_counter() - t0
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            print(root, {a: record[a]["success"] for a in ARMS}, f"{record['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run", "prep", "smoke", "s1", "collect"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    a = parser.parse_args()
    main(a.run, a.prep, a.smoke, a.s1, a.collect, a.first, a.count)
