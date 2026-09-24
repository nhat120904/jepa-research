"""Gate S1 collection: follow P0, branch all 8 siblings per decision, continue each 2 chunks.

docs/SIBLING_READER_OFFLINE_PROTOCOL.md. One shard = 50 roots, batched across episodes.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ti_wm.contract import candidate_seed, require_compute, sibling_seed
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, done, physical_state, reset_branch, run_prefix

K = 8
CONT_CHUNKS = 2


def collect(runner, roots, cloner):
    states = [reset_branch(r) for r in roots]
    out = {k: [] for k in ("root", "decision", "t", "ctx", "ctx_pos", "end", "end_prev", "end_pos", "end_prev_pos",
                           "cov8", "done8", "phys8", "goal16", "cov24")}
    d = 0
    while True:
        active = [i for i, s in enumerate(states) if not done(s)]
        if not active:
            break
        chunks = runner.draw([states[i].hist for i in active for _ in range(K)],
                             [candidate_seed(roots[i], d, k) for i in active for k in range(K)])
        sibs = [run_prefix(states[i], chunks[j * K + k], cloner) for j, i in enumerate(active) for k in range(K)]
        conts = list(sibs)
        for c in range(CONT_CHUNKS):
            live = [n for n, b in enumerate(conts) if not done(b)]
            if live:
                cch = runner.draw([conts[n].hist for n in live],
                                  [sibling_seed(roots[active[n // K]], d, n % K, c) for n in live])
                for m, n in enumerate(live):
                    conts[n] = run_prefix(conts[n], cch[m], cloner)
        for j, i in enumerate(active):
            s, group, cgroup = states[i], sibs[j * K:(j + 1) * K], conts[j * K:(j + 1) * K]
            out["root"].append(roots[i])
            out["decision"].append(d)
            out["t"].append(s.t)
            out["ctx"].append(s.hist[-1]["pixels"])
            out["ctx_pos"].append(np.stack([s.hist[-1]["agent_pos"], s.hist[-2]["agent_pos"]]))
            out["end"].append(np.stack([b.hist[-1]["pixels"] for b in group]))
            out["end_prev"].append(np.stack([b.hist[-2]["pixels"] for b in group]))
            out["end_pos"].append(np.stack([b.hist[-1]["agent_pos"] for b in group]))
            out["end_prev_pos"].append(np.stack([b.hist[-2]["agent_pos"] for b in group]))
            out["cov8"].append([b.coverage for b in group])
            out["done8"].append([b.success for b in group])
            out["phys8"].append(np.stack([physical_state(b.env) for b in group]))
            out["goal16"].append(np.stack([b.hist[-1]["pixels"] for b in cgroup]))
            out["cov24"].append([b.coverage for b in cgroup])
            states[i] = group[0]
        d += 1
        print(f"decision {d}: {len(active)} active", flush=True)
    outcomes = [{"root": r, "success": s.success, "steps": s.t} for r, s in zip(roots, states)]
    return {k: np.asarray(v) for k, v in out.items()}, outcomes


def main(out, prep, smoke, first, count):
    require_compute()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    smoke_report = json.loads((smoke / "smoke.json").read_text())
    assert smoke_report["status"] == "SMOKE_PASS"
    cloner = CLONERS[smoke_report["clone_method"]]
    runner = PolicyRunner(prep / "checkpoint")
    t0 = time.perf_counter()
    data, outcomes = collect(runner, list(range(first, first + count)), cloner)
    np.savez_compressed(out / f"shard_{first}_{first + count - 1}.npz", **data)
    report = {"roots": [first, first + count - 1], "decisions": int(len(data["root"])),
              "p0_success": sum(o["success"] for o in outcomes), "seconds": time.perf_counter() - t0,
              "outcomes": outcomes, "clone_method": smoke_report["clone_method"]}
    (out / f"shard_{first}_{first + count - 1}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "outcomes"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("out", "prep", "smoke"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    a = parser.parse_args()
    main(a.out, a.prep, a.smoke, a.first, a.count)
