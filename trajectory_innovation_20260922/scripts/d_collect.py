"""Gate D collection: follow P0, branch all 8 siblings per decision, keep each sibling's segment and actions.

docs/GATE_D_CODEC_PROTOCOL.md. No continuations (not needed for D). The stored chunks also serve
as world-model training data for gate E.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ti_wm.contract import candidate_seed, require_compute
from ti_wm.cta_runtime import KEEP, run_segment
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, done, physical_state, reset_branch

K = 8


def collect(runner, roots, cloner, max_decisions=None):
    states = [reset_branch(r) for r in roots]
    out = {k: [] for k in ("root", "decision", "t", "ctx", "ctx_prev", "ctx_pos", "chunk", "seg", "end", "end_pos",
                           "end_prev_pos", "cov8", "done8", "phys8")}
    d = 0
    while max_decisions is None or d < max_decisions:
        active = [i for i, s in enumerate(states) if not done(s)]
        if not active:
            break
        chunks = runner.draw([states[i].hist for i in active for _ in range(K)],
                             [candidate_seed(roots[i], d, k) for i in active for k in range(K)])
        results = [run_segment(states[i], chunks[j * K + k], cloner) for j, i in enumerate(active) for k in range(K)]
        for j, i in enumerate(active):
            s, group = states[i], results[j * K:(j + 1) * K]
            sibs = [g[0] for g in group]
            out["root"].append(roots[i])
            out["decision"].append(d)
            out["t"].append(s.t)
            out["ctx"].append(s.hist[-1]["pixels"])
            out["ctx_prev"].append(s.hist[-2]["pixels"])
            out["ctx_pos"].append(np.stack([s.hist[-1]["agent_pos"], s.hist[-2]["agent_pos"]]))
            out["chunk"].append(chunks[j * K:(j + 1) * K])
            out["seg"].append(np.stack([g[1] for g in group]))
            out["end"].append(np.stack([b.hist[-1]["pixels"] for b in sibs]))
            out["end_pos"].append(np.stack([b.hist[-1]["agent_pos"] for b in sibs]))
            out["end_prev_pos"].append(np.stack([b.hist[-2]["agent_pos"] for b in sibs]))
            out["cov8"].append([b.coverage for b in sibs])
            out["done8"].append([b.success for b in sibs])
            out["phys8"].append(np.stack([physical_state(b.env) for b in sibs]))
            states[i] = sibs[0]
        d += 1
        print(f"decision {d}: {len(active)} active", flush=True)
    outcomes = [{"root": r, "success": s.success, "steps": s.t} for r, s in zip(roots, states)]
    return {k: np.asarray(v) for k, v in out.items()}, outcomes


def main(out, prep, smoke, first, count, device="cuda", max_decisions=None):
    require_compute()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    smoke_report = json.loads((smoke / "smoke.json").read_text())
    assert smoke_report["status"] == "SMOKE_PASS"
    cloner = CLONERS[smoke_report["clone_method"]]
    runner = PolicyRunner(prep / "checkpoint", device)
    t0 = time.perf_counter()
    data, outcomes = collect(runner, list(range(first, first + count)), cloner, max_decisions)
    np.savez_compressed(out / f"shard_{first}_{first + count - 1}.npz", **data)
    report = {"roots": [first, first + count - 1], "decisions": int(len(data["root"])),
              "p0_success": sum(o["success"] for o in outcomes), "seconds": time.perf_counter() - t0,
              "outcomes": outcomes, "clone_method": smoke_report["clone_method"], "keep_steps": list(KEEP),
              "device": device, "max_decisions": max_decisions}
    (out / f"shard_{first}_{first + count - 1}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "outcomes"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("out", "prep", "smoke"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-decisions", type=int, default=None, help="smoke only: stop after this many decisions")
    a = parser.parse_args()
    main(a.out, a.prep, a.smoke, a.first, a.count, a.device, a.max_decisions)
