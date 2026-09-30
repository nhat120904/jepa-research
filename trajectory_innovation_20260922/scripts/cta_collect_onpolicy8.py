"""Collect all policy candidates on states reached by CTA4, without action perturbations.

The selected action advances the episode; the other seven branches are training
targets only. Roots here must be distinct from control-evaluation roots.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cta_diag_onpolicy import R4_NETS, load_nets
from cta_collect_plus import run_segment_states
from ti_wm.contract import candidate_seed, require_compute, select_candidate
from ti_wm.cta_batch import BatchScorer, K, block_pose
from ti_wm.cta_geometry import registration_score
from ti_wm.cta_runtime import Planner
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, VisualScorer, done, physical_state, reset_branch


def collect(runner, cloner, scorer, roots, max_decisions=None):
    states = [reset_branch(r) for r in roots]
    names = ("root", "decision", "t", "ctx", "ctx_prev", "ctx_pos", "chunk", "seg", "end",
             "end_pos", "end_prev_pos", "cov8", "native_cov8", "done8", "phys8", "chosen")
    rows = {k: [] for k in names}
    d = 0
    while max_decisions is None or d < max_decisions:
        active = [i for i, s in enumerate(states) if not done(s)]
        if not active:
            break
        flat = np.asarray(runner.draw([states[i].hist for i in active for _ in range(K)],
                                      [candidate_seed(roots[i], d, k) for i in active for k in range(K)]))
        chunks = flat.reshape(len(active), K, *flat.shape[1:]).astype(np.float32)
        sims = [run_segment_states(states[i], chunks[j, k], cloner)
                for j, i in enumerate(active) for k in range(K)]
        branches, segs = [x[0] for x in sims], [x[1] for x in sims]
        scores = scorer.scores([states[i] for i in active], chunks, branches, segs, ["CTA4"])["CTA4"]
        for j, i in enumerate(active):
            s = states[i]
            sibs = branches[j * K:(j + 1) * K]
            group = sims[j * K:(j + 1) * K]
            chosen = select_candidate([float(x) for x in scores[j]])
            poses = np.stack([block_pose(b) for b in sibs])
            end_pos = np.stack([b.hist[-1]["agent_pos"] for b in sibs])
            row = {"root": roots[i], "decision": d, "t": s.t, "ctx": s.hist[-1]["pixels"],
                   "ctx_prev": s.hist[-2]["pixels"],
                   "ctx_pos": np.stack([s.hist[-1]["agent_pos"], s.hist[-2]["agent_pos"]]),
                   "chunk": chunks[j], "seg": np.stack([g[1] for g in group]),
                   "end": np.stack([b.hist[-1]["pixels"] for b in sibs]),
                   "end_pos": end_pos,
                   "end_prev_pos": np.stack([b.hist[-2]["agent_pos"] for b in sibs]),
                   "cov8": registration_score(poses),
                   "native_cov8": np.array([b.coverage for b in sibs]),
                   "done8": np.array([b.success for b in sibs]),
                   "phys8": np.stack([physical_state(b.env) for b in sibs]), "chosen": chosen}
            for key in names:
                rows[key].append(row[key])
            states[i] = sibs[chosen]
            s.env.close()
            for k, b in enumerate(sibs):
                if k != chosen:
                    b.env.close()
        d += 1
        print(f"decision {d}: {len(active)} active", flush=True)
    outcomes = [{"root": r, "success": bool(s.success), "steps": int(s.t)} for r, s in zip(roots, states)]
    for s in states:
        s.env.close()
    return {k: np.asarray(v) for k, v in rows.items()}, outcomes


def main(a):
    require_compute()
    if a.count < 1 or a.first <= 2299 and a.first + a.count - 1 >= 2000:
        raise ValueError("Collection must use train roots, disjoint from evaluation roots 2000-2299")
    torch.manual_seed(0)
    smoke = json.loads((a.smoke / "smoke.json").read_text())
    if smoke["status"] != "SMOKE_PASS":
        raise ValueError("Simulator clone contract failed")
    device = torch.device("cuda")
    visual = VisualScorer("cuda")
    goals = np.load(a.smoke / "goal_frames.npz")["frames"]
    planner = Planner(a.parent / "cta.pt", visual, goals)
    extra, _, _ = load_nets(a.r4 / "round4.pt", {"task": R4_NETS["task"]}, device)
    scorer = BatchScorer(planner, extra)
    runner = PolicyRunner(a.prep / "checkpoint", "cuda")
    data, outcomes = collect(runner, CLONERS[smoke["clone_method"]],
                             scorer, list(range(a.first, a.first + a.count)), a.max_decisions)
    a.out.mkdir(parents=True, exist_ok=True)
    name = f"shard_{a.first}_{a.first + a.count - 1}"
    np.savez_compressed(a.out / f"{name}.npz", **data)
    (a.out / f"{name}.json").write_text(json.dumps({"roots": [a.first, a.first + a.count - 1],
                                                    "decisions": len(data["root"]), "outcomes": outcomes,
                                                    "mean_override": float(np.mean(data["chosen"] != 0)),
                                                    "model": str(a.r4), "max_decisions": a.max_decisions}, indent=2))
    print(f"COLLECT_OK {name}: {len(data['root'])} decisions", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("out", "prep", "smoke", "parent", "r4"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--max-decisions", type=int)
    main(p.parse_args())
