"""CTA training data, round 4 (docs/CTA_ONPOLICY_DATA_PROTOCOL_20260927.md).

Like scripts/d_collect.py, many roots advance in lockstep. Per decision two banks share the context:
  bank 0: the policy's 8 seeded candidates (identical to the deployment bank);
  bank 1: the same 8 chunks, each shifted by one constant offset of scale SIGMAS[k] (training data only).
Execution: even roots follow candidate 0 (P0); odd roots execute the geometry-best standard candidate on a seeded
half of the decisions, so the data also covers states a planner reaches. Privileged simulator state is used only to
choose the executed branch and to compute labels; learned deployment never reads it.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ti_wm.contract import candidate_seed, require_compute, select_candidate
from ti_wm.cta_batch import SIGMAS, block_pose, oracle_step, perturb
from ti_wm.cta_geometry import registration_score, world_vertices
from ti_wm.cta_runtime import KEEP, keep_steps
from ti_wm.pusht_runtime import CLONERS, MAX_STEPS, Branch, PolicyRunner, done, physical_state, reset_branch

K, BANKS = 8, 2
PX = 512. / 96.


def run_segment_states(branch, actions, cloner, keep=KEEP):
    """ti_wm.cta_runtime.run_segment, plus the physical state and contact count after every executed step
    (padded with the last state / zero contacts when the segment stops early). For path-dependent queries later."""
    out = Branch(cloner(branch.env), list(branch.hist), branch.t, branch.success, branch.coverage, branch.max_coverage)
    frames, states, contacts = {}, [], []
    for step, action in enumerate(actions, start=1):
        if out.success or out.t >= MAX_STEPS:
            break
        obs, _, terminated, _, info = out.env.step(np.asarray(action, dtype=np.float32))
        out.t += 1
        out.hist = [out.hist[-1], obs]
        out.coverage = float(info["coverage"])
        out.max_coverage = max(out.max_coverage, out.coverage)
        out.success = bool(terminated)
        states.append(physical_state(out.env))
        contacts.append(int(out.env.n_contact_points))
        if step in keep:
            frames[step] = obs["pixels"]
    last = out.hist[-1]["pixels"]
    executed = len(states)
    fill = states[-1] if states else physical_state(out.env)
    states += [fill] * (len(actions) - executed)
    contacts += [0] * (len(actions) - executed)
    return out, np.stack([frames.get(s, last) for s in keep]), np.stack(states), np.asarray(contacts), executed


def spread_stats(pose, end_pos):
    """Per-bank sibling spread: max corresponding-vertex displacement of the block, mean agent end distance."""
    verts = world_vertices(pose)                                                 # (K, 8, 2)
    block = np.linalg.norm(verts[:, None] - verts[None], axis=-1).max(-1).max()
    agent = np.linalg.norm(end_pos[:, None] - end_pos[None], axis=-1)[np.triu_indices(K, 1)].mean()
    return float(block), float(agent)


def collect(runner, roots, cloner, max_decisions=None, keep=KEEP):
    states = [reset_branch(r) for r in roots]
    dec = {k: [] for k in ("root", "decision", "t", "ctx", "ctx_prev", "ctx_pos", "executed", "oracle_step")}
    bank = {k: [] for k in ("chunk", "seg", "end", "end_pos", "end_prev_pos", "cov8", "done8", "phys8",
                            "step_states", "step_contacts", "steps_executed")}
    stats = {"block_px": [[], []], "agent": [[], []]}
    d = 0
    while max_decisions is None or d < max_decisions:
        active = [i for i, s in enumerate(states) if not done(s)]
        if not active:
            break
        flat = np.asarray(runner.draw([states[i].hist for i in active for _ in range(K)],
                                      [candidate_seed(roots[i], d, k) for i in active for k in range(K)]))
        chunks = flat.reshape(len(active), K, *flat.shape[1:]).astype(np.float32)
        both = np.stack([chunks, np.stack([perturb(chunks[j], roots[i], d) for j, i in enumerate(active)])], 1)
        sims = [run_segment_states(states[i], both[j, b, k], cloner, keep)
                for j, i in enumerate(active) for b in range(BANKS) for k in range(K)]
        for j, i in enumerate(active):
            s, group = states[i], sims[j * BANKS * K:(j + 1) * BANKS * K]
            sibs = [g[0] for g in group]
            poses = np.stack([block_pose(b) for b in sibs]).reshape(BANKS, K, 3)
            end_pos = np.stack([b.hist[-1]["agent_pos"] for b in sibs]).reshape(BANKS, K, 2)
            use_oracle = oracle_step(roots[i], d)
            executed = select_candidate([float(x) for x in registration_score(poses[0])]) if use_oracle else 0
            dec["root"].append(roots[i])
            dec["decision"].append(d)
            dec["t"].append(s.t)
            dec["ctx"].append(s.hist[-1]["pixels"])
            dec["ctx_prev"].append(s.hist[-2]["pixels"])
            dec["ctx_pos"].append(np.stack([s.hist[-1]["agent_pos"], s.hist[-2]["agent_pos"]]))
            dec["executed"].append(executed)
            dec["oracle_step"].append(use_oracle)
            bank["chunk"].append(both[j])
            bank["seg"].append(np.stack([g[1] for g in group]).reshape(BANKS, K, *group[0][1].shape))
            bank["end"].append(np.stack([b.hist[-1]["pixels"] for b in sibs]).reshape(BANKS, K, *sibs[0].hist[-1]["pixels"].shape))
            bank["end_pos"].append(end_pos)
            bank["end_prev_pos"].append(np.stack([b.hist[-2]["agent_pos"] for b in sibs]).reshape(BANKS, K, 2))
            bank["cov8"].append(np.array([b.coverage for b in sibs]).reshape(BANKS, K))
            bank["done8"].append(np.array([b.success for b in sibs]).reshape(BANKS, K))
            bank["phys8"].append(np.stack([physical_state(b.env) for b in sibs]).reshape(BANKS, K, -1))
            bank["step_states"].append(np.stack([g[2] for g in group]).astype(np.float32).reshape(BANKS, K, *group[0][2].shape))
            bank["step_contacts"].append(np.stack([g[3] for g in group]).astype(np.int16).reshape(BANKS, K, -1))
            bank["steps_executed"].append(np.array([g[4] for g in group], np.int8).reshape(BANKS, K))
            for b in range(BANKS):
                block, agent = spread_stats(poses[b], end_pos[b])
                stats["block_px"][b].append(block / PX)
                stats["agent"][b].append(agent)
            states[i] = sibs[executed]            # always a standard-bank branch
        d += 1
        print(f"decision {d}: {len(active)} active", flush=True)
    outcomes = [{"root": r, "success": bool(s.success), "steps": int(s.t), "mixed": r % 2 == 1}
                for r, s in zip(roots, states)]
    data = {**{k: np.asarray(v) for k, v in dec.items()}, **{k: np.asarray(v) for k, v in bank.items()}}
    summary = {f"bank{b}": {"block_ge_1px_share": float(np.mean(np.asarray(stats["block_px"][b]) >= 1)),
                            "block_px_median": float(np.median(stats["block_px"][b])),
                            "agent_mean_pairwise_median": float(np.median(stats["agent"][b]))} for b in range(BANKS)}
    return data, outcomes, summary


def main(a):
    require_compute()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    smoke = json.loads((a.smoke / "smoke.json").read_text())
    assert smoke["status"] == "SMOKE_PASS"
    if getattr(a, "proposal_mode", "batched") == "canonical":
        from ti_wm.pusht_canonical import CanonicalPolicyRunner
        runner = CanonicalPolicyRunner(a.prep / "checkpoint", "cuda", n_exec=a.n_exec)
    else:
        runner = PolicyRunner(a.prep / "checkpoint", "cuda", n_exec=a.n_exec)
    n_exec = runner.end - runner.start
    keep = keep_steps(n_exec)
    t0 = time.perf_counter()
    data, outcomes, summary = collect(runner, list(range(a.first, a.first + a.count)), CLONERS[smoke["clone_method"]],
                                      a.max_decisions, keep)
    name = f"shard_{a.first}_{a.first + a.count - 1}"
    np.savez_compressed(a.out / f"{name}.npz", **data)
    report = {"roots": [a.first, a.first + a.count - 1], "decisions": int(len(data["root"])),
              "executed_success": sum(o["success"] for o in outcomes if not o["mixed"]),
              "executed_success_mixed": sum(o["success"] for o in outcomes if o["mixed"]),
              "oracle_step_share": float(np.mean(data["oracle_step"])) if len(data["root"]) else None,
              "spread": summary, "sigmas": list(SIGMAS), "keep_steps": list(keep), "n_exec": n_exec,
              "seconds": time.perf_counter() - t0, "outcomes": outcomes, "clone_method": smoke["clone_method"],
              "max_decisions": a.max_decisions, "proposal_mode": getattr(a, "proposal_mode", "batched"),
              "proposal_microbatch": getattr(runner, "proposal_microbatch", None)}
    (a.out / f"{name}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "outcomes"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("out", "prep", "smoke"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    p.add_argument("--max-decisions", type=int, default=None, help="smoke only")
    p.add_argument("--n-exec", type=int, default=None,
                   help="executed actions per decision (default: the policy's native 8); docs/CTA_REPLAN_INTERVAL_PROTOCOL.md")
    p.add_argument("--proposal-mode", choices=("batched", "canonical"), default="batched",
                   help="canonical: fixed proposal shapes independent of active roots and candidate count")
    main(p.parse_args())
