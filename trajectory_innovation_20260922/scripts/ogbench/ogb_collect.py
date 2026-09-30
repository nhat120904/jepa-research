"""Branched data for CTA on OGBench (docs/CTA_OGBENCH_PROTOCOL.md, "Data").

At every decision of a train-root episode: the frozen goal-conditioned policy draws its 8 seeded chunks (deployment
seed rule); each chunk is simulated from an exact restore, rendering the 5 future frames; stored with per-step
progress, success and cube positions. The executed chunk is candidate 0, or on odd episodes the oracle-best candidate
on 50% of decisions (seeded). The executed chunk is replayed on the live env and must reproduce its branch's last
frame exactly. One npz per episode. Compute node only.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gcfbc  # noqa: E402
from ti_wm import ogb_runtime as ogb  # noqa: E402
from ti_wm.contract import candidate_seed, require_compute, select_candidate  # noqa: E402

K = 8
MIX_STREAM = 6000
KEEP = (2, 3, 4, 5)        # rendered future frames per candidate: steps 2-4 (path) and 5 (end); rendering dominates cost


def oracle_step(root, ep, d):
    if ep % 2 == 0:
        return False
    return bool(np.random.default_rng(candidate_seed(root, d, MIX_STREAM)).random() < 0.5)


def branch(env, state, chunk):
    """Simulate one chunk from `state`: rendered frames after steps KEEP (last rendered repeated if the episode ends
    early), progress after each step, success, cube positions at the end. The env is restored afterwards."""
    u = env.unwrapped
    ogb.load_state(env, state)
    frames, prog, success, ended, last = {}, [], False, False, None
    for step, a in enumerate(chunk, start=1):
        if not ended:
            # OGBench visual envs use success_timing='pre': the step on which the pre-step success flag is set still
            # executes its action and then terminates. Mirror that so the branch equals the live replay.
            ended = bool(u._success)
            success = ogb.physics_step(env, a) or success
            if step in KEEP or ended:
                last = np.asarray(u.compute_observation())
            prog.append(ogb.progress(env))
        else:
            prog.append(prog[-1])
        if step in KEEP:
            frames[step] = last
    frames = np.stack([frames[s] for s in KEEP])
    cubes = np.stack([u._data.joint(f"object_joint_{i}").qpos[:3].copy() for i in range(u._num_cubes)])
    ogb.load_state(env, state)
    return np.stack(frames), np.asarray(prog, np.float32), success, cubes


def episode(env, policy, device, task, ep):
    root = gcfbc.root_id(task, ep)
    ob, goal = ogb.reset(env, task, seed=root)
    ob, prev = np.asarray(ob), np.asarray(ob)
    goal_t = torch.from_numpy(np.asarray(goal))[None].to(device)
    rec = {k: [] for k in ("cur", "prev", "chunks", "fut", "prog", "success", "cubes", "executed", "t")}
    t, d, done, success = 0, 0, False, False
    while not done:
        feat = policy.features(torch.from_numpy(ob)[None].to(device), goal_t)
        chunks = policy.sample(feat.repeat(K, 1), gcfbc.seeded_noise([candidate_seed(root, d, j) for j in range(K)],
                                                                     device)).float().cpu().numpy()
        state = ogb.save_state(env)
        sims = [branch(env, state, c) for c in chunks]
        labels = [s[1][-1] for s in sims]
        chosen = select_candidate([float(x) for x in labels]) if oracle_step(root, ep, d) else 0
        for key, val in (("cur", ob), ("prev", prev), ("chunks", chunks), ("fut", np.stack([s[0] for s in sims])),
                         ("prog", np.stack([s[1] for s in sims])), ("success", np.array([s[2] for s in sims])),
                         ("cubes", np.stack([s[3] for s in sims])), ("executed", chosen), ("t", t)):
            rec[key].append(val)
        last = None
        for a in chunks[chosen]:
            prev = ob
            ob, _, terminated, truncated, info = env.step(a)
            ob = np.asarray(ob)
            last = ob
            t += 1
            success = success or bool(info.get("success", False))
            if terminated or truncated:
                done = True
                break
        steps_run = t - rec["t"][-1]
        if steps_run >= KEEP[0] and not np.array_equal(last, sims[chosen][0][KEEP.index(min(steps_run, KEEP[-1]))]):
            raise RuntimeError(f"replay of the executed chunk differs from its branch (root {root}, decision {d})")
        d += 1
    out = {k: np.asarray(v) for k, v in rec.items()}
    out.update(goal=np.asarray(goal), root=root, task=task, ep=ep, episode_success=success)
    return out


def main(a):
    require_compute()
    device = torch.device(a.device)
    blob = torch.load(a.checkpoint, map_location=device)
    policy = gcfbc.GCFlowPolicy().to(device).eval()
    policy.load_state_dict(blob["policy"])
    env = ogb.make_env(a.env)
    ogb.reset(env, 1, seed=gcfbc.root_id(1, 999))
    report = {"env": a.env, "checkpoint": str(a.checkpoint), "restore_max_dqpos":
              ogb.check_restore(env, np.random.default_rng(0).uniform(-1, 1, (20, gcfbc.ACT))),
              "task": a.task, "episodes": [a.first, a.first + a.count - 1], "K": K, "job": os.environ.get("SLURM_JOB_ID"),
              "sim_steps": 0, "decisions": 0, "episode_success": []}
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for ep in range(a.first, a.first + a.count):
        e = episode(env, policy, device, a.task, ep)
        np.savez_compressed(a.out / f"task{a.task}_ep{ep}.npz", **e)
        n = len(e["t"])
        report["decisions"] += n
        report["sim_steps"] += int(n * K * gcfbc.H)
        report["episode_success"].append(bool(e["episode_success"]))
        report["seconds"] = time.perf_counter() - t0
        (a.out / f"report_task{a.task}_ep{a.first}.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"ep": ep, "decisions": n, "success": bool(e["episode_success"]),
                          "sec": round(time.perf_counter() - t0)}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--task", type=int, required=True)
    p.add_argument("--first", type=int, default=1000)
    p.add_argument("--count", type=int, default=60)
    p.add_argument("--device", default="cpu", help="the policy is small; rendering (CPU) dominates the cost")
    main(p.parse_args())
