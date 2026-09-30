"""LIBERO L4: continuation value of candidate chunks (docs/CTA_LIBERO_CONTINUATION_PROTOCOL.md).

Label = task success when the frozen policy continues from the end of a candidate chunk, not goal progress at the
end of the chunk (L3 showed that endpoint-vs-goal selection has no closed-loop gain).

Per root (task, init) the live env follows P0. At the pre-registered anchor times ANCHORS_T (env steps; the first
decision at or after each) all K candidates are branched from the saved live state (exact clone, L2). Each branch
runs its chunk, then continues with P0 under STREAMS independent noise streams to the episode end. Stream s uses the
same seeds for every sibling (common random numbers). Streams 0-1 select, streams 2-3 evaluate (aggregate mode).
Recorded per candidate: chunk frames, eef and scene qpos per step, goal progress at chunk end, continuation outcomes.
Anchors are fixed by time only; an anchor that the episode never reaches is recorded as missing, not replaced.

Modes: collect (compute node, LIBERO venv) and aggregate (CPU node; numpy only).
"""

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

K, STREAMS, SELECT_STREAMS = 8, 4, (0, 1)
ANCHORS_T = (20, 60, 120)
ROOT_BASE, STREAM_BASE = 100_000, 1_000_000


def root_id(task, init):
    return ROOT_BASE + 100 * task + init


def stream_root(root, s):
    return root + STREAM_BASE * (s + 1)


def continue_p0(policy, env, obs, text, root_s, d, t, max_steps):
    """Frozen policy from the current env state to success or the step limit; returns (success, steps taken)."""
    from ti_wm.libero_runtime import candidate_seeds, run_chunk

    t0, success = t, False
    while not success and t < max_steps:
        chunk = policy.bank(obs, text, candidate_seeds(root_s, d, 1))[:, : max_steps - t]
        obs, success, n, _, _ = run_chunk(env, chunk[0])
        t += n
        d += 1
    return success, t - t0


def branch(policy, worker, state, bank, text, root, d, t, max_steps):
    """All K candidates from one live state: chunk record plus STREAMS continuation outcomes each."""
    from ti_wm.libero_runtime import goal_progress, load_state, qpos, run_chunk, save_state, scene_state

    cands, frames = [], []
    for k in range(K):
        load_state(worker, state)
        keep = set(range(1, bank.shape[1] + 1))
        obs, chunk_success, n, kept, trace = run_chunk(worker, bank[k], keep=keep)
        end = save_state(worker)
        scene = scene_state(worker)
        rec = {"k": k, "chunk_steps": n, "chunk_success": bool(chunk_success),
               "progress_end": 1.0 if chunk_success else float(goal_progress(worker, policy.goal_ref)),
               "eef_end": scene["eef"].tolist(), "cont": [], "cont_steps": []}
        frames.append({"pixels": {c: np.stack([kept[i]["pixels"][c] for i in sorted(kept)]) for c in obs["pixels"]},
                       "qpos": np.stack(trace), "scene_qpos_end": scene["scene_qpos"]})
        for s in range(STREAMS):
            if chunk_success:
                ok, steps = True, 0
            else:
                obs_s = load_state(worker, end)
                ok, steps = continue_p0(policy, worker, obs_s, text, stream_root(root, s), d + 1, t + n, max_steps)
            rec["cont"].append(bool(ok))
            rec["cont_steps"].append(int(steps))
        cands.append(rec)
    return cands, frames


def run_root(policy, out_dir, task, init):
    from ti_wm.libero_runtime import candidate_seeds, goal_reference, make_env, run_chunk, save_state

    root = root_id(task, init)
    live, worker = make_env(policy.cfg, task, init), make_env(policy.cfg, task, init)
    try:
        obs, _ = live.reset(seed=0)
        worker.reset(seed=0)
        policy.goal_ref = goal_reference(live)
        text, max_steps = live.task_description, live._max_episode_steps
        t, d, success, anchors, pending = 0, 0, False, [], list(ANCHORS_T)
        while not success and t < max_steps:
            at_anchor = bool(pending) and t >= pending[0]
            bank = policy.bank(obs, text, candidate_seeds(root, d, K if at_anchor else 1))[:, : max_steps - t]
            if at_anchor:
                target = pending.pop(0)
                t0 = time.perf_counter()
                cands, frames = branch(policy, worker, save_state(live), bank, text, root, d, t, max_steps)
                anchor_pixels = {c: obs["pixels"][c] for c in obs["pixels"]}
                np.savez_compressed(out_dir / f"root{root}_t{target}.npz", actions=bank,
                                    **{f"anchor_{c}": v for c, v in anchor_pixels.items()},
                                    **{f"k{k}_{c}": f["pixels"][c] for k, f in enumerate(frames) for c in f["pixels"]},
                                    **{f"k{k}_qpos": f["qpos"] for k, f in enumerate(frames)},
                                    **{f"k{k}_scene_end": f["scene_qpos_end"] for k, f in enumerate(frames)})
                anchors.append({"target_t": target, "t": t, "d": d, "candidates": cands,
                                "seconds": time.perf_counter() - t0})
                print(json.dumps({"root": root, "anchor": target, "t": t, "sec": round(anchors[-1]["seconds"]),
                                  "cont_mean": [round(float(np.mean(c["cont"])), 2) for c in cands]}), flush=True)
            obs, success, n, _, _ = run_chunk(live, bank[0])
            t += n
            d += 1
        for target in pending:
            anchors.append({"target_t": target, "missing": True, "reason": "success" if success else "step limit"})
        return {"task": task, "init": init, "root": root, "language": text, "p0_success": bool(success),
                "p0_steps": t, "anchors": anchors}
    finally:
        live.close()
        worker.close()


def collect(run, setup, tasks, inits, device):
    from ti_wm.libero_runtime import Policy

    assert os.environ.get("SLURM_JOB_ID"), "Run through sbatch"
    ckpt = json.loads((setup / "checkpoints.json").read_text())["HuggingFaceVLA/smolvla_libero"]["path"]
    policy = Policy(ckpt, device)
    tag = f"tasks{'-'.join(map(str, tasks))}_inits{'-'.join(map(str, inits))}"
    (run / "frames").mkdir(parents=True, exist_ok=True)
    with open(run / f"l4_{tag}.jsonl", "w") as stream:
        for task in tasks:
            for init in inits:
                t0 = time.perf_counter()
                try:
                    rec = run_root(policy, run / "frames", task, init)
                except Exception:
                    rec = {"task": task, "init": init, "root": root_id(task, init), "error": traceback.format_exc()}
                rec["seconds"] = time.perf_counter() - t0
                stream.write(json.dumps(rec) + "\n")
                stream.flush()
                print(json.dumps({"task": task, "init": init, "sec": round(rec["seconds"]),
                                  "p0": rec.get("p0_success"), **({"error": rec["error"][-400:]} if "error" in rec else {})}),
                      flush=True)


# ----------------------------------------------------------------------------- aggregate

def first_best(scores):
    return max(range(len(scores)), key=lambda i: scores[i])


def anchor_stats(anchor):
    """Held-out gain of three selectors over candidate 0, all evaluated on streams 2-3."""
    c = anchor["candidates"]
    sel = [float(np.mean([x["cont"][s] for s in SELECT_STREAMS])) for x in c]
    ev = [float(np.mean([x["cont"][s] for s in range(STREAMS) if s not in SELECT_STREAMS])) for x in c]
    allv = [float(np.mean(x["cont"])) for x in c]
    prog = [x["progress_end"] for x in c]
    return {"cont_select": ev[first_best(sel)] - ev[0], "progress_select": ev[first_best(prog)] - ev[0],
            "oracle_all_streams": max(allv) - allv[0], "varies": len(set(allv)) > 1,
            "rho_progress_value": _spearman(prog, allv)}


def _spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def endpoint_ties(anchor, eps=0.02):
    """Sibling pairs whose chunk-end goal progress is within eps but whose continuation value differs by >= 0.5."""
    c, n, differ = anchor["candidates"], 0, 0
    for i in range(K):
        for j in range(i + 1, K):
            if abs(c[i]["progress_end"] - c[j]["progress_end"]) < eps:
                n += 1
                differ += int(abs(np.mean(c[i]["cont"]) - np.mean(c[j]["cont"])) >= 0.5)
    return n, differ


def bootstrap(by_root, key, reps=10_000, seed=0):
    roots = sorted(by_root)
    vals = np.array([np.mean([a[key] for a in by_root[r]]) for r in roots])
    rng = np.random.default_rng(seed)
    boots = vals[rng.integers(0, len(vals), (reps, len(vals)))].mean(1)
    return {"mean": float(vals.mean()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
            "n_roots": len(vals)}


def aggregate(run):
    rows = [json.loads(line) for f in sorted(run.glob("l4_*.jsonl")) for line in open(f)]
    errors = [r for r in rows if "error" in r]
    by_root, ties, missing, per_anchor_t = {}, [0, 0], 0, {}
    for r in rows:
        for a in r.get("anchors", []):
            if a.get("missing"):
                missing += 1
                continue
            s = anchor_stats(a)
            by_root.setdefault(r["root"], []).append(s)
            per_anchor_t.setdefault(a["target_t"], []).append(s)
            n, dif = endpoint_ties(a)
            ties[0] += n
            ties[1] += dif
    report = {"roots": len(rows), "errors": len(errors), "anchors_present": sum(map(len, by_root.values())),
              "anchors_missing": missing,
              "p0_success": float(np.mean([r["p0_success"] for r in rows if "p0_success" in r])) if rows else None,
              "anchors_with_varying_value": float(np.mean([bool(s["varies"]) for v in by_root.values() for s in v])),
              "gain_cont_select_heldout": bootstrap(by_root, "cont_select"),
              "gain_progress_select_heldout": bootstrap(by_root, "progress_select"),
              "oracle_all_streams_optimistic": bootstrap(by_root, "oracle_all_streams"),
              "median_rho_progress_vs_value": float(np.nanmedian([s["rho_progress_value"]
                                                                  for v in by_root.values() for s in v])),
              "endpoint_tied_pairs": ties[0], "endpoint_tied_pairs_value_differs": ties[1],
              "by_anchor_t": {t: {"n": len(v), "cont_select": float(np.mean([s["cont_select"] for s in v])),
                                  "progress_select": float(np.mean([s["progress_select"] for s in v]))}
                              for t, v in sorted(per_anchor_t.items())}}
    (run / "l4_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["collect", "aggregate"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--setup", type=Path)
    p.add_argument("--tasks", type=int, nargs="+")
    p.add_argument("--inits", type=int, nargs="+", default=[0, 1])
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    if a.mode == "collect":
        collect(a.run, a.setup, a.tasks, a.inits, a.device)
    else:
        aggregate(a.run)
