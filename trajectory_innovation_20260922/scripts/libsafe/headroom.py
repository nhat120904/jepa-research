"""LIBERO-Safety headroom run: P0 and ORACLE<K> with the frozen pi0.5 (docs/CTA_LIBSAFE_PROTOCOL.md, step 1).

P0 executes candidate 0 of the seeded K-bank (the policy as released; the bank is drawn in full so candidate 0 is the
same batched computation for every arm). ORACLE<K> (privileged, never deployable) simulates each candidate's executed
5 steps from the exact state in a camera-free twin instance and executes the best oracle_score (no violation first,
then goal progress). Per decision it records every candidate's native labels, which gives how often the bank contains
both violating and violation-free chunks. Fidelity: after the chosen chunk runs in the real episode, its qpos and
violation flags are compared with the twin's prediction for that chunk.
Episodes are sharded over worker processes (--shard/--shards); aggregate with --aggregate. Compute node only.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ti_wm import libsafe_runtime as ls  # noqa: E402
from ti_wm.contract import require_compute, select_candidate  # noqa: E402
from ti_wm.pi05_policy import bank_seeds  # noqa: E402

ROOT_BASE = 700_000                      # disjoint from PushT (2xxx/3xxxx), LIBERO (1xxxxx), OGBench (5xxxxx)


def root_id(suite, level, level_id, init):
    return ROOT_BASE + 100_000 * ls.SUITES.index(suite) + 10_000 * level + 1_000 * level_id + init


def parse_tasks(spec):
    """'obstacle_avoidance:1:0-4,obstacle_avoidance_human:1:0-4' -> [(suite, level, level_id)]."""
    out = []
    for part in spec.split(","):
        suite, level, ids = part.split(":")
        lo, hi = (int(x) for x in ids.split("-"))
        out += [(suite, int(level), i) for i in range(lo, hi + 1)]
    return out


def episode(pi, env, benv, lang, init_state, root, arm, k):
    horizon = ls.max_steps(lang)
    if benv is not None:
        benv.seed(root)
        benv.reset()
    obs = ls.start(env, init_state, root)
    ref = ls.goal_reference(env)
    t = d = 0
    success, viol_steps, cons = False, 0, {}
    times = {"policy": 0.0, "branch": 0.0, "step": 0.0}
    dec = []
    while t < horizon and not success:
        t0 = time.perf_counter()
        chunks = pi.bank(ls.policy_inputs(obs, lang), bank_seeds(root, d, k))[:, :ls.REPLAN]
        t1 = time.perf_counter()
        times["policy"] += t1 - t0
        chosen, labels = 0, None
        if arm.startswith("ORACLE"):
            state = ls.save_state(env)
            labels = [ls.branch(benv, state, c, ref) for c in chunks]
            chosen = select_candidate([ls.oracle_score(x["success"], x["violation"], x["progress"]) for x in labels])
        t2 = time.perf_counter()
        times["branch"] += t2 - t1
        viol, executed = [], 0
        for a in chunks[chosen]:
            obs, success, cost = ls.step(env, a)
            t += 1
            executed += 1
            v = ls.violated(cost)
            viol.append(v)
            viol_steps += v
            for name, c in cost.items():
                cons[name] = cons.get(name, 0) + int(c)
            if success or t >= horizon:
                break
        times["step"] += time.perf_counter() - t2
        rec = {"d": d, "chosen": int(chosen), "viol": any(viol)}
        if labels is not None:
            rec["cand_viol"] = [bool(x["violation"]) for x in labels]
            rec["cand_success"] = [bool(x["success"]) for x in labels]
            rec["cand_progress"] = [round(float(x["progress"]), 5) for x in labels]
            lab = labels[chosen]
            if executed == len(lab["viol_steps"]):     # chunk ran as far as the twin ran it
                rec["fid_dqpos"] = float(np.max(np.abs(ls.qpos(env) - lab["qpos"])))
                rec["fid_dmocap"] = float(np.max(np.abs(ls.mocap(env) - lab["mocap"]))) if lab["mocap"].size else 0.0
                rec["fid_viol_match"] = bool(lab["viol_steps"] == viol)
        dec.append(rec)
        d += 1
    return {"root": root, "arm": arm, "success": bool(success), "violation": viol_steps > 0, "viol_steps": viol_steps,
            "constraints": cons, "steps": t, "decisions": d, "horizon": horizon, "sec": times, "decision_log": dec}


def run(a):
    require_compute()
    from ti_wm.pi05_policy import Pi05

    ls.write_config(str(a.run / "libero_config"), a.repo)
    pi = Pi05(a.checkpoint)
    jobs = []
    for suite, level, level_id in parse_tasks(a.tasks):
        for init in range(a.first, a.first + a.count):
            for arm in a.arms.split(","):
                jobs.append((suite, level, level_id, init, arm))
    jobs = jobs[a.shard::a.shards]
    out = (a.run / f"episodes_shard{a.shard}.jsonl").open("a")
    cache = {}
    for suite, level, level_id, init, arm in jobs:
        key = (suite, level, level_id)
        if key not in cache:
            for env, benv, *_ in cache.values():
                env.close()
                benv.close()
            cache.clear()
            lang, bddl, inits = ls.task_info(suite, level, level_id)
            root0 = root_id(suite, level, level_id, 0)
            cache[key] = (ls.make_env(bddl, root0), ls.make_env(bddl, root0, cameras=False), lang, inits)
        env, benv, lang, inits = cache[key]
        k = int(arm[len("ORACLE"):]) if arm.startswith("ORACLE") else a.k
        e = episode(pi, env, benv if arm.startswith("ORACLE") else None, lang, inits[init],
                    root_id(suite, level, level_id, init), arm, k)
        e.update({"suite": suite, "level": level, "level_id": level_id, "init": init, "language": lang,
                  "job": os.environ.get("SLURM_JOB_ID")})
        out.write(json.dumps(e) + "\n")
        out.flush()
        print(arm, suite, level, level_id, init, "success", e["success"], "violation", e["violation"], "steps",
              e["steps"], {k2: round(v, 1) for k2, v in e["sec"].items()}, flush=True)


def aggregate(a):
    eps = [json.loads(l) for f in sorted(a.run.glob("episodes_shard*.jsonl")) for l in open(f)]
    rep = {"episodes": len(eps), "arms": {}}
    for arm in sorted({e["arm"] for e in eps}):
        es = [e for e in eps if e["arm"] == arm]
        steps = sum(e["steps"] for e in es)
        r = {"n": len(es), "success": float(np.mean([e["success"] for e in es])),
             "violation": float(np.mean([e["violation"] for e in es])),
             "safe_success": float(np.mean([e["success"] and not e["violation"] for e in es])),
             "by_task": {}, "sec_per_step": {k: sum(e["sec"][k] for e in es) / max(steps, 1) for k in es[0]["sec"]}}
        for key in sorted({(e["suite"], e["level"], e["level_id"]) for e in es}):
            te = [e for e in es if (e["suite"], e["level"], e["level_id"]) == key]
            r["by_task"]["%s:L%d:%d" % key] = {"n": len(te), "success": float(np.mean([e["success"] for e in te])),
                                               "violation": float(np.mean([e["violation"] for e in te]))}
        logs = [x for e in es for x in e["decision_log"] if "cand_viol" in x]
        if logs:
            nv = np.array([sum(x["cand_viol"]) for x in logs])
            fid = [x for x in logs if "fid_dqpos" in x]
            r["bank"] = {"decisions": len(logs), "mixed_violation": float(np.mean((nv > 0) & (nv < len(logs[0]["cand_viol"])))),
                         "all_violate": float(np.mean(nv == len(logs[0]["cand_viol"]))),
                         "any_violate": float(np.mean(nv > 0)),
                         "fid_checked": len(fid),
                         "fid_max_dqpos": max((x["fid_dqpos"] for x in fid), default=None),
                         "fid_max_dmocap": max((x["fid_dmocap"] for x in fid), default=None),
                         "fid_viol_mismatch": int(sum(not x["fid_viol_match"] for x in fid))}
        rep["arms"][arm] = r
    arms = sorted(rep["arms"])
    if len(arms) == 2:                                   # paired by root
        by = {(e["arm"], e["root"]): e for e in eps}
        roots = sorted({e["root"] for e in eps if (arms[0], e["root"]) in by and (arms[1], e["root"]) in by})
        for m in ("success", "violation"):
            x = np.array([by[(arms[0], r)][m] for r in roots], int)
            y = np.array([by[(arms[1], r)][m] for r in roots], int)
            rep[f"paired_{m}"] = {"roots": len(roots), f"{arms[1]}-{arms[0]}": float(np.mean(y - x)),
                                  "discordant": [int(np.sum((x == 1) & (y == 0))), int(np.sum((x == 0) & (y == 1)))]}
    (a.run / "headroom_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--repo", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--tasks", default="obstacle_avoidance:1:0-4,obstacle_avoidance_human:1:0-4")
    p.add_argument("--arms", default="P0,ORACLE8")
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args()
    aggregate(args) if args.aggregate else run(args)
