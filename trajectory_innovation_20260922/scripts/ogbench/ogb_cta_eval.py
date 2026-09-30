"""Closed-loop evaluation of CTA and the matched rerankers on OGBench (docs/CTA_OGBENCH_PROTOCOL.md, "Evaluation").

Same frozen policy and the same seeded 8-chunk bank as P0/ORACLE (scripts/ogbench/gcfbc.py eval): at every decision
the policy draws its 8 chunks; the arm scores them from (decision frame, previous frame, goal image) and executes the
best (tie -> candidate 0). Deployable arms never see simulated futures. FULL / CODE are privileged ceilings: each chunk
is simulated from an exact restore (as in collection) and its actual future is read.
Per-decision wall time is split into policy sampling, tokenization (shared by all learned arms), and the arm's scorer
(world model + reader), with CUDA synchronization. Compute node only.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import gcfbc  # noqa: E402
import ogb_collect  # noqa: E402
from ogb_cta_train import build_stage1  # noqa: E402
from ti_wm import ogb_runtime as ogb  # noqa: E402
from ti_wm.codec import pool_grid  # noqa: E402
from ti_wm.contract import candidate_seed, require_compute, select_candidate  # noqa: E402
from ti_wm.cta_ogb import ActEndpointWM, ActParallelFSQWM, FrameWM, fut_from_frames, zeros_prop  # noqa: E402
from ti_wm.pusht_runtime import VisualScorer  # noqa: E402
from ti_wm.sibling import project  # noqa: E402

K = 8
DEPLOYABLE = ("P0", "CTA", "ENDPOINT", "FRAME", "DIRECT")
PRIVILEGED = ("FULL", "CODE")


class Tokens:
    def __init__(self, pca_path, device):
        self.visual = VisualScorer(str(device))
        blob = torch.load(pca_path, map_location=device)
        self.mean, self.basis = blob["pca_mean"].to(device), blob["pca_basis"].to(device)

    @torch.inference_mode()
    def __call__(self, frames, side=None):
        x = project(self.visual.features(np.asarray(frames)), self.mean, self.basis)
        return (pool_grid(x, side) if side else x).half()


class Arms:
    def __init__(self, ckpt, device):
        blob = torch.load(ckpt, map_location=device)
        if blob["config"].get("levels"):              # v2 / play checkpoints may use other code levels
            import ogb_cta_train_v2 as v2
            self.mods = v2.build_stage1(blob["config"], device)
            built = v2.build_wms(blob["config"], "cpu")
            self.wms = {"CTA": built["cta"], "ENDPOINT": built["endpoint"], "FRAME": built["frame"]}
        else:
            self.mods = build_stage1(blob["config"], device)
            self.wms = {"CTA": ActParallelFSQWM(m=blob["config"]["m"]), "ENDPOINT": ActEndpointWM(), "FRAME": FrameWM()}
        for k, m in self.mods.items():
            m.load_state_dict(blob["stage1"][k])
            m.eval().requires_grad_(False)
        for (n, w), key in zip(self.wms.items(), ("cta", "endpoint", "frame")):
            w.load_state_dict(blob["wms"][key])
            w.to(device).eval().requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def score(self, arm, ctx, goal, act, fut=None):
        m = self.mods
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if arm == "CTA":
                s = m["reader"](ctx, self.wms["CTA"](ctx, act)[0], goal)
            elif arm == "ENDPOINT":
                o = self.wms["ENDPOINT"](ctx, act)
                s = m["full"](ctx, {"end": o["end"], "prop": o["prop"]}, goal)
            elif arm == "FRAME":
                roll = self.wms["FRAME"].rollout(ctx, act)
                s = m["full"](ctx, {"end": roll[:, 3], "prop": zeros_prop(len(roll), roll.device)}, goal)
            elif arm == "DIRECT":
                s = m["direct"](ctx, act, goal)
            elif arm == "FULL":
                s = m["full"](ctx, fut, goal)
            elif arm == "CODE":
                s = m["reader"](ctx, m["enc"](ctx, fut), goal)
            else:
                raise ValueError(arm)
        return s.float().cpu().numpy()


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def episode(env, policy, arms, tok, device, task, ep, arm):
    root = gcfbc.root_id(task, ep)
    ob, goal = ogb.reset(env, task, seed=root)
    ob = np.asarray(ob)
    prev = ob
    goal_tok = tok(np.asarray(goal)[None]).repeat_interleave(K, 0) if arm != "P0" else None
    goal_img = torch.from_numpy(np.asarray(goal))[None].to(device)
    d, steps, success = 0, 0, False
    times = {"policy": 0., "tokens": 0., "scorer": 0., "simulate": 0.}
    choices = []
    while True:
        t0 = sync()
        feat = policy.features(torch.from_numpy(ob)[None].to(device), goal_img)
        chunks = policy.sample(feat.repeat(K, 1), gcfbc.seeded_noise([candidate_seed(root, d, j) for j in range(K)],
                                                                     device)).float()
        t1 = sync()
        times["policy"] += t1 - t0
        chosen = 0
        if arm != "P0":
            ctx = {"cur": tok(ob[None]).repeat_interleave(K, 0), "prev": tok(prev[None], 8).repeat_interleave(K, 0),
                   "prop": zeros_prop(K, device)}
            t2 = sync()
            times["tokens"] += t2 - t1
            fut = None
            if arm in PRIVILEGED:
                state = ogb.save_state(env)
                frames = np.stack([ogb_collect.branch(env, state, c)[0] for c in chunks.cpu().numpy()])
                t3 = sync()
                times["simulate"] += t3 - t2
                fut = fut_from_frames(torch.stack([tok(f) for f in frames]).to(device))
                t2 = sync()
            s = arms.score(arm, ctx, goal_tok, chunks, fut)
            times["scorer"] += sync() - t2
            chosen = select_candidate([float(x) for x in s])
        choices.append(int(chosen))
        done = False
        for a in chunks[chosen].cpu().numpy():
            prev = ob
            ob, _, terminated, truncated, info = env.step(a)
            ob = np.asarray(ob)
            steps += 1
            success = success or bool(info.get("success", False))
            if terminated or truncated:
                done = True
                break
        d += 1
        if done:
            return {"task": task, "ep": ep, "root": root, "arm": arm, "success": success, "steps": steps,
                    "decisions": d, "override": float(np.mean(np.array(choices) != 0)), "sec": times}


def main(a):
    require_compute()
    device = torch.device("cuda")
    blob = torch.load(a.policy, map_location=device)
    policy = gcfbc.GCFlowPolicy().to(device).eval()
    policy.load_state_dict(blob["policy"])
    arms = Arms(a.ckpt, device)
    tok = Tokens(a.pca, device)
    env = ogb.make_env(a.env)
    ogb.reset(env, 1, seed=gcfbc.root_id(1, 999))
    report = {"env": a.env, "policy": str(a.policy), "ckpt": str(a.ckpt), "arms": a.arms, "episodes": [a.first, a.first + a.count - 1],
              "restore_max_dqpos": ogb.check_restore(env, np.random.default_rng(0).uniform(-1, 1, (20, gcfbc.ACT))),
              "job": os.environ.get("SLURM_JOB_ID"), "results": {}}
    a.run.mkdir(parents=True, exist_ok=True)
    tasks = [int(t) for t in a.tasks.split(",")]
    with (a.run / "episodes.jsonl").open("a") as stream:
        for arm in a.arms.split(","):
            res = []
            for task in tasks:
                for ep in range(a.first, a.first + a.count):
                    e = episode(env, policy, arms, tok, device, task, ep, arm)
                    res.append(e)
                    stream.write(json.dumps(e) + "\n")
                    stream.flush()
            dec = sum(e["decisions"] for e in res)
            report["results"][arm] = {
                "success": float(np.mean([e["success"] for e in res])),
                "by_task": {t: float(np.mean([e["success"] for e in res if e["task"] == t])) for t in tasks},
                "override": float(np.mean([e["override"] for e in res])),
                "ms_per_decision": {k: 1000 * sum(e["sec"][k] for e in res) / max(dec, 1) for k in res[0]["sec"]}}
            print(arm, json.dumps(report["results"][arm]), flush=True)
            (a.run / "eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for k in ("run", "policy", "ckpt", "pca"):
        p.add_argument(f"--{k}", type=Path, required=True)
    p.add_argument("--env", required=True)
    p.add_argument("--arms", default="P0,CTA,DIRECT,ENDPOINT,FRAME")
    p.add_argument("--tasks", default="1,2,3,4,5")
    p.add_argument("--first", type=int, default=0)
    p.add_argument("--count", type=int, default=20)
    main(p.parse_args())
