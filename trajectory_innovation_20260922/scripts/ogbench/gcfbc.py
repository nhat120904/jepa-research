"""Goal-conditioned flow BC with action chunks: the frozen proposal policy of CTA on OGBench (docs/CTA_OGBENCH_PLAN.md).

OGBench releases no checkpoints and its baselines act one step at a time, so the design's proposal policy (a trained
goal-conditioned policy that samples action chunks from seeded noise) is trained here with published recipes only:
- OGBench visual GCBC (impls/agents/gcbc.py, hyperparameters.sh): IMPALA-small encoder with early fusion of the
  observation and goal images, random-crop augmentation (padding 3, p = 0.5, same crop for both), actor goals drawn
  uniformly from the future of the same trajectory (actor_p_trajgoal = 1), batch 256, lr 3e-4, 500k steps;
- flow-matching action-chunk head as in Q-chunking / FQL on OGBench: chunk H = 5, 10 Euler flow steps, MLP 4 x 512.
Nothing is tuned toward a weaker or stronger policy; its own success is reported as P0.

Modes: train | eval. eval runs arms on 5 tasks x N seeds: P0 (candidate 0), ORACLE<K> (privileged: simulate each of
the K seeded chunks from an exact restore and execute the one with the best cube-target progress). The oracle arms
are headroom diagnostics, never a deployable method. Compute node only.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ti_wm.contract import candidate_seed, require_compute, select_candidate  # noqa: E402

H, ACT, FLOW_STEPS = 5, 5, 10
ROOT_BASE = 500_000                      # OGBench roots are disjoint from PushT (2xxx/3xxxx) and LIBERO (1xxxxx)


def root_id(task, episode):
    return ROOT_BASE + 1000 * task + episode


class ResStack(nn.Module):
    def __init__(self, cin, c):
        super().__init__()
        self.conv = nn.Conv2d(cin, c, 3, padding=1)
        self.pool = nn.MaxPool2d(3, 2, padding=1)
        self.res = nn.Sequential(nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1))

    def forward(self, x):
        x = self.pool(self.conv(x))
        return x + self.res(x)


class Impala(nn.Module):
    """IMPALA-small (stacks 16/32/32, one residual block each, MLP 512), OGBench's visual encoder."""

    def __init__(self, cin=6, stacks=(16, 32, 32), out=512, size=64):
        super().__init__()
        layers, c = [], cin
        for s in stacks:
            layers.append(ResStack(c, s))
            c = s
        self.stacks = nn.Sequential(*layers)
        side = size
        for _ in stacks:
            side = (side + 1) // 2
        self.fc = nn.Linear(c * side * side, out)

    def forward(self, x):
        return F.relu(self.fc(F.relu(self.stacks(x)).flatten(1)))


class GCFlowPolicy(nn.Module):
    def __init__(self, width=512, depth=4):
        super().__init__()
        self.enc = Impala()
        dims = [512 + H * ACT + 1] + [width] * depth
        mlp = []
        for a, b in zip(dims[:-1], dims[1:]):
            mlp += [nn.Linear(a, b), nn.GELU()]
        self.mlp = nn.Sequential(*mlp, nn.Linear(width, H * ACT))

    def features(self, ob, goal):
        """uint8 (B, 64, 64, 3) images -> (B, 512); early fusion as OGBench's concat encoder."""
        x = torch.cat([ob, goal], -1).permute(0, 3, 1, 2).float() / 255.
        return self.enc(x)

    def velocity(self, feat, x, t):
        return self.mlp(torch.cat([feat, x.flatten(1), t[:, None]], -1)).view(-1, H, ACT)

    def loss(self, ob, goal, chunk):
        feat = self.features(ob, goal)
        x0 = torch.randn_like(chunk)
        t = torch.rand(len(chunk), device=chunk.device)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * chunk
        return F.mse_loss(self.velocity(feat, xt, t), chunk - x0)

    @torch.no_grad()
    def sample(self, feat, noise):
        x = noise
        for i in range(FLOW_STEPS):
            t = torch.full((len(x),), i / FLOW_STEPS, device=x.device)
            x = x + self.velocity(feat, x, t) / FLOW_STEPS
        return x.clamp(-1, 1)


def seeded_noise(seeds, device):
    return torch.stack([torch.randn(H, ACT, generator=torch.Generator().manual_seed(int(s))) for s in seeds]).to(device)


class Data:
    """Compact OGBench dataset in RAM: observations (N, 64, 64, 3) uint8, actions (N, 5), terminals (N,)."""

    def __init__(self, path):
        with np.load(path) as z:
            self.ob = torch.from_numpy(z["observations"])
            self.act = torch.from_numpy(z["actions"].astype(np.float32))
            term = z["terminals"].astype(bool)
        ends = np.flatnonzero(term)
        starts = np.r_[0, ends[:-1] + 1]
        self.final = np.empty(len(term), np.int64)
        for s, e in zip(starts, ends):
            self.final[s:e + 1] = e
        idx = np.arange(len(term))
        self.valid = idx[idx + H - 1 <= self.final]          # the whole chunk stays inside one trajectory
        self.n_traj = len(ends)

    def batch(self, rng, b, device, p_aug=.5):
        i = rng.choice(self.valid, b)
        u = rng.random(b)
        # OGBench GCDataset trajgoal: uniform between min(i + 1, final) and the trajectory's final state
        g = np.round(np.minimum(i + 1, self.final[i]) * u + self.final[i] * (1 - u)).astype(np.int64)
        ob, goal = self.ob[i], self.ob[g]
        chunk = torch.stack([self.act[i + k] for k in range(H)], 1)
        if p_aug > 0:
            ob, goal = crop_pair(ob, goal, rng, p_aug)
        return ob.to(device), goal.to(device), chunk.to(device)


def crop_pair(ob, goal, rng, p_aug, pad=3):
    """Random shift of both images with the same offsets (edge padding), applied to the whole batch with p_aug."""
    if rng.random() >= p_aug:
        return ob, goal
    b = len(ob)
    off = rng.integers(0, 2 * pad + 1, (b, 2))
    out = []
    for x in (ob, goal):
        xp = F.pad(x.permute(0, 3, 1, 2).float(), (pad, pad, pad, pad), mode="replicate")
        crops = torch.stack([xp[j, :, off[j, 0]:off[j, 0] + 64, off[j, 1]:off[j, 1] + 64] for j in range(b)])
        out.append(crops.permute(0, 2, 3, 1).to(torch.uint8))
    return out


def train(a):
    device = torch.device("cuda")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data = Data(a.data / f"{a.env}.npz")
    policy = GCFlowPolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=3e-4)
    report = {"env": a.env, "steps": a.steps, "batch": 256, "lr": 3e-4, "H": H, "flow_steps": FLOW_STEPS,
              "p_aug": .5, "seed": a.seed, "train_transitions": int(len(data.act)), "trajectories": data.n_traj,
              "job": os.environ.get("SLURM_JOB_ID")}
    (a.run / "train_report.json").write_text(json.dumps(report, indent=2))
    t0 = time.perf_counter()
    for step in range(1, a.steps + 1):
        ob, goal, chunk = data.batch(rng, 256, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = policy.loss(ob, goal, chunk)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 5000 == 0 or step == 1:
            rec = {"step": step, "loss": float(loss), "hours": (time.perf_counter() - t0) / 3600}
            with (a.run / "train_log.jsonl").open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
        if step % a.save_every == 0 or step == a.steps:
            torch.save({"policy": policy.state_dict(), "step": step, "config": report}, a.run / f"policy_{step}.pt")
    report["train_hours"] = (time.perf_counter() - t0) / 3600
    (a.run / "train_report.json").write_text(json.dumps(report, indent=2))


class Distilled:
    """Oracle-best chunk of every TRAIN bank of the branched collection (docs/CTA_OGBENCH_PROTOCOL.md, DISTILL): the
    data-matched control. Same episodes as the CTA training split (heldout = last 10% of each task's episodes)."""

    def __init__(self, collect, heldout_frac=0.1):
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ogb_encode import split_files
        train, _ = split_files(collect, heldout_frac)
        obs, goals, chunks = [], [], []
        for f in train:
            with np.load(f) as z:
                best = np.array([select_candidate([float(x) for x in row]) for row in z["prog"][:, :, -1]])
                obs.append(z["cur"])
                goals.append(np.repeat(z["goal"][None], len(best), 0))
                chunks.append(z["chunks"][np.arange(len(best)), best])
        self.ob = torch.from_numpy(np.concatenate(obs))
        self.goal = torch.from_numpy(np.concatenate(goals))
        self.chunk = torch.from_numpy(np.concatenate(chunks).astype(np.float32))
        self.n = len(self.ob)

    def batch(self, rng, b, device, p_aug=.5):
        i = rng.integers(0, self.n, b)
        ob, goal = self.ob[i], self.goal[i]
        if p_aug > 0:
            ob, goal = crop_pair(ob, goal, rng, p_aug)
        return ob.to(device), goal.to(device), self.chunk[i].to(device)


def distill(a):
    """Fine-tune the frozen policy's checkpoint by BC on the oracle-best collected chunks, mixed 50/50 with the
    original dataset batches (keeps the policy's support), lr 1e-4, a.steps updates."""
    device = torch.device("cuda")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    blob = torch.load(a.checkpoint, map_location=device)
    policy = GCFlowPolicy().to(device)
    policy.load_state_dict(blob["policy"])
    data, best = Data(a.data / f"{a.env}.npz"), Distilled(a.collect)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    report = {"env": a.env, "from": str(a.checkpoint), "collect": str(a.collect), "distilled_chunks": best.n,
              "steps": a.steps, "mix": .5, "lr": 1e-4, "job": os.environ.get("SLURM_JOB_ID")}
    (a.run / "distill_report.json").write_text(json.dumps(report, indent=2))
    for step in range(1, a.steps + 1):
        o1, g1, c1 = best.batch(rng, 128, device)
        o2, g2, c2 = data.batch(rng, 128, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = policy.loss(torch.cat([o1, o2]), torch.cat([g1, g2]), torch.cat([c1, c2]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 5000 == 0 or step == 1:
            print({"step": step, "loss": float(loss)}, flush=True)
    torch.save({"policy": policy.state_dict(), "step": step, "config": report}, a.run / f"policy_distill_{step}.pt")


def episode(env, policy, device, task, ep, arm, K, log):
    from ti_wm import ogb_runtime as ogb

    root = root_id(task, ep)
    ob, goal = ogb.reset(env, task, seed=root)
    goal_t = torch.from_numpy(np.asarray(goal))[None].to(device)
    d, success, steps, t_policy, t_sim = 0, False, 0, 0., 0.
    k = 1 if arm == "P0" else K
    while True:
        tick = time.perf_counter()
        feat = policy.features(torch.from_numpy(np.asarray(ob))[None].to(device), goal_t)
        chunks = policy.sample(feat.repeat(k, 1), seeded_noise([candidate_seed(root, d, j) for j in range(k)], device))
        chunks = chunks.float().cpu().numpy()
        t_policy += time.perf_counter() - tick
        chosen = 0
        if arm != "P0":
            tick = time.perf_counter()
            state = ogb.save_state(env)
            labels = [ogb.simulate_chunk(env, state, c) for c in chunks]
            chosen = select_candidate(labels)
            log.append({"task": task, "ep": ep, "d": d, "labels": labels, "chosen": int(chosen)})
            t_sim += time.perf_counter() - tick
        done = False
        for act in chunks[chosen]:
            ob, _, terminated, truncated, info = env.step(act)
            steps += 1
            success = success or bool(info.get("success", False))
            if terminated or truncated:
                done = True
                break
        d += 1
        if done:
            return {"task": task, "ep": ep, "root": root, "arm": arm, "success": success, "steps": steps,
                    "decisions": d, "sec_policy": t_policy, "sec_oracle_sim": t_sim}


def evaluate(a):
    from ti_wm import ogb_runtime as ogb

    device = torch.device("cuda")
    blob = torch.load(a.checkpoint, map_location=device)
    policy = GCFlowPolicy().to(device).eval()
    policy.load_state_dict(blob["policy"])
    env = ogb.make_env(a.env)
    ogb.reset(env, 1, seed=root_id(1, 0))
    dev = ogb.check_restore(env, np.random.default_rng(0).uniform(-1, 1, (20, ACT)))
    report = {"env": a.env, "checkpoint": str(a.checkpoint), "step": blob["step"], "restore_max_dqpos": dev,
              "episodes_per_task": a.episodes, "arms": a.arms, "job": os.environ.get("SLURM_JOB_ID"), "results": {}}
    tasks = [int(t) for t in a.tasks.split(",")]
    with (a.run / "episodes.jsonl").open("a") as stream:
        for arm in a.arms.split(","):
            K = int(arm[len("ORACLE"):]) if arm.startswith("ORACLE") else 1
            log, res = [], []
            for task in tasks:
                for ep in range(a.first_episode, a.first_episode + a.episodes):
                    e = episode(env, policy, device, task, ep, arm, K, log)
                    res.append(e)
                    stream.write(json.dumps(e) + "\n")
                    stream.flush()
            by_task = {t: float(np.mean([e["success"] for e in res if e["task"] == t])) for t in tasks}
            report["results"][arm] = {"success": float(np.mean([e["success"] for e in res])), "by_task": by_task,
                                      "sec_policy_per_decision": float(sum(e["sec_policy"] for e in res) /
                                                                       max(1, sum(e["decisions"] for e in res)))}
            if log:
                lab = np.array([x["labels"] for x in log])
                report["results"][arm]["headroom"] = {
                    "decisions": len(lab), "share_with_gain": float(((lab.max(1) - lab[:, 0]) > 1e-4).mean()),
                    "mean_gain_m": float((lab.max(1) - lab[:, 0]).mean())}
                np.savez(a.run / f"oracle_log_{arm}.npz", labels=lab, chosen=np.array([x["chosen"] for x in log]),
                         task=np.array([x["task"] for x in log]), ep=np.array([x["ep"] for x in log]))
            print(arm, json.dumps(report["results"][arm]), flush=True)
            (a.run / "eval_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "eval", "distill"])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--env", default="visual-cube-single-play-v0")
    p.add_argument("--data", type=Path, default=Path("/mnt/data/nhatnc129/jepa/ogbench/data"))
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--save-every", type=int, default=100_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--arms", default="P0,ORACLE8,ORACLE16")
    p.add_argument("--tasks", default="1,2,3,4,5")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--first-episode", type=int, default=0)
    p.add_argument("--collect", type=Path, help="distill: branched collection directory")
    a = p.parse_args()
    require_compute()
    a.run.mkdir(parents=True, exist_ok=True)
    {"train": train, "eval": evaluate, "distill": distill}[a.mode](a)
