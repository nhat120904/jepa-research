"""CTA and the matched rerankers on OGBench branched data (docs/CTA_OGBENCH_PROTOCOL.md, "Arms").

Every learned module sees the same banks (8 seeded policy chunks per decision, train split) and the same label (end-
of-chunk cube-target progress); checkpoints are selected on the heldout split by each module's own objective.
Stage 1 (PushT recipe, scripts/cta_train.py stage 1): source encoder + code reader + distortion decoder on actual
  futures; on the same batches the FULL reader (actual future) and DIRECT (actions). Pairwise ranking loss.
Stage 2 (Round-4 task recipe), stage-1 modules frozen:
  CTA       parallel FSQ world model: code NLL + score consistency + weighted ranking through the code reader
  ENDPOINT  end-frame predictor: normalized MSE + score consistency + weighted ranking through the FULL reader
  FRAME     per-frame latent world model: teacher-forced normalized MSE on steps 2-5 + consistency + ranking through
            the FULL reader on its autoregressive rollout
All stage-2 models: same updates, batch, optimizer. Offline ladder on heldout: FULL, CODE, CTA, ENDPOINT, FRAME, DIRECT.
"""
import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ti_wm.codec import pool_grid  # noqa: E402
from ti_wm.contract import require_compute  # noqa: E402
from ti_wm.cta import FutureDecoder, Scorer, SourceEncoder, saturation_penalty  # noqa: E402
from ti_wm.cta_eval import ranking_metrics  # noqa: E402
from ti_wm.cta_ogb import ActEndpointWM, ActParallelFSQWM, FrameWM, action_scorer, fut_from_frames, zeros_prop  # noqa: E402
from ti_wm.cta_parallel import score_consistency, weighted_rank  # noqa: E402
from ti_wm.sibling import rank_loss  # noqa: E402

K = 8


class Split:
    """Encoded split (scripts/ogbench/ogb_encode.py). fut is memory-mapped (N, K, 4, 256, 128) fp16; the rest in RAM."""

    def __init__(self, path, in_ram=False):
        meta = np.load(path / "meta.npz")
        self.root, self.task = meta["root"], meta["task"]
        self.label = torch.from_numpy(meta["prog"][:, :, -1].astype(np.float32))        # end-of-chunk progress
        self.chunk = torch.from_numpy(meta["chunk"].astype(np.float32))
        self.goal_index = torch.from_numpy(meta["goal_index"])
        self.cur = torch.from_numpy(np.load(path / "cur.npy"))
        self.prev = torch.from_numpy(np.load(path / "prev.npy"))
        self.goal = torch.from_numpy(np.load(path / "goal.npy"))
        fut = np.load(path / "fut.npy", mmap_mode=None if in_ram else "r")
        self.fut = fut
        self.n = len(self.root)
        self.spread = np.flatnonzero(np.ptp(meta["prog"][:, :, -1], axis=1) > 1e-3)

    def batch(self, i, device):
        i = np.sort(np.asarray(i))
        ti = torch.from_numpy(i)
        b = len(i)
        ctx = {"cur": self.cur[ti].to(device).repeat_interleave(K, 0),
               "prev": self.prev[ti].to(device).repeat_interleave(K, 0), "prop": zeros_prop(b * K, device)}
        frames = torch.from_numpy(np.asarray(self.fut[i])).to(device).flatten(0, 1)        # (b*K, 4, 256, D)
        goal = self.goal[self.goal_index[ti]].to(device).repeat_interleave(K, 0)
        act = self.chunk[ti].to(device).flatten(0, 1)
        return ctx, frames, goal, act, self.label[ti].to(device)


def build_stage1(cfg, device):
    m = cfg["m"]
    mods = {"enc": SourceEncoder(m), "reader": Scorer("code", m=m), "dec": FutureDecoder(m),
            "full": Scorer("future"), "direct": action_scorer(cfg["direct_layers"])}
    return {k: v.to(device) for k, v in mods.items()}


def schedule(opt, steps, warmup):
    ramp = lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    return torch.optim.lr_scheduler.LambdaLR(opt, ramp)


def copy_norms(split, rng, n=256):
    i = np.sort(rng.choice(split.n, min(n, split.n), replace=False))
    cur = split.cur[torch.from_numpy(i)].float()
    fr = torch.from_numpy(np.asarray(split.fut[i])).float()                               # (n, K, 4, 256, D)
    small = pool_grid(cur, 8)
    seg = torch.stack([pool_grid(fr[:, :, s].flatten(0, 1), 8) for s in range(3)], 1).view(len(i), K, 3, 64, -1)
    return {"end": float(((fr[:, :, 3] - cur[:, None]) ** 2).mean()),
            "seg": float(((seg - small[:, None, None]) ** 2).mean()),
            "frame": float(((fr - cur[:, None, None]) ** 2).mean())}


def stage1(mods, train, cfg, device, amp, rng, norms, log):
    params = [p for k in ("enc", "reader", "dec", "full", "direct") for p in mods[k].parameters()]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = schedule(opt, cfg["steps1"], cfg["warmup"])
    for m in mods.values():
        m.train()
    pool = train.spread if len(train.spread) else np.arange(train.n)
    for step in range(cfg["steps1"]):
        ctx, frames, goal, act, y = train.batch(rng.choice(pool, cfg["decisions"]), device)
        fut = fut_from_frames(frames)
        with amp():
            code, pre = mods["enc"](ctx, fut, return_pre=True)
            s_code = mods["reader"](ctx, code, goal).float().view(-1, K)
            end_hat, seg_hat = mods["dec"](ctx, code)
            l_rec = 0.5 * (((end_hat - fut["end"].float()) ** 2).mean() / norms["end"]
                           + ((seg_hat - fut["seg"].float()) ** 2).mean() / norms["seg"])
            s_full = mods["full"](ctx, fut, goal).float().view(-1, K)
            s_dir = mods["direct"](ctx, act, goal).float().view(-1, K)
        l_code, l_full, l_dir = rank_loss(s_code, y), rank_loss(s_full, y), rank_loss(s_dir, y)
        loss = l_code + cfg["rec"] * l_rec + cfg["sat"] * saturation_penalty(pre) + l_full + l_dir
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            log({"stage": 1, "step": step, "rank_code": float(l_code), "rec": float(l_rec), "rank_full": float(l_full),
                 "rank_direct": float(l_dir)})


@torch.inference_mode()
def source_codes(mods, split, device, amp, batch=8):
    out = []
    for s in range(0, split.n, batch):
        ctx, frames, _, _, _ = split.batch(np.arange(s, min(s + batch, split.n)), device)
        with amp():
            out.append(mods["enc"](ctx, fut_from_frames(frames)).float().cpu())
    return torch.cat(out).view(split.n, K, -1, 3)


def wm_parts(name, net, mods, ctx, frames, goal, act, y, src, norms, cfg, scale):
    """Objective of one stage-2 world model on a batch: prediction term + consistency + weighted ranking."""
    rank_cfg = (cfg["rank_scale"], cfg["rank_margin"])
    if name == "cta":
        expected, logits = net(ctx, act)
        pred = net.nll(logits, src)
        s = mods["reader"](ctx, expected, goal).float().view(-1, K)
        teacher = mods["reader"](ctx, src, goal).float().view(-1, K)
    elif name == "endpoint":
        out = net(ctx, act)
        pred = ((out["end"] - frames[:, 3].float()) ** 2).mean() / norms["end"]
        fut_hat = {"end": out["end"], "prop": out["prop"]}
        s = mods["full"](ctx, fut_hat, goal).float().view(-1, K)
        teacher = mods["full"](ctx, fut_from_frames(frames), goal).float().view(-1, K)
    else:                                                               # frame
        tf = net.rollout(ctx, act, teacher=frames)
        pred = ((tf - frames.float()) ** 2).mean() / norms["frame"]
        roll = net.rollout(ctx, act)
        s = mods["full"](ctx, {"end": roll[:, 3], "prop": zeros_prop(len(roll), roll.device)}, goal).float().view(-1, K)
        teacher = mods["full"](ctx, fut_from_frames(frames), goal).float().view(-1, K)
    parts = {"pred": pred, "consistency": score_consistency(s, teacher, scale),
             "rank": weighted_rank(s, y, *rank_cfg)}
    return parts


@torch.inference_mode()
def scores_on(split, mods, wms, device, amp, batch=8):
    """(N, K) scores of every tier on a split (goal = the decision's own goal image)."""
    names = ["full", "code", "direct"] + list(wms)
    out = {n: np.zeros((split.n, K), np.float32) for n in names}
    for s in range(0, split.n, batch):
        idx = np.arange(s, min(s + batch, split.n))
        ctx, frames, goal, act, _ = split.batch(idx, device)
        fut = fut_from_frames(frames)
        with amp():
            v = {"full": mods["full"](ctx, fut, goal), "code": mods["reader"](ctx, mods["enc"](ctx, fut), goal),
                 "direct": mods["direct"](ctx, act, goal)}
            for n, net in wms.items():
                if n == "cta":
                    v[n] = mods["reader"](ctx, net(ctx, act)[0], goal)
                elif n == "endpoint":
                    o = net(ctx, act)
                    v[n] = mods["full"](ctx, {"end": o["end"], "prop": o["prop"]}, goal)
                else:
                    roll = net.rollout(ctx, act)
                    v[n] = mods["full"](ctx, {"end": roll[:, 3], "prop": zeros_prop(len(roll), roll.device)}, goal)
        for n in names:
            out[n][s:s + len(idx)] = v[n].float().view(len(idx), K).cpu().numpy()
    return out


def ladder(split, scores):
    y = split.label.numpy()
    rep = {}
    for n, s in scores.items():
        m = ranking_metrics(s, y, split.root, ci=True)
        m.pop("chosen")
        rep[n] = m
    return rep


def train(a):
    require_compute()
    device = torch.device("cuda")
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    cfg = {"m": 16, "direct_layers": 8, "lr": 3e-4, "wd": 0.05, "warmup": 500, "steps1": a.steps1, "steps2": a.steps2,
           "decisions": 16, "rec": 1.0, "sat": 0.0, "rank_scale": .01, "rank_margin": .001, "eval_every": a.eval_every,
           "seed": a.seed, "data": str(a.data), "job": os.environ.get("SLURM_JOB_ID")}
    a.run.mkdir(parents=True, exist_ok=True)
    logf = (a.run / "metrics.jsonl").open("a")
    log = lambda d: (logf.write(json.dumps(d) + "\n"), logf.flush(), print(d, flush=True))
    train_s, held = Split(a.data / "train", in_ram=a.in_ram), Split(a.data / "heldout")
    norms = copy_norms(train_s, rng)
    cfg["norms"], cfg["train_decisions"], cfg["heldout_decisions"] = norms, train_s.n, held.n
    (a.run / "config.json").write_text(json.dumps(cfg, indent=2))
    mods = build_stage1(cfg, device)
    t0 = time.perf_counter()
    stage1(mods, train_s, cfg, device, amp, rng, norms, log)
    for m in mods.values():
        m.eval().requires_grad_(False)
    cfg["stage1_hours"] = (time.perf_counter() - t0) / 3600
    src_train, src_held = source_codes(mods, train_s, device, amp), source_codes(mods, held, device, amp)
    # consistency scale from the code reader's within-bank spread on train (as Round 4)
    with torch.no_grad():
        ctx, frames, goal, act, _ = train_s.batch(rng.choice(train_s.n, 64), device)
        with amp():
            t = mods["reader"](ctx, mods["enc"](ctx, fut_from_frames(frames)), goal).float().view(-1, K)
        scale = max(float((t - t.mean(1, keepdim=True)).square().mean().sqrt()), 1e-3)
    wms = {"cta": ActParallelFSQWM(m=cfg["m"]).to(device), "endpoint": ActEndpointWM().to(device),
           "frame": FrameWM().to(device)}
    opts = {n: torch.optim.AdamW(w.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]) for n, w in wms.items()}
    scheds = {n: schedule(o, cfg["steps2"], cfg["warmup"]) for n, o in opts.items()}
    best = {n: (float("inf"), None, None) for n in wms}
    pool = train_s.spread if len(train_s.spread) else np.arange(train_s.n)

    def held_objective():
        tot = {n: 0. for n in wms}
        cnt = 0
        for s in range(0, held.n, 8):
            idx = np.arange(s, min(s + 8, held.n))
            ctx, frames, goal, act, y = held.batch(idx, device)
            src = src_held[torch.from_numpy(idx)].to(device).flatten(0, 1)
            with torch.inference_mode(), amp():
                for n, w in wms.items():
                    parts = wm_parts(n, w, mods, ctx, frames, goal, act, y, src, norms, cfg, scale)
                    tot[n] += float(sum(parts.values())) * len(idx)
            cnt += len(idx)
        return {n: v / cnt for n, v in tot.items()}

    t1 = time.perf_counter()
    for step in range(cfg["steps2"]):
        i = rng.choice(pool, cfg["decisions"])
        ctx, frames, goal, act, y = train_s.batch(i, device)
        src = src_train[torch.from_numpy(np.sort(i))].to(device).flatten(0, 1)
        rec = {"stage": 2, "step": step}
        for n, w in wms.items():
            w.train()
            with amp():
                parts = wm_parts(n, w, mods, ctx, frames, goal, act, y, src, norms, cfg, scale)
                loss = sum(parts.values())
            opts[n].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(w.parameters(), 1.0)
            opts[n].step()
            scheds[n].step()
            rec.update({f"{n}/{k}": float(v) for k, v in parts.items()})
        if step % 500 == 0:
            log(rec)
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps2"]:
            for w in wms.values():
                w.eval()
            obj = held_objective()
            for n, w in wms.items():
                if obj[n] < best[n][0]:
                    best[n] = (obj[n], step + 1, copy.deepcopy({k: v.detach().cpu() for k, v in w.state_dict().items()}))
            log({"stage": 2, "heldout_objective": obj, "step": step + 1})
    for n, w in wms.items():
        w.load_state_dict(best[n][2])
        w.eval().requires_grad_(False)
    cfg["stage2_hours"] = (time.perf_counter() - t1) / 3600
    cfg["selected"] = {n: best[n][1] for n in wms}
    torch.save({"config": cfg, "stage1": {k: m.state_dict() for k, m in mods.items()},
                "wms": {n: w.state_dict() for n, w in wms.items()}, "consistency_scale": scale}, a.run / "cta_ogb.pt")
    sc = scores_on(held, mods, wms, device, amp)
    rep = ladder(held, sc)
    np.savez(a.run / "heldout_scores.npz", root=held.root, label=held.label.numpy(), **sc)
    (a.run / "offline_ladder.json").write_text(json.dumps(rep, indent=2))
    (a.run / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({n: round(r["retained_gap"]["ratio"], 3) for n, r in rep.items()}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--steps1", type=int, default=10_000)
    p.add_argument("--steps2", type=int, default=8_000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--in-ram", action="store_true")
    train(p.parse_args())
