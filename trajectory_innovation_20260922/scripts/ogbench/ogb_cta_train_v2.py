"""CTA v2 on OGBench visual-cube: multi-goal readers, held-out model selection, dropout.

Same data, modules, arms and checkpoint format as ogb_cta_train.py (v1, job 55565). Each change targets a failure seen
in 55565's offline ladder (held-out retained gap FULL .80 < DIRECT .94, CODE .75, CTA .91, CTA's world model best at
2k of 8k updates):
  1. Readers memorized the 4.7k informative banks (FULL train rank loss .003). Every scorer is now trained on several
     goals per bank: the official task goals and hindsight goals (later decision frames of the same collection, whose
     cube position is known). Labels are the environment's own progress for that target: 1 inside the 4 cm success
     radius, else minus the cube-target distance (identical to the collected label for the episode's own goal).
  2. Stage 1 had no model selection. Each group (source encoder + code reader + decoder, FULL, DIRECT) and each world
     model is selected on the held-out split by its retained gap, averaged over the training goals.
  3. Dropout in every Transformer (same rate for every module).
  4. A single-goal ranking target lets the code collapse to a scalar. Ranking for many targets needs the code to say
     where the cube ends up. Code usage (distinct values per code, perplexity, sibling distinctness) is reported.
  5. --heldout-goals (e.g. 4,5): those official goals, and hindsight goals within --exclude-radius of them, never enter
     training or selection. The ladder then reports ranking for goals no scorer was trained on ("query many").
Compute node only.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
from ogb_cta_train import Split, copy_norms, schedule  # noqa: E402
from ti_wm.contract import require_compute  # noqa: E402
from ti_wm.cta import LEVELS, FutureDecoder, Scorer, SourceEncoder, nested_drop, saturation_penalty, vocab_size  # noqa: E402
from ti_wm.cta_eval import bank_distinct, perplexity, ranking_metrics  # noqa: E402
from ti_wm.cta_ogb import ActEndpointWM, ActParallelFSQWM, FrameWM, action_scorer, fut_from_frames, zeros_prop  # noqa: E402
from ti_wm.cta_parallel import score_consistency, weighted_rank  # noqa: E402
from ti_wm.sibling import rank_loss  # noqa: E402

K = 8
SUCCESS_RADIUS = 0.04
# cube-single official task goals (ogbench/manipspace/envs/cube_env.py, task_infos goal_xyzs; no reset noise)
GOAL_XYZ = np.array([[0.425, -0.1, 0.02], [0.50, 0.0, 0.02], [0.35, 0.0, 0.02], [0.50, 0.2, 0.02], [0.50, -0.2, 0.02]],
                    np.float32)


def progress_label(cubes, target):
    """Environment progress of a cube position for a target (numpy or torch): 1 within 4 cm, else -distance (m)."""
    if isinstance(cubes, torch.Tensor):
        d = (cubes - target).norm(dim=-1)
        return torch.where(d <= SUCCESS_RADIUS, torch.ones_like(d), -d)
    d = np.linalg.norm(cubes - target, axis=-1)
    return np.where(d <= SUCCESS_RADIUS, 1.0, -d).astype(np.float32)


class Data:
    """An encoded split plus end-of-chunk cube positions, the official goal images, and (train) the hindsight pool."""

    def __init__(self, path, in_ram=False):
        self.s = Split(path, in_ram)
        meta = np.load(path / "meta.npz")
        self.cubes = meta["cubes"][:, :, 0, :].astype(np.float32)                 # (N, K, 3) end of each chunk
        self.task = meta["task"]
        self.n = self.s.n
        own = progress_label(self.cubes, GOAL_XYZ[self.task - 1][:, None, :])
        if np.abs(own - self.s.label.numpy()).max() > 1e-5:
            raise ValueError("recomputed progress does not match the collected label")
        # one goal image per official task: that task's first episode in this split
        self.goal_tok = {int(t): self.s.goal[int(self.s.goal_index[np.flatnonzero(self.task == t)[0]])]
                         for t in np.unique(self.task)}
        # hindsight goals: the frame of decision i+1 (same episode) with its cube position = end of the executed chunk
        root, d, ex = meta["root"], meta["d"], meta["executed"]
        i = np.flatnonzero((root[1:] == root[:-1]) & (d[1:] == d[:-1] + 1))
        pos = self.cubes[i, ex[i]]
        still = np.ptp(self.cubes[i + 1], axis=1).max(-1) < 1e-6                     # checkable: nothing moves the cube
        bad = still & (np.abs(self.cubes[i + 1, 0] - pos).max(-1) > 1e-3)
        self.hind_frame, self.hind_pos = i[~bad] + 1, pos[~bad]
        self.hind_dropped = int(bad.sum())

    def restrict_hindsight(self, excluded_goals, radius):
        if not excluded_goals:
            return 0
        xy = GOAL_XYZ[np.asarray(excluded_goals) - 1, :2]
        near = (np.linalg.norm(self.hind_pos[:, None, :2] - xy[None], axis=-1) < radius).any(1)
        self.hind_frame, self.hind_pos = self.hind_frame[~near], self.hind_pos[~near]
        return int(near.sum())


def expand(x, b, g):
    """Rows ordered (bank, candidate) -> (bank, goal, candidate): repeat each bank's K rows for its g goals."""
    r = torch.arange(b * K, device=x["cur"].device if isinstance(x, dict) else x.device).view(b, 1, K).expand(b, g, K).reshape(-1)
    return {k: v[r] for k, v in x.items()} if isinstance(x, dict) else x[r]


def sample_batch(data, rng, cfg, device):
    """B banks (mostly informative ones) and G goals per bank. Returns tensors in the order of Split.batch (sorted)."""
    b, g = cfg["decisions"], cfg["goals_per_bank"]
    n_sp = int(round(cfg["spread_frac"] * b))
    i = np.concatenate([rng.choice(data.s.spread, n_sp), rng.choice(data.n, b - n_sp)])
    i = np.sort(i)
    ctx, frames, _, act, _ = data.s.batch(i, device)
    official = rng.random((b, g)) < cfg["p_official"]
    goal_j = rng.choice(cfg["train_goals"], (b, g))
    hind = rng.integers(0, len(data.hind_frame), (b, g))
    toks, pos = [], np.zeros((b, g, 3), np.float32)
    for bi in range(b):
        for gi in range(g):
            if official[bi, gi]:
                toks.append(data.goal_tok[int(goal_j[bi, gi])])
                pos[bi, gi] = GOAL_XYZ[goal_j[bi, gi] - 1]
            else:
                toks.append(data.s.cur[int(data.hind_frame[hind[bi, gi]])])
                pos[bi, gi] = data.hind_pos[hind[bi, gi]]
    goal = torch.stack(toks).to(device).repeat_interleave(K, 0)                    # (B*G*K, 256, D)
    labels = torch.from_numpy(progress_label(data.cubes[i][:, None], pos[:, :, None])).to(device).view(b * g, K)
    return i, ctx, frames, act, goal, labels


def levels_of(cfg):
    return tuple(cfg.get("levels") or LEVELS)


def build_stage1(cfg, device):
    m, p, lv = cfg["m"], cfg.get("dropout", 0.0), levels_of(cfg)
    mods = {"enc": SourceEncoder(m, levels=lv, dropout=p), "reader": Scorer("code", m=m, levels=lv, dropout=p),
            "dec": FutureDecoder(m, levels=lv, dropout=p), "full": Scorer("future", dropout=p),
            "direct": action_scorer(cfg["direct_layers"], dropout=p)}
    return {k: v.to(device) for k, v in mods.items()}


def build_wms(cfg, device):
    p = cfg.get("dropout", 0.0)
    return {"cta": ActParallelFSQWM(m=cfg["m"], levels=levels_of(cfg), dropout=p).to(device),
            "endpoint": ActEndpointWM(dropout=p).to(device), "frame": FrameWM(dropout=p).to(device)}


GROUPS = {"codec": ("enc", "reader", "dec"), "full": ("full",), "direct": ("direct",)}
TEACHER = {"cta": "reader", "endpoint": "full", "frame": "full"}    # the frozen reader each world model is read by
GROUP_TIER = {"codec": "code", "full": "full", "direct": "direct"}


# ---------------------------------------------------------------------------------------------------- evaluation
@torch.inference_mode()
def evidence(tier, mods, wms, ctx, frames, act):
    """Per-candidate evidence and the reader that reads it."""
    if tier == "full":
        return fut_from_frames(frames), mods["full"]
    if tier == "code":
        return mods["enc"](ctx, fut_from_frames(frames)), mods["reader"]
    if tier == "direct":
        return act, mods["direct"]
    if tier == "cta":
        return wms["cta"](ctx, act)[0], mods["reader"]
    if tier == "endpoint":
        o = wms["endpoint"](ctx, act)
        return {"end": o["end"], "prop": o["prop"]}, mods["full"]
    if tier == "frame":
        roll = wms["frame"].rollout(ctx, act)
        return {"end": roll[:, 3], "prop": zeros_prop(len(roll), roll.device)}, mods["full"]
    raise ValueError(tier)


@torch.inference_mode()
def score_split(data, tiers, goal_names, mods, wms, device, amp, batch=16):
    """{(tier, goal): (N, K)} scores. goal_names: 'own' (each decision's own episode goal image) or official ids."""
    out = {(t, gname): np.zeros((data.n, K), np.float32) for t in tiers for gname in goal_names}
    for s in range(0, data.n, batch):
        idx = np.arange(s, min(s + batch, data.n))
        ctx, frames, own_goal, act, _ = data.s.batch(idx, device)
        with amp():
            for t in tiers:
                ev, reader = evidence(t, mods, wms, ctx, frames, act)
                for gname in goal_names:
                    goal = own_goal if gname == "own" else data.goal_tok[int(gname)].to(device)[None].expand(len(ctx["cur"]), -1, -1)
                    out[(t, gname)][s:s + len(idx)] = reader(ctx, ev, goal).float().view(len(idx), K).cpu().numpy()
    return out


def goal_labels(data, gname):
    return data.s.label.numpy() if gname == "own" else progress_label(data.cubes, GOAL_XYZ[int(gname) - 1][None, None])


def gaps(data, scores, tiers, goal_names, ci=False):
    rep = {}
    for t in tiers:
        rep[t] = {}
        for gname in goal_names:
            m = ranking_metrics(scores[(t, gname)], goal_labels(data, gname), data.s.root, ci=ci)
            m.pop("chosen")
            rep[t][str(gname)] = m
    return rep


def mean_gap(rep, tier, goal_names):
    v = float(np.nanmean([rep[tier][str(gname)]["retained_gap"]["ratio"] for gname in goal_names]))
    return v if math.isfinite(v) else -float("inf")


# ---------------------------------------------------------------------------------------------------- training
def stage1(mods, train, held, cfg, device, amp, rng, norms, log):
    params = [p for m in mods.values() for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = schedule(opt, cfg["steps1"], cfg["warmup"])
    best = {g: (-float("inf"), None, None) for g in GROUPS}
    g_n = cfg["goals_per_bank"]
    for step in range(cfg["steps1"]):
        for m in mods.values():
            m.train()
        i, ctx, frames, act, goal, y = sample_batch(train, rng, cfg, device)
        fut = fut_from_frames(frames)
        b = len(i)
        with amp():
            code, pre = mods["enc"](ctx, fut, return_pre=True)
            drop = None
            if cfg.get("nested", 0.0) > 0:          # nested dropout on a share of rows (the rest see the whole code)
                drop = nested_drop(len(code), code.shape[1], code.device)
                drop &= (torch.rand(len(code), 1, device=code.device) < cfg["nested"])
            end_hat, seg_hat = mods["dec"](ctx, code, drop)
            l_rec = 0.5 * (((end_hat - fut["end"].float()) ** 2).mean() / norms["end"]
                           + ((seg_hat - fut["seg"].float()) ** 2).mean() / norms["seg"])
            ctx_r = expand(ctx, b, g_n)
            s_code = mods["reader"](ctx_r, expand(code, b, g_n), goal,
                                    None if drop is None else expand(drop, b, g_n)).float().view(-1, K)
            s_full = mods["full"](ctx_r, {"end": expand(fut["end"], b, g_n), "prop": expand(fut["prop"], b, g_n)},
                                  goal).float().view(-1, K)
            s_dir = mods["direct"](ctx_r, expand(act, b, g_n), goal).float().view(-1, K)
        l_code, l_full, l_dir = rank_loss(s_code, y), rank_loss(s_full, y), rank_loss(s_dir, y)
        loss = l_code + cfg["rec"] * l_rec + cfg["sat"] * saturation_penalty(pre) + l_full + l_dir
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            idx = mods["enc"].fsq.codes_to_indices(code.detach()).view(b, K, -1).cpu().numpy()
            log({"stage": 1, "step": step, "rank_code": float(l_code), "rec": float(l_rec), "rank_full": float(l_full),
                 "rank_direct": float(l_dir), "distinct_per_code": code_distinct(idx),
                 "saturated_frac": float((pre.detach().abs() > 1.5).float().mean())})
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps1"]:
            for m in mods.values():
                m.eval()
            sc = score_split(held, ("code", "full", "direct"), cfg["train_goals"], mods, {}, device, amp)
            rep = gaps(held, sc, ("code", "full", "direct"), cfg["train_goals"])
            sel = {g: mean_gap(rep, GROUP_TIER[g], cfg["train_goals"]) for g in GROUPS}
            for g, v in sel.items():
                if v > best[g][0] or best[g][2] is None:
                    best[g] = (v, step + 1, {n: copy.deepcopy({k: t.detach().cpu() for k, t in mods[n].state_dict().items()})
                                              for n in GROUPS[g]})
            log({"stage": 1, "step": step + 1, "heldout_mean_gap": sel})
    for g, (_, _, states) in best.items():
        for n, st in states.items():
            mods[n].load_state_dict(st)
    return {g: {"gap": best[g][0], "step": best[g][1]} for g in GROUPS}


@torch.inference_mode()
def source_codes(mods, data, device, amp, batch=8):
    out = []
    for s in range(0, data.n, batch):
        ctx, frames, _, _, _ = data.s.batch(np.arange(s, min(s + batch, data.n)), device)
        with amp():
            out.append(mods["enc"](ctx, fut_from_frames(frames)).float().cpu())
    return torch.cat(out).view(data.n, K, -1, 3)


def wm_parts(name, net, mods, ctx, frames, act, goal, y, src, norms, cfg, scale, b):
    g_n = cfg["goals_per_bank"]
    rank_cfg = (cfg["rank_scale"], cfg["rank_margin"])
    ctx_r = expand(ctx, b, g_n)
    if name == "cta":
        expected, logits = net(ctx, act)
        pred = net.nll(logits, src)
        s = mods["reader"](ctx_r, expand(expected, b, g_n), goal).float().view(-1, K)
        with torch.no_grad():
            teacher = mods["reader"](ctx_r, expand(src, b, g_n), goal).float().view(-1, K)
    else:
        fut = fut_from_frames(frames)
        if name == "endpoint":
            out = net(ctx, act)
            pred = ((out["end"] - frames[:, 3].float()) ** 2).mean() / norms["end"]
            end, prop = out["end"], out["prop"]
        else:
            tf = net.rollout(ctx, act, teacher=frames)
            pred = ((tf - frames.float()) ** 2).mean() / norms["frame"]
            roll = net.rollout(ctx, act)
            end, prop = roll[:, 3], zeros_prop(len(roll), roll.device)
        s = mods["full"](ctx_r, {"end": expand(end, b, g_n), "prop": expand(prop, b, g_n)}, goal).float().view(-1, K)
        with torch.no_grad():
            teacher = mods["full"](ctx_r, {"end": expand(fut["end"], b, g_n), "prop": expand(fut["prop"], b, g_n)},
                                   goal).float().view(-1, K)
    return {"pred": pred, "consistency": score_consistency(s, teacher, scale[TEACHER[name]]),
            "rank": weighted_rank(s, y, *rank_cfg)}


def stage2(mods, wms, train, held, src_train, cfg, device, amp, rng, norms, scale, log):
    opts = {n: torch.optim.AdamW(w.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]) for n, w in wms.items()}
    scheds = {n: schedule(o, cfg["steps2"], cfg["warmup"]) for n, o in opts.items()}
    best = {n: (-float("inf"), None, None) for n in wms}
    for step in range(cfg["steps2"]):
        i, ctx, frames, act, goal, y = sample_batch(train, rng, cfg, device)
        src = src_train[torch.from_numpy(i)].to(device).flatten(0, 1)
        rec = {"stage": 2, "step": step}
        for n, w in wms.items():
            w.train()
            with amp():
                parts = wm_parts(n, w, mods, ctx, frames, act, goal, y, src, norms, cfg, scale, len(i))
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
            sc = score_split(held, tuple(wms), cfg["train_goals"], mods, wms, device, amp)
            rep = gaps(held, sc, tuple(wms), cfg["train_goals"])
            sel = {n: mean_gap(rep, n, cfg["train_goals"]) for n in wms}
            for n, v in sel.items():
                if v > best[n][0] or best[n][2] is None:
                    best[n] = (v, step + 1, copy.deepcopy({k: t.detach().cpu() for k, t in wms[n].state_dict().items()}))
            log({"stage": 2, "step": step + 1, "heldout_mean_gap": sel})
    for n, w in wms.items():
        w.load_state_dict(best[n][2])
    return {n: {"gap": best[n][0], "step": best[n][1]} for n in wms}


# ---------------------------------------------------------------------------------------------------- diagnostics
def code_distinct(idx):
    """Mean number of distinct token values inside one code. idx (..., M)."""
    flat = idx.reshape(-1, idx.shape[-1])
    return float(np.mean([len(np.unique(r)) for r in flat]))


@torch.inference_mode()
def code_report(mods, wms, data, src, device, amp, batch=8):
    """Usage of the source code and agreement of the CTA world model's most likely code with it (held-out split)."""
    fsq = mods["enc"].fsq
    idx = fsq.codes_to_indices(src.to(device)).cpu().numpy()                      # (N, K, M)
    agree, n = 0., 0
    net = wms["cta"]
    for s in range(0, data.n, batch):
        sl = np.arange(s, min(s + batch, data.n))
        ctx, _, _, act, _ = data.s.batch(sl, device)
        with amp():
            _, logits = net(ctx, act)
        digits = torch.stack([p.argmax(-1) for p in logits], -1)                  # (B*K, M, 3)
        code = torch.stack([(digits[..., j].float() - c // 2) / (c // 2) for j, c in enumerate(net.levels)], -1)
        pred = fsq.codes_to_indices(code).cpu().numpy().reshape(len(sl), K, -1)
        agree += float((pred == idx[sl]).mean()) * len(sl)
        n += len(sl)
    m = idx.shape[-1]
    return {"distinct_values_per_code": code_distinct(idx),
            "perplexity_per_position": perplexity(idx.reshape(-1, m), vocab_size(tuple(net.levels))),
            "indices_used": int(len(np.unique(idx))), "sibling_codes_distinct": bank_distinct(idx),
            "levels": list(net.levels),
            "cta_argmax_token_agreement": agree / max(n, 1)}


# ---------------------------------------------------------------------------------------------------- main
def train(a):
    require_compute()
    device = torch.device("cuda")
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    held_out = [int(x) for x in a.heldout_goals.split(",")] if a.heldout_goals else []
    train_goals = [j for j in range(1, 6) if j not in held_out]
    cfg = {"version": 2, "m": a.m, "levels": [int(x) for x in a.levels.split(",")], "direct_layers": 8, "lr": 3e-4, "wd": 0.05, "warmup": 500, "dropout": a.dropout,
           "steps1": a.steps1, "steps2": a.steps2, "decisions": a.decisions, "goals_per_bank": a.goals_per_bank,
           "spread_frac": .75, "p_official": a.p_official, "rec": 1.0, "sat": a.sat, "rank_scale": .01, "rank_margin": .001,
           "eval_every": a.eval_every, "seed": a.seed, "data": str(a.data), "train_goals": train_goals,
           "heldout_goals": held_out, "exclude_radius": a.exclude_radius, "nested": a.nested, "job": os.environ.get("SLURM_JOB_ID")}
    a.run.mkdir(parents=True, exist_ok=True)
    logf = (a.run / "metrics.jsonl").open("a")
    log = lambda d: (logf.write(json.dumps(d) + "\n"), logf.flush(), print(d, flush=True))
    t0 = time.perf_counter()
    train_d, held = Data(a.data / "train", in_ram=a.in_ram), Data(a.data / "heldout")
    if a.limit:
        keep = np.sort(np.random.default_rng(1).choice(train_d.n, min(a.limit, train_d.n), replace=False))
        train_d.s.spread = np.intersect1d(train_d.s.spread, keep)
        train_d.s.n = train_d.n = int(keep.max()) + 1
    cfg["hindsight"] = {"pool": len(train_d.hind_frame), "dropped_inconsistent": train_d.hind_dropped,
                        "excluded_near_heldout_goals": train_d.restrict_hindsight(held_out, a.exclude_radius)}
    cfg["hindsight"]["pool_after_exclusion"] = len(train_d.hind_frame)
    norms = copy_norms(train_d.s, rng)
    cfg.update(norms=norms, train_decisions=train_d.n, heldout_decisions=held.n, load_seconds=time.perf_counter() - t0)
    (a.run / "config.json").write_text(json.dumps(cfg, indent=2))
    log({"config": cfg})
    mods = build_stage1(cfg, device)
    t1 = time.perf_counter()
    cfg["selected_stage1"] = stage1(mods, train_d, held, cfg, device, amp, rng, norms, log)
    cfg["stage1_hours"] = (time.perf_counter() - t1) / 3600
    for m in mods.values():
        m.eval().requires_grad_(False)
    src_train, src_held = source_codes(mods, train_d, device, amp), source_codes(mods, held, device, amp)
    # consistency normalizer of each frozen reader: its within-bank score spread on actual futures (train batches)
    spread = {"reader": [], "full": []}
    with torch.no_grad():
        for _ in range(8):
            i, ctx, frames, act, goal, y = sample_batch(train_d, rng, cfg, device)
            g_n, fut = cfg["goals_per_bank"], fut_from_frames(frames)
            with amp():
                t_code = mods["reader"](expand(ctx, len(i), g_n),
                                        expand(src_train[torch.from_numpy(i)].to(device).flatten(0, 1), len(i), g_n), goal)
                t_full = mods["full"](expand(ctx, len(i), g_n),
                                      {"end": expand(fut["end"], len(i), g_n), "prop": expand(fut["prop"], len(i), g_n)}, goal)
            for n, t in (("reader", t_code), ("full", t_full)):
                t = t.float().view(-1, K)
                spread[n].append(float((t - t.mean(1, keepdim=True)).square().mean()))
    scale = {n: max(float(np.sqrt(np.mean(v))), 1e-3) for n, v in spread.items()}
    cfg["consistency_scale"] = scale
    wms = build_wms(cfg, device)
    t2 = time.perf_counter()
    cfg["selected_stage2"] = stage2(mods, wms, train_d, held, src_train, cfg, device, amp, rng, norms, scale, log)
    cfg["stage2_hours"] = (time.perf_counter() - t2) / 3600
    for w in wms.values():
        w.eval().requires_grad_(False)
    torch.save({"config": cfg, "stage1": {k: m.state_dict() for k, m in mods.items()},
                "wms": {n: w.state_dict() for n, w in wms.items()}, "consistency_scale": scale}, a.run / "cta_ogb.pt")
    tiers = ("full", "code", "direct", "cta", "endpoint", "frame")
    goal_names = ["own"] + list(range(1, 6))
    sc = score_split(held, tiers, goal_names, mods, wms, device, amp)
    np.savez(a.run / "heldout_scores.npz", root=held.s.root, task=held.task, cubes=held.cubes,
             **{f"{t}__{g}": v for (t, g), v in sc.items()})
    rep = gaps(held, sc, tiers, goal_names, ci=True)
    summary = {t: {"own": rep[t]["own"]["retained_gap"]["ratio"],
                   "train_goals": mean_gap(rep, t, train_goals),
                   **({"heldout_goals": mean_gap(rep, t, held_out)} if held_out else {})} for t in tiers}
    if held_out:   # own goal restricted to the held-out tasks' episodes: the deployment case for a new goal
        mask = np.isin(held.task, held_out)
        for t in tiers:
            m = ranking_metrics(sc[(t, "own")][mask], held.s.label.numpy()[mask], held.s.root[mask], ci=True)
            m.pop("chosen")
            rep[t]["own_on_heldout_tasks"] = m
            summary[t]["own_on_heldout_tasks"] = m["retained_gap"]["ratio"]
    report = {"summary_retained_gap": summary, "ladder": rep, "codes": code_report(mods, wms, held, src_held, device, amp),
              "config": cfg}
    (a.run / "offline_ladder.json").write_text(json.dumps(report, indent=2))
    (a.run / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({"summary": summary, "codes": report["codes"]}, indent=1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--steps1", type=int, default=8000)
    p.add_argument("--steps2", type=int, default=6000)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--decisions", type=int, default=16)
    p.add_argument("--goals-per-bank", type=int, default=2)
    p.add_argument("--p-official", type=float, default=0.5)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--m", type=int, default=16, help="code tokens")
    p.add_argument("--nested", type=float, default=0.0, help="share of stage-1 rows trained with nested dropout")
    p.add_argument("--levels", default="8,8,4", help="FSQ levels per code token (8,8,4 = 8 bits, v1/v2 default)")
    p.add_argument("--sat", type=float, default=1.0, help="FSQ saturation penalty weight (0 in v1 -> code collapse)")
    p.add_argument("--heldout-goals", default="")
    p.add_argument("--exclude-radius", type=float, default=0.08)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--in-ram", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="smoke only: first-N-ish train banks")
    train(p.parse_args())
