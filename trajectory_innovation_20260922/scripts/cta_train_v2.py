"""CTA v2 on PushT: every module retrained from scratch on all collected banks, with held-out selection and dropout.

The PushT CTA world models so far (Rounds 3-6, on-policy continuation) were all trained through ONE source encoder and
code reader, checkpoint 55018, trained on the 22.7k Round-0 banks (roots 30250-31049) and frozen since. Its readers
never saw the 1,200 Round-4 episodes or the CTA-visited on-policy states, stage 1 had no model selection, and the
16-token code collapsed to ~1.8 distinct values. v2 is the from-scratch recipe the paper needs (same architecture as
scripts/cta_train.py + scripts/cta_round4.py):
  data     Round-0 train banks, Round-4 standard banks (full tokens: scripts/cta_encode_r4_full.py), on-policy train
           banks; optionally the Round-4 perturbed banks (--with-perturbed). Geometry (registration) label, original PCA.
  stage 1  source encoder + code reader + distortion decoder, FULL reader, DIRECT; pairwise ranking + reconstruction;
           dropout; each group selected by its retained gap on the selection banks.
  stage 2  CTA (parallel FSQ world model: code NLL + consistency + weighted ranking through the frozen code reader) and
           ENDPOINT (end-frame + proprio predictor read by the frozen FULL reader); selected the same way.
Selection banks: Round-0 dev roots 2000-2099 and on-policy selection roots 32650-32699 (disjoint from every closed-loop
root range: 2100-2399 development, 3000+ sealed). Scores average the goal images as at deployment.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ti_wm.contract import require_compute  # noqa: E402
from ti_wm.cta import (FutureDecoder, Scorer, SourceEncoder, action_features, goal_scores, proprio,  # noqa: E402
                       saturation_penalty, vocab_size)
from ti_wm.cta_eval import bank_distinct, perplexity, ranking_metrics  # noqa: E402
from ti_wm.cta_parallel import EndpointWM, ParallelFSQWM, score_consistency, weighted_rank  # noqa: E402
from ti_wm.sibling import RANK_MARGIN, rank_loss  # noqa: E402

K = 8
TI = Path("/mnt/data/nhatnc129/jepa/trajectory_innovation")
SOURCES = {"r0": TI / "cta_geometry_e2e_55018/features/train",
           "onpolicy": TI / "cta_onpolicy8_encode_55347/features/train"}
SELECTION = {"r0dev": TI / "cta_geometry_e2e_55018/features/dev",
             "onpolicydev": TI / "cta_onpolicy8_encode_55347/features/dev"}
GOALS = TI / "cta_geometry_e2e_55018/features/goals.npy"
SELECT_GOALS = (0, 5, 10, 15)         # goal images averaged during selection (all 16 in the final ladder)


class Bank:
    """One feature cache (scripts/cta_encode.py format), tokens memory-mapped."""

    def __init__(self, path, name, aux=None, success_bonus=0.0):
        meta = np.load(path / "meta.npz")
        self.name, self.n = name, len(meta["root"])
        self.root = meta["root"]
        self.geom = meta["cov8"].astype(np.float32)                       # registration (geometry) label: evaluation
        self.label = self.geom                                            # training label
        self.aux = None
        if aux is not None:                                               # v2.1 (scripts/cta_v2_aux.py)
            z = np.load(aux)
            self.aux = {k: z[k] for k in z.files}
            if len(self.aux["end_pose"]) != self.n:
                raise ValueError(f"{name}: aux rows {len(self.aux['end_pose'])} != {self.n}")
            if success_bonus:
                self.label = (self.geom + success_bonus * self.aux["success"]).astype(np.float32)
        self.ctx_pos = torch.from_numpy(meta["ctx_pos"]).float()
        self.chunk = torch.from_numpy(meta["chunk"]).float()
        self.end_prop = proprio(torch.from_numpy(meta["end_pos"]), torch.from_numpy(meta["end_prev_pos"]))
        self.mm = {k: np.load(path / f"{k}.npy", mmap_mode="r") for k in ("cur", "prev", "end", "seg")}
        self.spread = np.flatnonzero(np.ptp(self.label, axis=1) > RANK_MARGIN)

    def batch(self, i, device):
        i = np.asarray(i)
        take = lambda k: torch.from_numpy(np.asarray(self.mm[k][i]))
        ctx = {"cur": take("cur").to(device).repeat_interleave(K, 0), "prev": take("prev").to(device).repeat_interleave(K, 0),
               "prop": proprio(self.ctx_pos[i, 0], self.ctx_pos[i, 1]).to(device).repeat_interleave(K, 0)}
        fut = {"end": take("end").flatten(0, 1).to(device), "seg": take("seg").flatten(0, 1).to(device),
               "prop": self.end_prop[i].flatten(0, 1).to(device)}
        act = action_features(self.chunk[i], self.ctx_pos[i, 0][:, None]).flatten(0, 1).to(device)
        return ctx, fut, act, torch.from_numpy(self.label[i]).to(device)


class Pool:
    """Several banks sampled as one training set: `spread_frac` of each batch from informative banks."""

    def __init__(self, banks):
        self.banks = banks
        self.all = np.concatenate([np.stack([np.full(b.n, j), np.arange(b.n)], 1) for j, b in enumerate(banks)])
        self.inf = np.concatenate([np.stack([np.full(len(b.spread), j), b.spread], 1) for j, b in enumerate(banks)])
        hind = [(j, f, p) for j, b in enumerate(banks) if b.aux is not None and b.aux["hind_frame"].max(initial=-1) < b.n
                for f, p in zip(b.aux["hind_frame"], b.aux["hind_pose"])]
        self.hind = hind

    def hindsight(self, rng, rows, goal, y, device):
        """v2.1: replace the goal of the rows flagged in `rows` by hindsight goals (a decision frame with a known block
        pose); their labels become the registration score of each candidate's end pose to that pose."""
        from ti_wm.cta_geometry import registration_score
        for r, (j, i) in rows:
            bj, frame, pose = self.hind[int(rng.integers(0, len(self.hind)))]
            goal[r] = torch.from_numpy(np.asarray(self.banks[bj].mm["cur"][frame])).to(device)
            y[r] = torch.from_numpy(registration_score(self.banks[j].aux["end_pose"][i], goal=pose).astype(np.float32)).to(device)
        return goal, y

    def sample(self, rng, b, spread_frac, device, return_rows=False):
        n_sp = int(round(spread_frac * b))
        rows = np.concatenate([self.inf[rng.integers(0, len(self.inf), n_sp)], self.all[rng.integers(0, len(self.all), b - n_sp)]])
        rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
        parts = [self.banks[j].batch(rows[rows[:, 0] == j, 1], device) for j in np.unique(rows[:, 0])]
        cat = lambda xs: {k: torch.cat([x[k] for x in xs]) for k in xs[0]} if isinstance(xs[0], dict) else torch.cat(xs)
        out = tuple(cat([p[t] for p in parts]) for t in range(4))
        return (*out, rows) if return_rows else out


def train_batch(pool, goals, rng, cfg, device):
    """Banks, their training labels and one goal image per bank (v2.1: a share of hindsight goals)."""
    ctx, fut, act, y, rows = pool.sample(rng, cfg["decisions"], cfg["spread_frac"], device, return_rows=True)
    goal = goals[torch.as_tensor(rng.integers(0, len(goals), len(y)), device=device)]
    if cfg.get("hindsight", 0.0) > 0:
        flag = [(r, tuple(rows[r])) for r in np.flatnonzero(rng.random(len(y)) < cfg["hindsight"])]
        goal, y = pool.hindsight(rng, flag, goal.clone(), y.clone(), device)
    return ctx, fut, act, y, goal.repeat_interleave(K, 0)


def copy_norms(pool, rng, n=256):
    ctx, fut, _, _ = pool.sample(rng, n // K, 0.0, "cpu")
    cur = ctx["cur"].float()
    from ti_wm.codec import pool_grid
    small = pool_grid(cur, 8)
    return {"end": float(((fut["end"].float() - cur) ** 2).mean()),
            "seg": float(((fut["seg"].float() - small[:, None]) ** 2).mean())}


def schedule(opt, steps, warmup):
    ramp = lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    return torch.optim.lr_scheduler.LambdaLR(opt, ramp)


def build_stage1(cfg, device):
    m, p = cfg["m"], cfg["dropout"]
    mods = {"enc": SourceEncoder(m, dropout=p), "reader": Scorer("code", m=m, dropout=p), "dec": FutureDecoder(m, dropout=p),
            "full": Scorer("future", dropout=p),
            "direct": Scorer("action", layers=cfg["direct_layers"], dropout=p, chunk=cfg.get("chunk", 8))}
    return {k: v.to(device) for k, v in mods.items()}


def build_wms(cfg, device):
    chunk = cfg.get("chunk", 8)
    return {"cta": ParallelFSQWM(m=cfg["m"], dropout=cfg["dropout"], chunk=chunk).to(device),
            "endpoint": EndpointWM(dropout=cfg["dropout"], chunk=chunk).to(device)}


GROUPS = {"codec": ("enc", "reader", "dec"), "full": ("full",), "direct": ("direct",)}
GROUP_TIER = {"codec": "code", "full": "full", "direct": "direct"}
TEACHER = {"cta": "reader", "endpoint": "full"}


@torch.inference_mode()
def score_banks(banks, tiers, mods, wms, goals, device, amp, limit=None, batch=4):
    """{tier: [(N_b, K) per bank]} with scores averaged over `goals` (as deployed)."""
    out = {t: [] for t in tiers}
    for bank in banks:
        n = bank.n if limit is None else min(limit, bank.n)
        res = {t: np.zeros((n, K), np.float32) for t in tiers}
        for s in range(0, n, batch):
            i = np.arange(s, min(s + batch, n))
            ctx, fut, act, _ = bank.batch(i, device)
            with amp():
                for t in tiers:
                    if t == "full":
                        v = goal_scores(mods["full"], ctx, {"end": fut["end"], "prop": fut["prop"]}, goals)
                    elif t == "code":
                        v = goal_scores(mods["reader"], ctx, mods["enc"](ctx, fut), goals)
                    elif t == "direct":
                        v = goal_scores(mods["direct"], ctx, act, goals)
                    elif t == "cta":
                        v = goal_scores(mods["reader"], ctx, wms["cta"](ctx, act)[0], goals)
                    elif t == "endpoint":
                        v = goal_scores(mods["full"], ctx, wms["endpoint"](ctx, act), goals)
                    else:
                        raise ValueError(t)
                    res[t][s:s + len(i)] = v.view(len(i), K).cpu().numpy()
        for t in tiers:
            out[t].append(res[t])
    return out


def pooled_gap(banks, scores, tier, limit=None, ci=False):
    lab = np.concatenate([b.geom[: (b.n if limit is None else min(limit, b.n))] for b in banks])
    root = np.concatenate([b.root[: (b.n if limit is None else min(limit, b.n))] for b in banks])
    m = ranking_metrics(np.concatenate(scores[tier]), lab, root, ci=ci)
    m.pop("chosen")
    return m


def stage1(mods, pool, sel, goals, cfg, device, amp, rng, norms, log):
    params = [p for m in mods.values() for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["wd"])
    sched = schedule(opt, cfg["steps1"], cfg["warmup"])
    best = {g: (-float("inf"), None, None) for g in GROUPS}
    sel_goals = goals[list(SELECT_GOALS)]
    for step in range(cfg["steps1"]):
        for m in mods.values():
            m.train()
        ctx, fut, act, y, goal = train_batch(pool, goals, rng, cfg, device)
        with amp():
            code, pre = mods["enc"](ctx, fut, return_pre=True)
            end_hat, seg_hat = mods["dec"](ctx, code)
            l_rec = 0.5 * (((end_hat - fut["end"].float()) ** 2).mean() / norms["end"]
                           + ((seg_hat - fut["seg"].float()) ** 2).mean() / norms["seg"])
            s_code = mods["reader"](ctx, code, goal).float().view(-1, K)
            s_full = mods["full"](ctx, {"end": fut["end"], "prop": fut["prop"]}, goal).float().view(-1, K)
            s_dir = mods["direct"](ctx, act, goal).float().view(-1, K)
        l_code, l_full, l_dir = rank_loss(s_code, y), rank_loss(s_full, y), rank_loss(s_dir, y)
        loss = l_code + cfg["rec"] * l_rec + cfg["sat"] * saturation_penalty(pre) + l_full + l_dir
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 500 == 0:
            idx = mods["enc"].fsq.codes_to_indices(code.detach()).cpu().numpy()
            log({"stage": 1, "step": step, "rank_code": float(l_code), "rec": float(l_rec), "rank_full": float(l_full),
                 "rank_direct": float(l_dir), "distinct_per_code": code_distinct(idx),
                 "saturated_frac": float((pre.detach().abs() > 1.5).float().mean())})
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps1"]:
            for m in mods.values():
                m.eval()
            sc = score_banks(sel, ("code", "full", "direct"), mods, {}, sel_goals, device, amp, cfg["select_limit"])
            val = {g: pooled_gap(sel, sc, GROUP_TIER[g], cfg["select_limit"])["retained_gap"]["ratio"] for g in GROUPS}
            for g, v in val.items():
                v = v if math.isfinite(v) else -float("inf")
                if v > best[g][0] or best[g][2] is None:
                    best[g] = (v, step + 1, {n: copy.deepcopy({k: t.detach().cpu() for k, t in mods[n].state_dict().items()})
                                              for n in GROUPS[g]})
            log({"stage": 1, "step": step + 1, "selection_gap": val})
    for g, (_, _, states) in best.items():
        for n, st in states.items():
            mods[n].load_state_dict(st)
    return {g: {"gap": best[g][0], "step": best[g][1]} for g in GROUPS}


def stage2(mods, wms, pool, sel, goals, cfg, device, amp, rng, norms, scale, log):
    opts = {n: torch.optim.AdamW(w.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"]) for n, w in wms.items()}
    scheds = {n: schedule(o, cfg["steps2"], cfg["warmup"]) for n, o in opts.items()}
    best = {n: (-float("inf"), None, None) for n in wms}
    rank_cfg = (cfg["rank_scale"], cfg["rank_margin"])
    sel_goals = goals[list(SELECT_GOALS)]
    for step in range(cfg["steps2"]):
        ctx, fut, act, y, goal = train_batch(pool, goals, rng, cfg, device)
        with torch.no_grad(), amp():
            src = mods["enc"](ctx, fut)
            t_code = mods["reader"](ctx, src, goal).float().view(-1, K)
            t_full = mods["full"](ctx, {"end": fut["end"], "prop": fut["prop"]}, goal).float().view(-1, K)
        rec = {"stage": 2, "step": step}
        for n, w in wms.items():
            w.train()
            with amp():
                if n == "cta":
                    expected, logits = w(ctx, act)
                    pred = w.nll(logits, src.float())
                    s = mods["reader"](ctx, expected, goal).float().view(-1, K)
                    teacher = t_code
                else:
                    out = w(ctx, act)
                    pred = (((out["end"] - fut["end"].float()) ** 2).mean() / norms["end"]
                            + ((out["prop"] - fut["prop"].float()) ** 2).mean() / norms["prop"])
                    s = mods["full"](ctx, out, goal).float().view(-1, K)
                    teacher = t_full
                parts = {"pred": pred, "consistency": score_consistency(s, teacher, scale[TEACHER[n]]),
                         "rank": weighted_rank(s, y, *rank_cfg)}
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
            sc = score_banks(sel, tuple(wms), mods, wms, sel_goals, device, amp, cfg["select_limit"])
            val = {n: pooled_gap(sel, sc, n, cfg["select_limit"])["retained_gap"]["ratio"] for n in wms}
            for n, v in val.items():
                v = v if math.isfinite(v) else -float("inf")
                if v > best[n][0] or best[n][2] is None:
                    best[n] = (v, step + 1, copy.deepcopy({k: t.detach().cpu() for k, t in wms[n].state_dict().items()}))
            log({"stage": 2, "step": step + 1, "selection_gap": val})
    for n, w in wms.items():
        w.load_state_dict(best[n][2])
    return {n: {"gap": best[n][0], "step": best[n][1]} for n in wms}


def code_distinct(idx):
    flat = idx.reshape(-1, idx.shape[-1])
    return float(np.mean([len(np.unique(r)) for r in flat]))


@torch.inference_mode()
def code_report(mods, bank, device, amp, limit, batch=8):
    enc = mods["enc"]
    out = []
    for s in range(0, min(limit, bank.n), batch):
        i = np.arange(s, min(s + batch, limit, bank.n))
        ctx, fut, _, _ = bank.batch(i, device)
        with amp():
            out.append(enc.fsq.codes_to_indices(enc(ctx, fut)).view(len(i), K, -1).cpu().numpy())
    idx = np.concatenate(out)
    return {"distinct_values_per_code": code_distinct(idx), "perplexity_per_position": perplexity(idx.reshape(-1, idx.shape[-1]), vocab_size()),
            "indices_used": int(len(np.unique(idx))), "sibling_codes_distinct": bank_distinct(idx)}


def main(a):
    require_compute()
    device = torch.device("cuda")
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    # --only-r4: train on the banks of one collection only (replan-interval study: every bank must have the same
    # executed chunk length, so the 8-step Round-0 / on-policy caches cannot be mixed in).
    sources = {} if a.only_r4 else dict(SOURCES)
    sources["r4std"] = a.r4 / "r4std"
    if a.with_perturbed:
        sources["r4pert"] = a.r4 / "r4pert"
    cfg = {"version": 2, "m": 16, "direct_layers": 8, "lr": 3e-4, "wd": 0.05, "warmup": 500, "dropout": a.dropout,
           "steps1": a.steps1, "steps2": a.steps2, "decisions": a.decisions, "spread_frac": .75, "rec": 1.0, "sat": a.sat,
           "rank_scale": .01, "rank_margin": .001, "eval_every": a.eval_every, "select_limit": a.select_limit,
           "seed": a.seed, "sources": {k: str(v) for k, v in sources.items()},
           "selection": {k: str(v) for k, v in SELECTION.items()}, "job": os.environ.get("SLURM_JOB_ID"),
           "conditional": 1, "path": 1, "target": "negative_mean_vertex_distance_divided_by_512",
           "aux": str(a.aux) if a.aux else None, "success_bonus": a.success_bonus, "hindsight": a.hindsight}
    a.run.mkdir(parents=True, exist_ok=True)
    logf = (a.run / "metrics.jsonl").open("a")
    log = lambda d: (logf.write(json.dumps(d) + "\n"), logf.flush(), print(d, flush=True))
    aux = (lambda n: (a.aux / f"{n}.npz") if a.aux else None)
    banks = [Bank(p, n, aux(n), a.success_bonus) for n, p in sources.items()]
    if a.limit:
        for b in banks:
            b.n = min(b.n, a.limit)
            b.spread = b.spread[b.spread < b.n]
    pool = Pool(banks)
    selection = {"r4dev": a.select_r4 / "r4std"} if a.select_r4 else SELECTION
    sel = [Bank(p, n, aux(n)) for n, p in selection.items()]
    lengths = {b.name: int(b.chunk.shape[2]) for b in banks + sel}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"banks with different chunk lengths: {lengths}")
    cfg["chunk"] = lengths[banks[0].name]
    cfg["selection"] = {k: str(v) for k, v in selection.items()}
    goals = torch.from_numpy(np.load(GOALS)).to(device)
    norms = copy_norms(pool, rng)
    norms["prop"] = float(np.mean([((b.end_prop[: b.n] - proprio(b.ctx_pos[: b.n, 0], b.ctx_pos[: b.n, 1])[:, None]) ** 2).mean()
                                   for b in banks]))
    cfg.update(norms=norms, banks={b.name: {"n": b.n, "informative": int(len(b.spread))} for b in banks},
               selection_banks={b.name: b.n for b in sel})
    (a.run / "config.json").write_text(json.dumps(cfg, indent=2))
    log({"config": cfg})
    mods = build_stage1(cfg, device)
    t1 = time.perf_counter()
    cfg["selected_stage1"] = stage1(mods, pool, sel, goals, cfg, device, amp, rng, norms, log)
    cfg["stage1_hours"] = (time.perf_counter() - t1) / 3600
    for m in mods.values():
        m.eval().requires_grad_(False)
    spread = {"reader": [], "full": []}
    with torch.no_grad():
        for _ in range(8):
            ctx, fut, act, y = pool.sample(rng, cfg["decisions"], cfg["spread_frac"], device)
            goal = goals[torch.as_tensor(rng.integers(0, len(goals), len(y)), device=device)].repeat_interleave(K, 0)
            with amp():
                t_code = mods["reader"](ctx, mods["enc"](ctx, fut), goal)
                t_full = mods["full"](ctx, {"end": fut["end"], "prop": fut["prop"]}, goal)
            for n, t in (("reader", t_code), ("full", t_full)):
                t = t.float().view(-1, K)
                spread[n].append(float((t - t.mean(1, keepdim=True)).square().mean()))
    scale = {n: max(float(np.sqrt(np.mean(v))), 1e-3) for n, v in spread.items()}
    cfg["consistency_scale"] = scale
    wms = build_wms(cfg, device)
    t2 = time.perf_counter()
    cfg["selected_stage2"] = stage2(mods, wms, pool, sel, goals, cfg, device, amp, rng, norms, scale, log)
    cfg["stage2_hours"] = (time.perf_counter() - t2) / 3600
    for w in wms.values():
        w.eval().requires_grad_(False)
    torch.save({"config": cfg, "stage1": {k: m.state_dict() for k, m in mods.items()},
                "wms": {n: w.state_dict() for n, w in wms.items()}}, a.run / "cta_v2.pt")
    tiers = ("full", "code", "direct", "cta", "endpoint")
    sc = score_banks(sel, tiers, mods, wms, goals, device, amp, a.final_limit or None)
    lim = a.final_limit or None
    ladder = {"pooled": {t: pooled_gap(sel, sc, t, lim, ci=True) for t in tiers}}
    for j, b in enumerate(sel):
        ladder[b.name] = {t: pooled_gap([b], {t: [sc[t][j]]}, t, lim, ci=True) for t in tiers}
    np.savez(a.run / "selection_scores.npz", **{f"{t}__{b.name}": sc[t][j] for t in tiers for j, b in enumerate(sel)})
    report = {"ladder": ladder, "codes": code_report(mods, sel[0], device, amp, 1000), "config": cfg,
              "summary_retained_gap": {g: {t: v["retained_gap"]["ratio"] for t, v in d.items()} for g, d in ladder.items()}}
    (a.run / "offline_ladder.json").write_text(json.dumps(report, indent=2))
    (a.run / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps({"summary": report["summary_retained_gap"], "codes": report["codes"]}, indent=1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--r4", type=Path, required=True, help="output of scripts/cta_encode_r4_full.py")
    p.add_argument("--with-perturbed", action="store_true")
    p.add_argument("--aux", type=Path, default=None, help="v2.1: scripts/cta_v2_aux.py output")
    p.add_argument("--success-bonus", type=float, default=0.0, help="v2.1: training label += bonus * success in chunk")
    p.add_argument("--hindsight", type=float, default=0.0, help="v2.1: share of banks read against a hindsight goal")
    p.add_argument("--steps1", type=int, default=16000)
    p.add_argument("--steps2", type=int, default=10000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--decisions", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--sat", type=float, default=1.0, help="FSQ saturation penalty weight (0 in v1 -> code collapse)")
    p.add_argument("--select-limit", type=int, default=1200, help="selection banks per source during training")
    p.add_argument("--final-limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="smoke only: banks per source")
    p.add_argument("--only-r4", action="store_true", help="train only on the --r4 banks (no Round-0 / on-policy caches)")
    p.add_argument("--select-r4", type=Path, default=None,
                   help="select on this cta_encode_r4_full.py output (its r4std split) instead of the 8-step dev caches")
    main(p.parse_args())
