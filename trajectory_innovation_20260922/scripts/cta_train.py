"""CTA end-to-end training and the offline attribution ladder (docs/CTA_E2E_PROTOCOL.md).

Stage 1, codec: source encoder + reader + distortion decoder on actual futures (pairwise ranking loss on cov8,
RESEARCH_DESIGN §5a, plus conditional reconstruction of the future tokens). On the SAME batches, with the same loss
and steps: the tier-1 reader on the uncompressed actual future and the direct scorer D_direct(C, A, q).
Stage 2, world model: p(S | C, A) and the context-only prior r(S | C) on stop-gradient source codes. No labels.
Stage 3 (optional, --steps3 > 0): source blocks with + lam * predictability under a frozen WM copy, alternating
with WM blocks on EMA-encoder codes.
Optional reader adaptation (--adapt-steps > 0): the reader is fine-tuned on a mix of source and predicted codes.

Ladder on dev roots: full (tier 1, actual future), code (tier 2, source code of the actual future),
pred (tier 3, WM greedy code), pred_soft (WM expected code), direct (D_direct).
"""

import argparse
import contextlib
import copy
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ti_wm.codec import pool_grid
from ti_wm.contract import require_compute
from ti_wm.cta import (
    CodeWM, FutureDecoder, Scorer, SourceEncoder, action_features, goal_scores, proprio, saturation_penalty,
    sibling_contrast, vocab_size,
)
from ti_wm.cta_eval import bank_distinct, choice_agreement, perplexity, ranking_metrics
from ti_wm.sibling import RANK_MARGIN, rank_loss
from ti_wm.wb import Logger

K = 8
TIERS = ("full", "code", "pred", "pred_soft", "direct")
GROUPS = {"codec": ("enc", "reader", "dec"), "full": ("full",), "direct": ("direct",), "wm": ("wm", "prior")}


class Split:
    """Cached features of one split (cta_encode.py). Tokens stay on the CPU and move per batch."""

    def __init__(self, path, n_max=None):
        meta = np.load(path / "meta.npz")
        n = len(meta["root"]) if n_max is None else min(n_max, len(meta["root"]))
        self.n, self.root, self.cov8 = n, meta["root"][:n], meta["cov8"][:n].astype(np.float32)
        self.tok = {k: torch.from_numpy(np.ascontiguousarray(np.load(path / f"{k}.npy", mmap_mode="r")[:n]))
                    for k in ("cur", "prev", "end", "seg")}
        self.ctx_pos = torch.from_numpy(meta["ctx_pos"][:n]).float()
        self.chunk = torch.from_numpy(meta["chunk"][:n]).float()
        self.end_prop = proprio(torch.from_numpy(meta["end_pos"][:n]), torch.from_numpy(meta["end_prev_pos"][:n]))
        self.cov = torch.from_numpy(self.cov8)
        self.spread = np.flatnonzero(np.ptp(self.cov8, axis=1) > RANK_MARGIN)
        # ranking batches come from decisions with a label spread (all decisions only if none has one: smoke data)
        self.rank_pool = self.spread if len(self.spread) else np.arange(n)

    def context(self, i, device):
        ctx = {"cur": self.tok["cur"][i], "prev": self.tok["prev"][i],
               "prop": proprio(self.ctx_pos[i, 0], self.ctx_pos[i, 1])}
        return {k: v.to(device, non_blocking=True).repeat_interleave(K, 0) for k, v in ctx.items()}

    def future(self, i, device):
        return {"end": self.tok["end"][i].flatten(0, 1).to(device), "seg": self.tok["seg"][i].flatten(0, 1).to(device),
                "prop": self.end_prop[i].flatten(0, 1).to(device)}

    def actions(self, i, device):
        return action_features(self.chunk[i], self.ctx_pos[i, 0][:, None]).flatten(0, 1).to(device)


def build(cfg, device):
    m = cfg["m"]
    models = {"enc": SourceEncoder(m, conditional=cfg["conditional"], path=cfg["path"]),
              "reader": Scorer("code", m=m), "dec": FutureDecoder(m),
              "full": Scorer("future"), "direct": Scorer("action", layers=cfg["direct_layers"]),
              "wm": CodeWM(m), "prior": CodeWM(m, use_actions=False)}
    return {k: v.to(device) for k, v in models.items()}


def params(models, names):
    return [p for n in names for p in models[n].parameters()]


def optimizer(models, groups, lr, wd, steps, warmup):
    opt = torch.optim.AdamW([{"params": params(models, GROUPS[g])} for g in groups], lr=lr, weight_decay=wd)
    ramp = lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, ramp)


def clip(models, groups):
    return {g: float(torch.nn.utils.clip_grad_norm_(params(models, GROUPS[g]), 1.0)) for g in groups}


def pair_acc(s, cov):
    better = (cov[:, :, None] - cov[:, None, :]) > RANK_MARGIN
    return float(((s[:, :, None] > s[:, None, :]) & better).sum() / better.sum().clamp(min=1))


def copy_baseline(split, rng, n=256):
    """Squared error of predicting the future tokens by the decision frame: the distortion normalizers."""
    i = torch.as_tensor(np.sort(rng.choice(split.n, min(n, split.n), replace=False)))
    cur = split.tok["cur"][i].float()
    end = split.tok["end"][i].float()
    seg = split.tok["seg"][i].float()
    small = pool_grid(cur, 8)
    return {"end": float(((end - cur[:, None]) ** 2).mean()), "seg": float(((seg - small[:, None, None]) ** 2).mean())}


def codec_losses(models, ctx, fut, goal, cov, norms):
    code, pre = models["enc"](ctx, fut, return_pre=True)
    s = models["reader"](ctx, code, goal).float().view(-1, K)
    end_hat, seg_hat = models["dec"](ctx, code)
    l_end = ((end_hat - fut["end"].float()) ** 2).mean() / norms["end"]
    l_seg = ((seg_hat - fut["seg"].float()) ** 2).mean() / norms["seg"]
    return code, s, rank_loss(s, cov), 0.5 * (l_end + l_seg), saturation_penalty(pre)


def sample_goals(goals, rng, b, device):
    return goals[torch.as_tensor(rng.integers(0, len(goals), b), device=device)].repeat_interleave(K, 0)


def stage1(models, train, dev, goals, cfg, device, log, rng, norms, amp):
    groups = ("codec", "full", "direct")
    opt, sched = optimizer(models, groups, cfg["lr"], cfg["wd"], cfg["steps1"], cfg["warmup"])
    for m in models.values():
        m.train()
    t0 = time.perf_counter()
    for step in range(cfg["steps1"]):
        i = torch.as_tensor(rng.choice(train.rank_pool, cfg["decisions"]))
        ctx, fut, act = train.context(i, device), train.future(i, device), train.actions(i, device)
        cov, goal = train.cov[i].to(device), sample_goals(goals, rng, len(i), device)
        with amp():
            code, s_code, l_code, l_rec, l_sat = codec_losses(models, ctx, fut, goal, cov, norms)
            s_full = models["full"](ctx, fut, goal).float().view(-1, K)
            s_dir = models["direct"](ctx, act, goal).float().view(-1, K)
        l_full, l_dir = rank_loss(s_full, cov), rank_loss(s_dir, cov)
        loss = l_code + cfg["rec"] * l_rec + cfg["sat"] * l_sat + l_full + l_dir
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norms_g = clip(models, groups)
        opt.step()
        sched.step()
        if step % cfg["log_every"] == 0:
            idx = models["enc"].fsq.codes_to_indices(code.detach()).cpu().numpy()
            log.log({"stage": 1, "s1/rank_code": float(l_code), "s1/rec_delta": float(l_rec), "s1/saturation": float(l_sat),
                     "s1/rank_full": float(l_full), "s1/rank_direct": float(l_dir),
                     "s1/pair_acc_code": pair_acc(s_code, cov), "s1/pair_acc_full": pair_acc(s_full, cov),
                     "s1/pair_acc_direct": pair_acc(s_dir, cov), "s1/code_perplexity": perplexity(idx, vocab_size()),
                     "s1/lr": sched.get_last_lr()[0], "s1/grad_norm": norms_g,
                     "s1/sec_per_step": (time.perf_counter() - t0) / (step + 1)}, step=step)
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps1"]:
            rep, _ = ladder(models, dev, goals, device, amp, ("full", "code", "direct"), n=cfg["eval_n"], ci=False)
            log.log({"stage": 1, **{f"dev/{k}": v for k, v in brief(rep).items()}}, step=step)
            for m in models.values():
                m.train()
    return cfg["steps1"]


@torch.inference_mode()
def source_indices(models, split, device, amp, batch=64):
    enc = models["enc"].eval()
    out = []
    for s in range(0, split.n, batch):
        i = torch.arange(s, min(s + batch, split.n))
        with amp():
            code = enc(split.context(i, device), split.future(i, device))
        out.append(enc.fsq.codes_to_indices(code).view(len(i), K, -1).cpu())
    return torch.cat(out)


def wm_losses(models, ctx, act, tgt, contrast_banks=0):
    """Teacher-forced CE of the WM and the prior; with contrast_banks > 0 also the within-bank action InfoNCE
    (ti_wm.cta.sibling_contrast) on the first contrast_banks banks of the batch."""
    mem = models["wm"].encode(ctx, act)
    lw = models["wm"].logits(mem, tgt).float()
    lp = models["prior"].logits(models["prior"].encode(ctx), tgt).float()
    ce_w = F.cross_entropy(lw.flatten(0, 1), tgt.flatten())
    ce_p = F.cross_entropy(lp.flatten(0, 1), tgt.flatten())
    if contrast_banks:
        n = contrast_banks * K
        contrast = sibling_contrast(models["wm"], mem[:n], tgt[:n], K)
    else:
        contrast = (ce_w * 0, ce_w.new_tensor(float("nan")))
    return lw, ce_w, ce_p, contrast


def wm_log(prefix, lw, ce_w, ce_p, tgt, m):
    return {f"{prefix}/ce_wm": float(ce_w), f"{prefix}/ce_prior": float(ce_p),
            f"{prefix}/action_info_bits": float(ce_p - ce_w) * m / math.log(2),
            f"{prefix}/tf_token_acc": float((lw.argmax(-1) == tgt).float().mean())}


def stage2(models, train, dev, goals, codes, cfg, device, log, rng, amp, step0):
    opt, sched = optimizer(models, ("wm",), cfg["lr"], cfg["wd"], cfg["steps2"], cfg["warmup"])
    models["wm"].train(), models["prior"].train()
    t0 = time.perf_counter()
    for step in range(cfg["steps2"]):
        i = torch.as_tensor(rng.integers(0, train.n, cfg["wm_decisions"]))
        ctx, act = train.context(i, device), train.actions(i, device)
        tgt = codes[i].flatten(0, 1).to(device)
        with amp():
            lw, ce_w, ce_p, (l_con, con_acc) = wm_losses(models, ctx, act, tgt,
                                                          cfg["contrast_banks"] if cfg["wm_contrast"] else 0)
        opt.zero_grad(set_to_none=True)
        (ce_w + ce_p + cfg["wm_contrast"] * l_con).backward()
        norms_g = clip(models, ("wm",))
        opt.step()
        sched.step()
        if step % cfg["log_every"] == 0:
            log.log({"stage": 2, **wm_log("s2", lw, ce_w, ce_p, tgt, cfg["m"]), "s2/grad_norm": norms_g,
                     "s2/sibling_contrast": float(l_con), "s2/sibling_contrast_acc": float(con_acc),
                     "s2/lr": sched.get_last_lr()[0], "s2/sec_per_step": (time.perf_counter() - t0) / (step + 1)},
                    step=step0 + step)
        if (step + 1) % cfg["eval_every"] == 0 or step + 1 == cfg["steps2"]:
            rep, _ = ladder(models, dev, goals, device, amp, ("code", "pred", "pred_soft"), n=cfg["eval_n"], ci=False)
            log.log({"stage": 2, **{f"dev/{k}": v for k, v in brief(rep).items()}}, step=step0 + step)
            models["wm"].train(), models["prior"].train()
    return step0 + cfg["steps2"]


def stage3(models, train, dev, goals, cfg, device, log, rng, norms, amp, step0):
    """Alternating co-design (paper Eq. source): predictability enters as the squared distance of the
    straight-through code to the frozen WM's expected code, a differentiable surrogate of -log p(S | C, A)."""
    ema = copy.deepcopy(models["enc"]).eval().requires_grad_(False)
    opt_c, sch_c = optimizer(models, ("codec",), cfg["lr3"], cfg["wd"], cfg["steps3"], cfg["warmup"])
    opt_w, sch_w = optimizer(models, ("wm",), cfg["lr3"], cfg["wd"], cfg["steps3"], cfg["warmup"])
    codebook = models["enc"].fsq.codebook
    step = 0
    while step < cfg["steps3"]:
        frozen = copy.deepcopy(models["wm"]).eval().requires_grad_(False)
        for m in ("enc", "reader", "dec"):
            models[m].train()
        for _ in range(cfg["src_block"]):
            i = torch.as_tensor(rng.choice(train.rank_pool, cfg["decisions"]))
            ctx, fut, act = train.context(i, device), train.future(i, device), train.actions(i, device)
            cov, goal = train.cov[i].to(device), sample_goals(goals, rng, len(i), device)
            with amp():
                code, s_code, l_code, l_rec, l_sat = codec_losses(models, ctx, fut, goal, cov, norms)
                with torch.no_grad():
                    idx = models["enc"].fsq.codes_to_indices(code.detach())
                    probs = frozen.logits(frozen.encode(ctx, act), idx).float().softmax(-1)
            l_pred = ((code.float() - probs @ codebook) ** 2).sum(-1).mean()
            loss = l_code + cfg["rec"] * l_rec + cfg["sat"] * l_sat + cfg["lam"] * l_pred
            opt_c.zero_grad(set_to_none=True)
            loss.backward()
            clip(models, ("codec",))
            opt_c.step()
            sch_c.step()
            with torch.no_grad():
                for pe, p in zip(ema.parameters(), models["enc"].parameters()):
                    pe.lerp_(p, 1 - cfg["ema"])
        models["wm"].train(), models["prior"].train()
        for _ in range(cfg["wm_block"]):
            i = torch.as_tensor(rng.integers(0, train.n, cfg["wm_decisions"]))
            ctx, act = train.context(i, device), train.actions(i, device)
            with torch.no_grad(), amp():
                tgt = ema.fsq.codes_to_indices(ema(ctx, train.future(i, device)))
            with amp():
                lw, ce_w, ce_p, (l_con, _) = wm_losses(models, ctx, act, tgt,
                                                       cfg["contrast_banks"] if cfg["wm_contrast"] else 0)
            opt_w.zero_grad(set_to_none=True)
            (ce_w + ce_p + cfg["wm_contrast"] * l_con).backward()
            clip(models, ("wm",))
            opt_w.step()
            sch_w.step()
        step += cfg["src_block"]
        log.log({"stage": 3, "s3/rank_code": float(l_code), "s3/rec_delta": float(l_rec), "s3/pred": float(l_pred),
                 **wm_log("s3", lw, ce_w, ce_p, tgt, cfg["m"])}, step=step0 + step)
        if step % cfg["eval_every"] < cfg["src_block"]:
            rep, _ = ladder(models, dev, goals, device, amp, ("code", "pred"), n=cfg["eval_n"], ci=False)
            log.log({"stage": 3, **{f"dev/{k}": v for k, v in brief(rep).items()}}, step=step0 + step)
    models["enc"].load_state_dict(ema.state_dict())
    return step0 + step


def adapt(models, train, dev, goals, cfg, device, log, rng, amp, step0):
    """Reader adaptation: fine-tune the reader on source codes mixed with the WM's greedy codes (encoder, WM frozen)."""
    opt = torch.optim.AdamW(models["reader"].parameters(), lr=cfg["lr_adapt"], weight_decay=cfg["wd"])
    codebook = models["enc"].fsq.codebook
    for m in ("enc", "wm"):
        models[m].eval()
    models["reader"].train()
    for step in range(cfg["adapt_steps"]):
        i = torch.as_tensor(rng.choice(train.rank_pool, cfg["decisions"]))
        ctx, fut, act = train.context(i, device), train.future(i, device), train.actions(i, device)
        cov, goal = train.cov[i].to(device), sample_goals(goals, rng, len(i), device)
        with torch.no_grad(), amp():
            src = models["enc"](ctx, fut)
            pred = codebook[models["wm"].decode(models["wm"].encode(ctx, act))]
            use_pred = (torch.rand(len(i), device=device) < cfg["adapt_mix"]).repeat_interleave(K)[:, None, None]
            code = torch.where(use_pred, pred, src.float())
        with amp():
            s = models["reader"](ctx, code, goal).float().view(-1, K)
        loss = rank_loss(s, cov)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models["reader"].parameters(), 1.0)
        opt.step()
        if step % cfg["log_every"] == 0:
            log.log({"stage": 4, "adapt/rank": float(loss), "adapt/pair_acc": pair_acc(s, cov)}, step=step0 + step)
    return step0 + cfg["adapt_steps"]


@torch.inference_mode()
def ladder(models, split, goals, device, amp, tiers=TIERS, n=None, ci=True, chunk=4, samples=0, norms=None):
    """Scores every dev decision with each tier; returns (report, scores)."""
    for m in models.values():
        m.eval()
    n = split.n if n is None else min(n, split.n)
    enc, reader, wm, prior = models["enc"], models["reader"], models["wm"], models["prior"]
    codebook = enc.fsq.codebook
    tiers = tuple(tiers) + ((f"pred_s{samples}",) if samples else ())
    scores = {t: np.zeros((n, K), np.float32) for t in tiers}
    need_src = any(t != "full" and t != "direct" for t in tiers)
    need_wm = any(t.startswith("pred") for t in tiers)
    src_all, pred_all = [], []
    acc = {"ce_w": 0.0, "ce_p": 0.0, "tf": 0.0, "exact": 0.0, "tok": 0, "rows": 0, "sse_end": 0.0, "sse_seg": 0.0}
    for s in range(0, n, chunk):
        i = torch.arange(s, min(s + chunk, n))
        b, sl = len(i), slice(s, s + len(i))
        ctx = split.context(i, device)
        put = lambda t, x: scores[t].__setitem__(sl, x.view(b, K).cpu().numpy())
        with amp():
            if "full" in tiers or need_src:
                fut = split.future(i, device)
            if "full" in tiers:
                put("full", goal_scores(models["full"], ctx, {"end": fut["end"], "prop": fut["prop"]}, goals))
            if "direct" in tiers or need_wm:
                act = split.actions(i, device)
            if "direct" in tiers:
                put("direct", goal_scores(models["direct"], ctx, act, goals))
            if need_src:
                code = enc(ctx, fut)
                src = enc.fsq.codes_to_indices(code)
                src_all.append(src.view(b, K, -1).cpu())
                if "code" in tiers:
                    put("code", goal_scores(reader, ctx, code, goals))
                if norms is not None:
                    end_hat, seg_hat = models["dec"](ctx, code)
                    acc["sse_end"] += float(((end_hat - fut["end"].float()) ** 2).mean()) * len(src)
                    acc["sse_seg"] += float(((seg_hat - fut["seg"].float()) ** 2).mean()) * len(src)
                    acc["rows"] += len(src)
            if need_wm:
                mem = wm.encode(ctx, act)
                pidx = wm.decode(mem)
                pred_all.append(pidx.view(b, K, -1).cpu())
                if "pred" in tiers:
                    put("pred", goal_scores(reader, ctx, codebook[pidx], goals))
                if "pred_soft" in tiers:
                    put("pred_soft", goal_scores(reader, ctx, wm.logits(mem, pidx).float().softmax(-1) @ codebook, goals))
                if samples:
                    put(f"pred_s{samples}", sum(goal_scores(reader, ctx, codebook[wm.decode(mem, sample=True)], goals)
                                                for _ in range(samples)) / samples)
                lw = wm.logits(mem, src).float()
                lp = prior.logits(prior.encode(ctx), src).float()
                acc["ce_w"] += float(F.cross_entropy(lw.flatten(0, 1), src.flatten(), reduction="sum"))
                acc["ce_p"] += float(F.cross_entropy(lp.flatten(0, 1), src.flatten(), reduction="sum"))
                acc["tf"] += float((lw.argmax(-1) == src).sum())
                acc["exact"] += float((pidx == src).all(-1).sum())
                acc["tok"] += src.numel()
    labels, roots = split.cov8[:n], split.root[:n]
    report = {}
    for t in tiers:
        r = ranking_metrics(scores[t], labels, roots, ci=ci)
        r.pop("chosen")
        report[t] = r
    pairs = [("code", "full"), ("pred", "code"), ("pred", "full"), ("direct", "full"), ("pred", "direct")]
    report["agree"] = {f"{a}_vs_{b}": choice_agreement(scores[a], scores[b], labels)
                       for a, b in pairs if a in scores and b in scores}
    codes = {}
    if src_all:
        src = torch.cat(src_all).numpy()
        codes.update({"src_perplexity": perplexity(src.reshape(-1, src.shape[-1]), vocab_size()),
                      "src_bank_distinct": bank_distinct(src)})
    if pred_all:
        pred = torch.cat(pred_all).numpy()
        m = pred.shape[-1]
        codes.update({"pred_perplexity": perplexity(pred.reshape(-1, m), vocab_size()),
                      "pred_bank_distinct": bank_distinct(pred),
                      "wm_ce_nats": acc["ce_w"] / acc["tok"], "prior_ce_nats": acc["ce_p"] / acc["tok"],
                      "action_info_bits": (acc["ce_p"] - acc["ce_w"]) / acc["tok"] * m / math.log(2),
                      "wm_tf_token_acc": acc["tf"] / acc["tok"], "wm_greedy_exact_code": acc["exact"] / (acc["tok"] / m)})
    if acc["rows"]:
        codes.update({"r2_delta_end": 1 - acc["sse_end"] / acc["rows"] / norms["end"],
                      "r2_delta_seg": 1 - acc["sse_seg"] / acc["rows"] / norms["seg"]})
    report["codes"] = codes
    return report, scores


def brief(report):
    """Point estimates only, for step-wise logging."""
    out = {}
    for t, r in report.items():
        if t in ("agree", "codes"):
            out.update({f"{t}/{k}": v for k, v in r.items()})
        else:
            out.update({f"{t}/spearman": r["within_bank_spearman"]["ratio"], f"{t}/retained_gap": r["retained_gap"]["ratio"],
                        f"{t}/chose_default": r["chose_default"]})
    return out


def main(a):
    require_compute()
    run, features = a.run, a.features
    cfg = {k: v for k, v in vars(a).items() if k not in ("run", "features")}
    cfg["features"], cfg["slurm_job"] = str(features), os.environ.get("SLURM_JOB_ID")
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device(a.device)
    amp = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device.type == "cuda"
           else contextlib.nullcontext)
    report = {"status": "RUNNING", "config": cfg}
    log = Logger(run, "train", cfg, name=f"train_{cfg['slurm_job']}_s{a.seed}")
    try:
        t0 = time.perf_counter()
        train, dev = Split(features / "train", a.n_train), Split(features / "dev")
        goals = torch.from_numpy(np.load(features / "goals.npy")).to(device)
        report["data"] = {"train_decisions": train.n, "train_spread": int(len(train.spread)), "dev_decisions": dev.n,
                          "dev_roots": int(len(np.unique(dev.root))), "load_seconds": time.perf_counter() - t0}
        print(json.dumps(report["data"]), flush=True)
        models = build(cfg, device)
        report["parameters"] = {k: sum(p.numel() for p in m.parameters()) for k, m in models.items()}
        norms = copy_baseline(train, rng)
        report["copy_baseline"] = norms
        log.summary({"parameters": report["parameters"], "data": report["data"]})
        step = stage1(models, train, dev, goals, cfg, device, log, rng, norms, amp)
        codes = source_indices(models, train, device, amp)
        step = stage2(models, train, dev, goals, codes, cfg, device, log, rng, amp, step)
        if a.steps3:
            step = stage3(models, train, dev, goals, cfg, device, log, rng, norms, amp, step)
        if a.adapt_steps:
            step = adapt(models, train, dev, goals, cfg, device, log, rng, amp, step)
        ladder_report, scores = ladder(models, dev, goals, device, amp, TIERS, samples=a.samples, norms=norms)
        report["ladder"] = ladder_report
        np.savez(run / "dev_scores.npz", root=dev.root, decision=np.arange(dev.n), cov8=dev.cov8, **scores)
        torch.save({"config": cfg, "state": {k: m.state_dict() for k, m in models.items()}, "norms": norms,
                    **torch.load(features / "pca.pt")}, run / "cta.pt")
        log.log({"stage": 9, **{f"final/{k}": v for k, v in brief(ladder_report).items()}}, step=step)
        log.summary({"final": {t: {k: v for k, v in r.items() if k != "chose_default"} for t, r in ladder_report.items()}})
        log.table("final/ladder", ("tier", "spearman", "spearman_lo", "spearman_hi", "gap", "gap_lo", "gap_hi", "chose_default"),
                  [(t, r["within_bank_spearman"]["ratio"], r["within_bank_spearman"]["lo"], r["within_bank_spearman"]["hi"],
                    r["retained_gap"]["ratio"], r["retained_gap"]["lo"], r["retained_gap"]["hi"], r["chose_default"])
                   for t, r in ladder_report.items() if t not in ("agree", "codes")])
        report["seconds"] = time.perf_counter() - t0
        report["status"] = "DONE"
        print(json.dumps({k: v for k, v in ladder_report.items()}, indent=1, default=float), flush=True)
    except Exception:
        report["status"] = "FAILED"
        report["error"] = traceback.format_exc()
        raise
    finally:
        (run / "train_report.json").write_text(json.dumps(report, indent=2, default=float))
        log.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-train", type=int, default=None, help="cap on training decisions (smoke only)")
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--conditional", type=int, default=1)
    p.add_argument("--path", type=int, default=1)
    p.add_argument("--direct-layers", type=int, default=8)
    p.add_argument("--rec", type=float, default=1.0)
    p.add_argument("--sat", type=float, default=0.0, help="weight of the FSQ saturation penalty (round 1+)")
    p.add_argument("--wm-contrast", type=float, default=0.0, help="weight of the within-bank action InfoNCE (round 1+)")
    p.add_argument("--contrast-banks", type=int, default=16, help="banks per WM batch that enter the contrast")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--decisions", type=int, default=16)
    p.add_argument("--wm-decisions", type=int, default=32)
    p.add_argument("--steps1", type=int, default=10_000)
    p.add_argument("--steps2", type=int, default=20_000)
    p.add_argument("--steps3", type=int, default=0)
    p.add_argument("--lam", type=float, default=0.0)
    p.add_argument("--lr3", type=float, default=1e-4)
    p.add_argument("--src-block", type=int, default=200)
    p.add_argument("--wm-block", type=int, default=400)
    p.add_argument("--ema", type=float, default=0.99)
    p.add_argument("--adapt-steps", type=int, default=0)
    p.add_argument("--adapt-mix", type=float, default=0.5)
    p.add_argument("--lr-adapt", type=float, default=1e-4)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-n", type=int, default=600)
    p.add_argument("--log-every", type=int, default=50)
    main(p.parse_args())
