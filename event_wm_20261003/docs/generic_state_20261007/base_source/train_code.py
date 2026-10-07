#!/usr/bin/env python3
"""Slow binary event code on frozen patch tokens (no state labels).

A head maps the frozen encoder's patch tokens to K Bernoulli bits. Training uses frame
pairs (t, t + gap) from one episode and four terms:
  slow  mean |p_t - p_{t+gap}|          object state changes only at events; the arm moves every step
  var   hinge on each bit's batch std    no constant bits (DWMR / VICReg-style)
  cor   mean |off-diagonal correlation|  bits carry different information
  bin   mean p (1 - p)                   bits are near 0/1, so rounding is lossless

Evaluation on validation frames uses the privileged button states only to score the code:
bit-light alignment, code purity (does the code determine the configuration), event detection
(code changes vs true toggles) and the event vocabulary (XOR patterns at detected events).
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

from common import Split, detect_events, load_encoder, make_code_head, pack, save_json, size_of, tokens
from lightsout import solver


def encode_probs(enc, head, tmu, tsd, obs, dev, bs=2048):
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(obs), bs):
            tok = (tokens(enc, obs[i:i + bs], dev) - tmu) / tsd
            out.append(torch.sigmoid(head(tok)).float().cpu())
    return torch.cat(out).numpy()


def stable_changes(b, ep, m):
    """t with code change between t and t+1, stable for m frames before and after (same episode)."""
    n = len(b)
    diff = (b[1:] != b[:-1]).any(1)
    cand = np.nonzero(diff & (ep[1:] == ep[:-1]))[0]
    keep = []
    for t in cand:
        lo, hi = t - m + 1, t + m + 1
        if lo < 0 or hi >= n or ep[lo] != ep[hi]:
            continue
        if (b[lo:t + 1] == b[t]).all() and (b[t + 1:hi + 1] == b[t + 1]).all():
            keep.append(t)
    return np.array(keep, np.int64)


def evaluate(p, val: Split, rows, cols, m=3, tol=5):
    b = (p > 0.5).astype(np.uint8)
    s = val.button_states
    ep = val.ep
    same = ep[1:] == ep[:-1]
    K, L = b.shape[1], s.shape[1]
    res = {"bits": K, "lights": L}
    flip = (b[1:] != b[:-1])[same].mean(0)
    res["bit_flip_rate"] = np.round(np.sort(flip), 5).tolist()
    res["light_flip_rate_mean"] = float((s[1:] != s[:-1])[same].mean())
    # Slow/fast split: 2-means on log flip rate (no hand-set threshold).
    lf = np.log(flip + 1e-6)
    c = np.array([lf.min(), lf.max()])
    for _ in range(50):
        lab = np.abs(lf[:, None] - c[None]).argmin(1)
        c = np.array([lf[lab == j].mean() if (lab == j).any() else c[j] for j in range(2)])
    slow = lab == 0
    res["slow_bits"] = int(slow.sum())
    res["slow_fast_mean_rate"] = [float(np.exp(c[0])), float(np.exp(c[1]))]
    # Alignment: best |agreement - 0.5| + 0.5 per light and per bit.
    agree = (b[:, :, None] == s[:, None, :]).mean(0)              # (K, L)
    sc = np.maximum(agree, 1 - agree)
    res["per_light_best_bit_acc"] = np.round(sc.max(0), 4).tolist()
    res["lights_with_bit_acc_ge_0.99"] = int((sc.max(0) >= 0.99).sum())
    for name, sel in (("all", np.ones(K, bool)), ("slow", slow)):
        if sel.sum() == 0 or sel.sum() > 63:
            continue
        key, cfg = pack(b[:, sel]), pack(s)
        # purity: does the code value determine the configuration?
        order = np.lexsort((cfg, key))
        k2, c2 = key[order], cfg[order]
        grp = np.r_[0, np.nonzero(k2[1:] != k2[:-1])[0] + 1, len(k2)]
        hit = 0
        for g0, g1 in zip(grp[:-1], grp[1:]):
            _, cnt = np.unique(c2[g0:g1], return_counts=True)
            hit += cnt.max()
        n_codes = len(np.unique(key))
        n_cfg = len(np.unique(cfg))
        r = {"purity": hit / len(key), "distinct_codes": int(n_codes), "distinct_configs": int(n_cfg)}
        # event detection vs true toggles (per-bit debounce + merge changes within 10 frames)
        ts, te, cb = detect_events(b[:, sel], ep, m)
        true = val.true_events()
        if len(ts) and len(true):
            j = np.searchsorted(true, ts - tol)                     # first true event >= start - tol
            jj = np.clip(j, 0, len(true) - 1)
            det_hit = (j < len(true)) & (true[jj] <= te + tol)
            k = np.searchsorted(ts, true + tol, side="right") - 1   # last detection starting <= true + tol
            kk = np.clip(k, 0, len(ts) - 1)
            true_hit = (k >= 0) & (te[kk] >= true - tol)
            r["detected_events"] = int(len(ts))
            r["true_events"] = int(len(true))
            r["precision"] = float(det_hit.mean())
            r["recall"] = float(true_hit.mean())
            r["span_median"] = float(np.median(te - ts))
            A = solver(rows, cols)[0]
            col = {pack(A[:, i][None])[0]: i for i in range(A.shape[1])}
            pat = pack(cb[te + 1] ^ cb[ts])
            tog = pack(s[true[jj] + 1] ^ s[true[jj]])
            btn = np.array([col.get(x, -1) for x in tog])
            ok = det_hit & (btn >= 0)
            u, cnt = np.unique(pat[ok], return_counts=True)
            cov = np.cumsum(np.sort(cnt)[::-1]) / max(cnt.sum(), 1)
            r["patterns"] = int(len(u))
            r["patterns_for_95pct"] = int(np.searchsorted(cov, 0.95) + 1) if len(cov) else 0
            hit = 0
            for x in u:
                hit += np.bincount(btn[ok][pat[ok] == x]).max()
            r["pattern_to_button_purity"] = hit / max(ok.sum(), 1)
        res[name] = r
    return res, slow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True, help="cache/<env> from cache_data.py")
    ap.add_argument("--env", required=True)
    ap.add_argument("--base", type=Path, required=True, help="train_wm.py checkpoint (frozen encoder)")
    ap.add_argument("--bits", type=int, default=32)
    ap.add_argument("--train-frames", type=int, default=600_000)
    ap.add_argument("--val-frames", type=int, default=100_000)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--w-slow", type=float, default=1.0)
    ap.add_argument("--w-var", type=float, default=1.0)
    ap.add_argument("--w-cor", type=float, default=1.0)
    ap.add_argument("--w-bin", type=float, default=0.1)
    ap.add_argument("--gamma", type=float, default=0.45)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    rows, cols = size_of(a.env)
    t0 = time.time()
    train = Split(a.cache, "train", a.train_frames)
    val = Split(a.cache, "val", a.val_frames)
    print({"train_frames": train.n, "val_frames": val.n, "load_min": round((time.time() - t0) / 60, 1)}, flush=True)
    enc, _ = load_encoder(a.base, dev)
    sample = tokens(enc, train.obs[rng.integers(0, train.n, 4096)], dev)
    tmu, tsd = sample.mean((0, 1)), sample.std((0, 1)) + 1e-6
    head = make_code_head(a.bits).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = min(500, a.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    a.out.mkdir(parents=True, exist_ok=True)
    log = []
    K = a.bits
    eye = torch.eye(K, device=dev, dtype=torch.bool)
    for step in range(a.steps):
        t1, t2 = train.pairs(rng, a.batch, a.gap)
        tok = (tokens(enc, np.concatenate([train.obs[t1], train.obs[t2]]), dev) - tmu) / tsd
        p = torch.sigmoid(head(tok).float())
        p1, p2 = p[:a.batch], p[a.batch:]
        slow = (p1 - p2).abs().mean()
        std = p.std(0)
        var = torch.relu(a.gamma - std).mean()
        pc = (p - p.mean(0)) / (std + 1e-6)
        corr = pc.T @ pc / (len(p) - 1)
        cor = corr.masked_fill(eye, 0).abs().sum() / (K * (K - 1))
        binr = (p * (1 - p)).mean()
        ramp = min(1.0, step / max(1, a.steps // 3))
        loss = a.w_slow * slow + a.w_var * var + a.w_cor * cor + a.w_bin * ramp * binr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == a.steps - 1:
            rec = {"step": step, "loss": loss.item(), "slow": slow.item(), "var": var.item(), "cor": cor.item(),
                   "bin": binr.item(), "min": round((time.time() - t0) / 60, 1)}
            log.append(rec)
            print(rec, flush=True)
    head.eval()
    pv = encode_probs(enc, head, tmu, tsd, val.obs, dev)
    res, slow_mask = evaluate(pv, val, rows, cols)
    res["train_log"] = log
    res["args"] = {k: str(v) for k, v in vars(a).items()}
    torch.save({"head": head.state_dict(), "bits": K, "token_mean": tmu.cpu(), "token_std": tsd.cpu(),
                "slow_mask": torch.as_tensor(slow_mask), "base": str(a.base), "env": a.env}, a.out / "code.pt")
    save_json(a.out / "code_eval.json", res)
    print({k: v for k, v in res.items() if k not in ("train_log", "bit_flip_rate", "per_light_best_bit_acc")}, flush=True)


if __name__ == "__main__":
    main()
