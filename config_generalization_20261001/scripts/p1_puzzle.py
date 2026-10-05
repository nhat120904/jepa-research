#!/usr/bin/env python3
"""P1 on OGBench visual puzzle: does a world model predict button-press outcomes in
unseen configurations as well as in seen ones, and does a goal-conditioned value
on the same latents keep judging progress there?

Novelty of a configuration = Hamming distance to the nearest configuration in the
training split (0 = seen; capped at 5).

World model (frozen, scripts/train_wm.py): for every validation press event e
(button states change between e and e+1, exactly one event in the window), the
context is the encoded frames at t0-10, t0-5, t0 (t0 = e-2) with the executed
action blocks; the model predicts t0+5 and t0+10. Linear probes (trained on real
training latents) decode button states. Metrics: effect accuracy (toggled
buttons), persistence accuracy (untouched buttons), exact configuration match;
the same probes on the real target latent give the representation ceiling.

Value (GCIVL objective: expectile 0.9 regression of r + gamma V'(s', g), reward -1
until the goal, goals from the current/future/random block) on the frozen block
latents of the training split. Metrics on validation: Spearman of V(s, g) with the
true minimal number of presses d*(s, g) (GF(2), scripts/lightsout.py), and for each
press event the sign agreement between V(post, g) - V(pre, g) and the true change
in d*.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lightsout import min_presses  # noqa: E402
from train_wm import SKIP, build_model, to_pixels  # noqa: E402

CAP = 4


def pack(bits):
    return (bits.astype(np.int64) << np.arange(bits.shape[1], dtype=np.int64)).sum(1)


def novelty(codes, train_set, n_bits):
    flips = {d: [sum(1 << i for i in c) for c in combinations(range(n_bits), d)] for d in range(1, CAP + 1)}
    out = np.full(len(codes), CAP + 1, np.int64)
    cache = {}
    for qi, q in enumerate(codes.tolist()):
        if q in cache:
            out[qi] = cache[q]
            continue
        d = 0 if q in train_set else next((k for k in range(1, CAP + 1)
                                           if any((q ^ f) in train_set for f in flips[k])), CAP + 1)
        cache[q] = out[qi] = d
    return out


def episodes(terminals):
    ep = np.concatenate([[0], np.cumsum(terminals[:-1])]).astype(np.int64)
    first = np.concatenate([[0], np.nonzero(terminals[:-1])[0] + 1])
    last = np.concatenate([np.nonzero(terminals)[0], [len(terminals) - 1]])[: len(first)]
    return ep, first, last


def encode(model, obs, idx, dev, bs=2048):
    import torch

    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(idx), bs):
            px = to_pixels(obs[idx[i:i + bs]][:, None], dev)
            out.append(model.encode({"pixels": px})["emb"][:, 0].float())
    return torch.cat(out)


def boot(x, seed=0, draws=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    r = np.random.default_rng(seed)
    b = x[r.integers(0, len(x), (draws, len(x)))].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(b, 2.5)), 4),
            round(float(np.percentile(b, 97.5)), 4), int(len(x))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--value-steps", type=int, default=100000)
    ap.add_argument("--patch-wm", type=Path, default=None,
                    help="patch-token predictor (train_patch_wm.py); latents become the encoder's patch tokens")
    ap.add_argument("--train-episodes", type=int, default=0, help="use only the first N training episodes (0 = all)")
    ap.add_argument("--exclude", type=int, nargs="*", default=[],
                    help="held-out region: configurations with all these buttons ON were removed from WM training; "
                         "probes and value are trained without them and results are split by membership")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy.stats import spearmanr

    dev = "cuda"
    rows, cols = map(int, a.env.split("-")[2].split("x"))
    ck = torch.load(a.wm, map_location="cpu", weights_only=False)
    model = build_model(ck["action_dim"]).to(dev).eval()
    model.load_state_dict(ck["state_dict"])
    model.requires_grad_(False)
    mu, sd = ck["action_mean"], ck["action_std"]
    patch = None
    if a.patch_wm is not None:
        from patch_wm import PatchPredictor, patch_tokens

        pk = torch.load(a.patch_wm, map_location="cpu", weights_only=False)
        tmu, tsd = pk["token_mean"].to(dev), pk["token_std"].to(dev)
        patch = PatchPredictor(dim=tmu.numel(), action_dim=SKIP * ck["action_dim"]).to(dev).eval()
        patch.load_state_dict(pk["predictor"])
        patch.requires_grad_(False)

        def encode(model, obs, idx, dev, bs=1024):  # noqa: F811  (patch tokens, flattened, fp16)
            out = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, len(idx), bs):
                    tk = patch_tokens(model.encoder, to_pixels(obs[idx[i:i + bs]], dev)).float()
                    out.append(((tk - tmu) / tsd).flatten(1).half())
            return torch.cat(out)

    tr, va = np.load(a.data / f"{a.env}.npz"), np.load(a.data / f"{a.env}-val.npz")
    tb, vb = tr["button_states"].astype(np.int64), va["button_states"].astype(np.int64)
    nb = tb.shape[1]
    inR = lambda bits: (bits[:, a.exclude] == 1).all(1) if a.exclude else np.zeros(len(bits), bool)
    train_set = set(np.unique(pack(tb[~inR(tb)])).tolist())
    vobs = va["observations"]
    vact = ((va["actions"].astype(np.float32) - mu) / sd).astype(np.float32)
    vterm = va["terminals"]
    vep, vfirst, vlast = episodes(vterm)
    tep, tfirst, tlast = episodes(tr["terminals"])

    # Block-grid frames (every 5th step from each episode start).
    def grid(first, last):
        return np.concatenate([np.arange(f, l + 1, SKIP) for f, l in zip(first, last)])

    if a.train_episodes:
        tfirst, tlast = tfirst[:a.train_episodes], tlast[:a.train_episodes]
    tgrid, vgrid = grid(tfirst, tlast), grid(vfirst, vlast)
    tobs = tr["observations"]
    zt = encode(model, tobs, tgrid, dev)
    del tobs
    zv = encode(model, vobs, vgrid, dev)
    yt = torch.as_tensor(tb[tgrid], device=dev).float()
    yv = vb[vgrid]

    # Linear probes, one logit per button.
    probe = nn.Linear(zt.shape[1], nb).to(dev)
    popt = torch.optim.Adam(probe.parameters(), lr=1e-3)
    if patch is None:
        mz, sz = zt.mean(0), zt.std(0) + 1e-6
        norm = lambda z: (z - mz) / sz
    else:
        norm = lambda z: z.float()          # tokens are already standardised per channel
    okp = torch.as_tensor(np.nonzero(~inR(tb[tgrid]))[0], device=dev)
    for _ in range(3000):
        i = okp[torch.randint(0, len(okp), (4096,), device=dev)]
        loss = F.binary_cross_entropy_with_logits(probe(norm(zt[i])), yt[i])
        popt.zero_grad()
        loss.backward()
        popt.step()
    def dec(z):
        with torch.no_grad():
            return np.concatenate([(probe(norm(z[i:i + 4096])) > 0).long().cpu().numpy() for i in range(0, len(z), 4096)])
    rep = {"env": a.env, "buttons": nb, "train_unique_configs": len(train_set),
           "probe_train_bit_acc": float((dec(zt[:20000]) == tb[tgrid[:20000]]).mean())}
    vnov = novelty(pack(yv), train_set, nb)
    pv = dec(zv)
    rep["probe_val_by_novelty"] = {str(k): {"bit_acc": float((pv[vnov == k] == yv[vnov == k]).mean()),
                                            "exact": float((pv[vnov == k] == yv[vnov == k]).all(1).mean()),
                                            "n": int((vnov == k).sum())}
                                   for k in range(CAP + 2) if (vnov == k).any()}

    # Press events with a clean window.
    ch = (vb[1:] != vb[:-1]).any(1) & (vterm[:-1] == 0)
    ev = np.nonzero(ch)[0]
    keep = []
    for e in ev:
        t0 = e - 2
        lo, hi = t0 - 10, t0 + 10 + SKIP - 1
        if lo < vfirst[vep[e]] or hi > vlast[vep[e]]:
            continue
        if ch[lo:t0 + 10].sum() != 1:
            continue
        keep.append(e)
    ev = np.array(keep)
    t0 = ev - 2
    frames = np.stack([t0 - 10, t0 - 5, t0, t0 + 5, t0 + 10], 1)
    zf = encode(model, vobs, frames.reshape(-1), dev).view(len(ev), 5, -1)
    blk = np.stack([vact[s:s + SKIP].reshape(-1) for s in (t0[:, None] + np.array([-10, -5, 0, 5])[None]).reshape(-1)])
    blk = torch.as_tensor(blk, device=dev).view(len(ev), 4, -1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        if patch is None:
            ae = model.action_encoder(blk)
            p1 = model.predict(zf[:, :3], ae[:, :3])[:, -1].float()
            ctx = torch.stack([zf[:, 1], zf[:, 2], p1], 1)
            p2 = model.predict(ctx, ae[:, 1:4])[:, -1].float()
        else:
            tk = zf.float().view(len(ev), 5, 64, -1)
            p1, p2 = [], []
            for i in range(0, len(ev), 256):
                q1 = patch(tk[i:i + 256, :3], blk[i:i + 256, :3])[:, -1].float()
                q2 = patch(torch.stack([tk[i:i + 256, 1], tk[i:i + 256, 2], q1], 1), blk[i:i + 256, 1:4])[:, -1].float()
                p1.append(q1.flatten(1))
                p2.append(q2.flatten(1))
            p1, p2 = torch.cat(p1), torch.cat(p2)
    pre, post = vb[ev], vb[ev + 1]
    tog = pre != post
    nov_pre = novelty(pack(pre), train_set, nb)
    nov_post = novelty(pack(post), train_set, nb)

    def scores(pred_bits):
        eff = np.array([(p[t] == q[t]).mean() for p, q, t in zip(pred_bits, post, tog)])
        per = np.array([(p[~t] == q[~t]).mean() if (~t).any() else 1.0 for p, q, t in zip(pred_bits, post, tog)])
        ex = (pred_bits == post).all(1).astype(float)
        return eff, per, ex

    arms = {"wm_1block": dec(p1), "wm_2block": dec(p2), "real_target": dec(zf[:, 4]), "copy_context": dec(zf[:, 2])}
    rep["events"] = int(len(ev))
    rep["event_by_post_novelty"] = {}
    for name, bits in arms.items():
        eff, per, ex = scores(bits)
        rep["event_by_post_novelty"][name] = {
            str(k): {"effect": boot(eff[nov_post == k]), "persist": boot(per[nov_post == k]), "exact": boot(ex[nov_post == k])}
            for k in range(CAP + 2) if (nov_post == k).any()}
    if a.exclude:
        heldp = inR(post)
        rep["event_by_heldout_post"] = {}
        for name, bits in arms.items():
            eff, per, ex = scores(bits)
            rep["event_by_heldout_post"][name] = {
                lab: {"effect": boot(eff[m]), "persist": boot(per[m]), "exact": boot(ex[m])}
                for lab, m in (("in_region", heldp), ("outside", ~heldp)) if m.any()}
    rep["event_pre_vs_post_novelty_counts"] = {f"{i},{j}": int(((nov_pre == i) & (nov_post == j)).sum())
                                               for i in range(CAP + 2) for j in range(CAP + 2)
                                               if ((nov_pre == i) & (nov_post == j)).any()}

    # ---- GCIVL value on frozen block latents ----
    ept = tep[tgrid]
    start_of = {e: i for i, e in reversed(list(enumerate(ept)))}
    end_of = {e: i for i, e in enumerate(ept)}
    blk_first = np.array([start_of[e] for e in ept])
    blk_last = np.array([end_of[e] for e in ept])
    inR_g = inR(tb[tgrid])
    valid = np.nonzero((np.arange(len(ept)) < blk_last) & ~inR_g & ~np.roll(inR_g, -1))[0]
    gamma = 0.99 ** SKIP

    class V(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.tok = None
            if patch is not None:                     # per-token projection, then flatten (64 x 16)
                self.tok = nn.Linear(d // 64, 16)
                d = 64 * 16
            self.f = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, 512), nn.LayerNorm(512), nn.GELU(),
                                                  nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
                                                  nn.Linear(512, 1)) for _ in range(2)])

        def forward(self, s, g):
            if self.tok is not None:
                s = self.tok(s.view(len(s), 64, -1)).flatten(1)
                g = self.tok(g.view(len(g), 64, -1)).flatten(1)
            x = torch.cat([s, g], -1)
            return torch.cat([f(x) for f in self.f], -1)        # (B, 2)

    vnet, vtarg = V(zt.shape[1]).to(dev), V(zt.shape[1]).to(dev)
    vtarg.load_state_dict(vnet.state_dict())
    vopt = torch.optim.Adam(vnet.parameters(), lr=3e-4)
    nzt = zt if patch is not None else norm(zt)
    r = np.random.default_rng(0)
    for step in range(a.value_steps):
        i = r.choice(valid, 1024)
        u = r.random(1024)
        geo = np.minimum(i + r.geometric(1 - gamma, 1024), blk_last[i])
        gi = np.where(u < 0.2, i, np.where(u < 0.7, geo, r.integers(0, len(zt), 1024)))
        gi = np.where(inR_g[gi], i, gi)                      # never use held-out goals
        s, s2, g = nzt[i].float(), nzt[i + 1].float(), nzt[gi].float()
        at_goal = torch.as_tensor(gi == i, device=dev).float()
        with torch.no_grad():
            tgt = -1 * (1 - at_goal) + gamma * (1 - at_goal) * vtarg(s2, g).min(-1).values
        v = vnet(s, g)
        diff = tgt[:, None] - v
        w = torch.where(diff > 0, 0.9, 0.1)
        loss = (w * diff.pow(2)).mean()
        vopt.zero_grad()
        loss.backward()
        vopt.step()
        with torch.no_grad():
            for p, q in zip(vnet.parameters(), vtarg.parameters()):
                q.mul_(0.995).add_(0.005 * p)
    Vf = lambda s, g: vnet(norm(s), norm(g)).mean(-1)

    # Spearman of V(s, g) with -d* on validation same-episode pairs, by novelty of s.
    vep_g = vep[vgrid]
    r2 = np.random.default_rng(1)
    si = r2.integers(0, len(vgrid), 6000)
    gj = np.array([r2.choice(np.nonzero(vep_g == vep_g[k])[0]) for k in si])
    dstar = np.array([min_presses(yv[x], yv[y], rows, cols) for x, y in zip(si, gj)])
    with torch.no_grad():
        vs = Vf(zv[si], zv[gj]).cpu().numpy()
    ns = vnov[si]
    rep["value_spearman_by_novelty"] = {
        str(k): [round(float(spearmanr(vs[ns == k], -dstar[ns == k]).correlation), 4), int((ns == k).sum())]
        for k in range(CAP + 2) if (ns == k).sum() >= 50}
    rep["value_spearman_all"] = round(float(spearmanr(vs, -dstar).correlation), 4)
    rep["value_mean_by_dstar"] = {str(k): [round(float(vs[dstar == k].mean()), 3), int((dstar == k).sum())]
                                  for k in np.unique(dstar) if (dstar == k).sum() >= 20}
    if a.exclude:
        gR = inR(yv[gj]) | inR(yv[si])
        rep["value_spearman_by_heldout"] = {lab: [round(float(spearmanr(vs[m], -dstar[m]).correlation), 4), int(m.sum())]
                                            for lab, m in (("state_or_goal_in_region", gR), ("outside", ~gR)) if m.sum() >= 50}

    # Event sign test: goal = a block in the same episode at least 3 blocks after the event.
    sign_ok, sign_nov, sign_d0, sign_R = [], [], [], []
    for k, e in enumerate(ev):
        cand = vgrid[(vep_g == vep[e]) & (vgrid >= e + 15)]
        if not len(cand):
            continue
        gt = cand[r2.integers(0, len(cand))]
        d0 = min_presses(vb[e], vb[gt], rows, cols)
        d1 = min_presses(vb[e + 1], vb[gt], rows, cols)
        if d0 == d1 or min(d0, d1) < 0:
            continue
        gz = encode(model, vobs, np.array([gt]), dev)
        with torch.no_grad():
            dv = (Vf(zf[k:k + 1, 4], gz) - Vf(zf[k:k + 1, 2], gz)).item()
        sign_ok.append(float(np.sign(dv) == np.sign(d0 - d1)))
        sign_nov.append(nov_post[k])
        sign_d0.append(d0)
        sign_R.append(bool(a.exclude) and bool(inR(np.stack([vb[e], vb[e + 1], vb[gt]])).any()))
    sign_ok, sign_nov, sign_d0, sign_R = map(np.array, (sign_ok, sign_nov, sign_d0, sign_R))
    bins = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 99)]
    rep["value_event_sign_by_goal_distance"] = {f"{lo}-{hi}": boot(sign_ok[(sign_d0 >= lo) & (sign_d0 <= hi)])
                                                for lo, hi in bins if ((sign_d0 >= lo) & (sign_d0 <= hi)).any()}
    if a.exclude:
        rep["value_event_sign_by_heldout"] = {"touches_region": boot(sign_ok[sign_R]), "outside": boot(sign_ok[~sign_R])}
    rep["value_event_sign_by_post_novelty"] = {str(k): boot(sign_ok[sign_nov == k])
                                               for k in range(CAP + 2) if (sign_nov == k).any()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
