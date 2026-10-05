#!/usr/bin/env python3
"""P1 on OGBench visual cube-triple: does a world model predict block-placement
outcomes in unseen arrangements as well as in seen ones, and does a goal-conditioned
value on the same latents keep judging progress there?

Events: rest periods are runs of >= 5 steps where every block moves < 1 mm per step.
A placement event e is the first step of a rest period in which one block ended
> 3 cm from where it rested in the previous rest period. Context ends at
t0 = e - 10 (block still being carried); the model predicts t0 + 5 and t0 + 10 = e.
Linear (ridge) probes trained on real training latents decode the 9 block
coordinates. Metrics: error of the moved block (effect) and of the other blocks
(persistence), in cm; the same probes on the real latent at e give the ceiling.

Novelty of the post-event arrangement: (a) stack structure (number of blocks resting
on another block: 0, 1, 2 = three-block tower) and (b) nearest-neighbour distance to
training rest arrangements (max over blocks of the xyz distance).

Value: the same GCIVL objective as p1_puzzle.py on frozen block latents; progress
d(s, g) = number of blocks more than 4 cm from their position in g. Metrics:
Spearman(V, -d) on same-episode validation pairs, and the event sign test.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from p1_puzzle import boot, encode, episodes  # noqa: E402
from train_wm import SKIP, build_model  # noqa: E402


def block_xyz(qpos, slices):
    return np.stack([qpos[:, s:s + 3] for s in slices], 1).astype(np.float32)   # (N, k, 3)


def stack_count(p):
    k = p.shape[1]
    on = np.zeros(len(p), np.int64)
    for i in range(k):
        for j in range(k):
            if i != j:
                dxy = np.linalg.norm(p[:, i, :2] - p[:, j, :2], axis=-1)
                on += ((dxy < 0.02) & (np.abs(p[:, i, 2] - p[:, j, 2] - 0.04) < 0.01)).astype(np.int64)
    return on


def placement_events(P, terminals):
    """First step of each rest period where one block moved > 3 cm since the previous rest."""
    speed = np.concatenate([np.linalg.norm(P[1:] - P[:-1], axis=-1).max(1), [1.0]])
    still = speed < 1e-3
    ep, first, last = episodes(terminals)
    events, moved = [], []
    for f, l in zip(first, last):
        prev_rest = None
        t = f
        while t <= l - 5:
            if still[t:t + 5].all():
                if prev_rest is not None:
                    d = np.linalg.norm(P[t] - P[prev_rest], axis=-1)
                    if (d > 0.03).sum() == 1:
                        events.append(t)
                        moved.append(int(d.argmax()))
                prev_rest = t
                while t <= l and still[t]:
                    t += 1
            else:
                t += 1
    return np.array(events), np.array(moved)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--wm", type=Path, required=True)
    ap.add_argument("--slices", type=int, nargs="+", required=True)
    ap.add_argument("--value-steps", type=int, default=100000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    import torch.nn as nn
    from scipy.stats import spearmanr

    dev = "cuda"
    ck = torch.load(a.wm, map_location="cpu", weights_only=False)
    model = build_model(ck["action_dim"]).to(dev).eval()
    model.load_state_dict(ck["state_dict"])
    model.requires_grad_(False)
    mu, sd = ck["action_mean"], ck["action_std"]
    tr, va = np.load(a.data / f"{a.env}.npz"), np.load(a.data / f"{a.env}-val.npz")
    Pt, Pv = block_xyz(tr["qpos"], a.slices), block_xyz(va["qpos"], a.slices)
    tep, tfirst, tlast = episodes(tr["terminals"])
    vterm = va["terminals"]
    vep, vfirst, vlast = episodes(vterm)
    grid = lambda f, l: np.concatenate([np.arange(x, y + 1, SKIP) for x, y in zip(f, l)])
    tgrid, vgrid = grid(tfirst, tlast), grid(vfirst, vlast)
    tobs = tr["observations"]
    zt = encode(model, tobs, tgrid, dev)
    del tobs
    vobs = va["observations"]
    vact = ((va["actions"].astype(np.float32) - mu) / sd).astype(np.float32)

    # Ridge probe latent -> 9 block coordinates.
    mz, sz = zt.mean(0), zt.std(0) + 1e-6
    norm = lambda z: (z - mz) / sz
    X = torch.cat([norm(zt), torch.ones(len(zt), 1, device=dev)], 1).double()
    Y = torch.as_tensor(Pt[tgrid].reshape(len(tgrid), -1), device=dev).double()
    W = torch.linalg.solve(X.T @ X + 1e-2 * torch.eye(X.shape[1], device=dev, dtype=torch.float64), X.T @ Y)
    dec = lambda z: (torch.cat([norm(z), torch.ones(len(z), 1, device=dev)], 1).double() @ W).float().cpu().numpy().reshape(len(z), -1, 3)

    # Training rest arrangements for nearest-neighbour novelty.
    tev, _ = placement_events(Pt, tr["terminals"])
    rest_t = Pt[tev]
    ev, moved = placement_events(Pv, vterm)
    keep = [(e, m) for e, m in zip(ev, moved) if e - 20 >= vfirst[vep[e]] and e + SKIP - 1 <= vlast[vep[e]]]
    ev, moved = np.array([k[0] for k in keep]), np.array([k[1] for k in keep])
    t0 = ev - 10
    frames = np.stack([t0 - 10, t0 - 5, t0, t0 + 5, t0 + 10], 1)
    zf = encode(model, vobs, frames.reshape(-1), dev).view(len(ev), 5, -1)
    blk = np.stack([vact[s:s + SKIP].reshape(-1) for s in (t0[:, None] + np.array([-10, -5, 0, 5])[None]).reshape(-1)])
    blk = torch.as_tensor(blk, device=dev).view(len(ev), 4, -1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        ae = model.action_encoder(blk)
        p1 = model.predict(zf[:, :3], ae[:, :3])[:, -1].float()
        p2 = model.predict(torch.stack([zf[:, 1], zf[:, 2], p1], 1), ae[:, 1:4])[:, -1].float()
    true = Pv[ev]
    stacks = stack_count(true)
    nn_d = np.empty(len(ev))
    for i in range(0, len(ev), 64):
        q = true[i:i + 64, None]
        nn_d[i:i + 64] = np.linalg.norm(q - rest_t[None], axis=-1).max(-1).min(1)
    qs = np.quantile(nn_d, [1 / 3, 2 / 3])
    nn_bin = np.digitize(nn_d, qs)

    def errs(pred):
        e = np.linalg.norm(pred - true, axis=-1) * 100          # cm
        eff = e[np.arange(len(e)), moved]
        mask = np.ones_like(e, bool)
        mask[np.arange(len(e)), moved] = False
        per = e[mask].reshape(len(e), -1).mean(1)
        return eff, per

    rep = {"env": a.env, "events": int(len(ev)), "train_rest_arrangements": int(len(rest_t)),
           "probe_train_err_cm": float(np.linalg.norm(dec(zt[:20000]) - Pt[tgrid[:20000]], axis=-1).mean() * 100),
           "nn_bin_edges_m": [float(x) for x in qs],
           "stack_counts": {str(k): int((stacks == k).sum()) for k in np.unique(stacks)}}
    arms = {"wm_1block": dec(p1), "wm_2block": dec(p2), "real_target": dec(zf[:, 4]), "copy_context": dec(zf[:, 2])}
    for name, pred in arms.items():
        eff, per = errs(pred)
        rep[name] = {"by_stack": {str(k): {"effect_cm": boot(eff[stacks == k]), "persist_cm": boot(per[stacks == k])}
                                  for k in np.unique(stacks)},
                     "by_nn_tercile": {str(k): {"effect_cm": boot(eff[nn_bin == k]), "persist_cm": boot(per[nn_bin == k])}
                                       for k in range(3)}}

    # ---- GCIVL value on frozen block latents (same objective as p1_puzzle.py) ----
    ept = tep[tgrid]
    blk_last = np.searchsorted(ept, ept, side="right") - 1          # episodes are contiguous
    valid = np.nonzero(np.arange(len(ept)) < blk_last)[0]
    gamma = 0.99 ** SKIP

    class V(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.f = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, 512), nn.LayerNorm(512), nn.GELU(),
                                                  nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
                                                  nn.Linear(512, 1)) for _ in range(2)])

        def forward(self, s, g):
            x = torch.cat([s, g], -1)
            return torch.cat([f(x) for f in self.f], -1)

    vnet, vtarg = V(zt.shape[1]).to(dev), V(zt.shape[1]).to(dev)
    vtarg.load_state_dict(vnet.state_dict())
    vopt = torch.optim.Adam(vnet.parameters(), lr=3e-4)
    nzt = norm(zt)
    r = np.random.default_rng(0)
    for _ in range(a.value_steps):
        i = r.choice(valid, 1024)
        u = r.random(1024)
        geo = np.minimum(i + r.geometric(1 - gamma, 1024), blk_last[i])
        gi = np.where(u < 0.2, i, np.where(u < 0.7, geo, r.integers(0, len(zt), 1024)))
        at_goal = torch.as_tensor(gi == i, device=dev).float()
        with torch.no_grad():
            tgt = -(1 - at_goal) + gamma * (1 - at_goal) * vtarg(nzt[i + 1], nzt[gi]).min(-1).values
        diff = tgt[:, None] - vnet(nzt[i], nzt[gi])
        loss = (torch.where(diff > 0, 0.9, 0.1) * diff.pow(2)).mean()
        vopt.zero_grad()
        loss.backward()
        vopt.step()
        with torch.no_grad():
            for p, q in zip(vnet.parameters(), vtarg.parameters()):
                q.mul_(0.995).add_(0.005 * p)
    Vf = lambda s, g: vnet(norm(s), norm(g)).mean(-1)
    prog = lambda x, g: int((np.linalg.norm(x - g, axis=-1) > 0.04).sum())

    zv = encode(model, vobs, vgrid, dev)
    vep_g = vep[vgrid]
    r2 = np.random.default_rng(1)
    si = r2.integers(0, len(vgrid), 6000)
    gj = np.array([r2.choice(np.nonzero(vep_g == vep_g[k])[0]) for k in si])
    d = np.array([prog(Pv[vgrid[x]], Pv[vgrid[y]]) for x, y in zip(si, gj)])
    with torch.no_grad():
        vs = Vf(zv[si], zv[gj]).cpu().numpy()
    st_s = stack_count(Pv[vgrid[si]])
    rep["value_spearman_by_stack"] = {str(k): [round(float(spearmanr(vs[st_s == k], -d[st_s == k]).correlation), 4), int((st_s == k).sum())]
                                      for k in np.unique(st_s) if (st_s == k).sum() >= 50}
    sign_ok, sign_st, sign_nn = [], [], []
    for k, e in enumerate(ev):
        cand = vgrid[(vep_g == vep[e]) & (vgrid >= e + 15)]
        if not len(cand):
            continue
        gt = cand[r2.integers(0, len(cand))]
        d0, d1 = prog(Pv[t0[k]], Pv[gt]), prog(Pv[e], Pv[gt])
        if d0 == d1:
            continue
        gz = encode(model, vobs, np.array([gt]), dev)
        with torch.no_grad():
            dv = (Vf(zf[k:k + 1, 4], gz) - Vf(zf[k:k + 1, 2], gz)).item()
        sign_ok.append(float(np.sign(dv) == np.sign(d0 - d1)))
        sign_st.append(stacks[k])
        sign_nn.append(nn_bin[k])
    sign_ok, sign_st, sign_nn = map(np.array, (sign_ok, sign_st, sign_nn))
    rep["value_event_sign_by_stack"] = {str(k): boot(sign_ok[sign_st == k]) for k in np.unique(sign_st)}
    rep["value_event_sign_by_nn_tercile"] = {str(k): boot(sign_ok[sign_nn == k]) for k in range(3)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
