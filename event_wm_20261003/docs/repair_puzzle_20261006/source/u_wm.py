#!/usr/bin/env python3
"""Unified backend, step 2: entity-set event world model, candidate events, learned cost-to-go, search.

State S: K identities x (u, v, r, g, b, covered). Event (e, x): act on identity e so that its state becomes
x. One architecture and rule set for every task family:
  world model  f(S, e, x) -> S': transformer over K entity tokens + one event token; per entity a residual
               for (pos, app) and a covered logit. Errors are measured in threshold units (pos / tol_pos,
               app / thr_app) so attributes of different scales are weighted by what counts as a change.
               Effects on other entities (neighbouring lights, a cube now covered, a lock) are learned.
  candidates   for each actable identity e: its goal state, and the prototypes of e's rest states in the
               play data (k-means, deduplicated): a light gets "the other appearance", a cube its typical
               places (buffers). Data-support rules from the play data: an identity is never acted on
               while covered, and no target position lies inside another identity's place (closer than
               the 1st percentile of that distance over play events).
  cost-to-go   h(S, G) by value iteration in the world model over the same candidates (DeepCubeA-style);
  search       batched weighted A* (f = lam g + h), goal test: every identity within tol_pos / thr_app,
               covered bits equal.
Run as a script: train the world model + cost-to-go on u_events.py output, report offline checks.
--event-pos-only: the agent decides where a moved entity goes; what it then looks like is the world model's to predict.
For an event that moves the acted entity (target position farther than tol_pos), the appearance part of the target
x is replaced by the entity's current appearance before it enters the model. Without this, the hindsight target
(x = the achieved after-state) carries the landing height in the state track, the model learns after = x, and an
unsupported top-of-tower placement looks feasible (57613, cube-triple task 5: 0/20). Off by default; meant for the
state track. On pixel tables place identities never move, and for colour identities it only replaces the read
appearance (nominally constant, reader noise) by the current one.
--h-goals walk: half of each cost-to-go batch pairs a dataset state S with a goal reached from it by k ~ U{0..walk_max}
imagined random events (DeepCubeA-style), so every distance up to the cap is trained. With dataset pairs only, goals
on a 20-button board are ~10 presses away, near-goal states are almost never sampled, and h sits at the cap far from
the goal (57614 puzzle-4x5 state: best h 29.96 after 20k expansions, tasks 2-5 0/6).
--h-absdiff: the cost-to-go also gets |s - g| (which entities differ, whatever the direction; for binary states this is
s XOR g, the input the puzzle-specific planner needed). With walk goals, 60k steps and width 2048 but without it, h was
right to ~5 presses and flat (~5) beyond (57630, local check against GF(2) distances).
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import os
import time
from pathlib import Path

import numpy as np

D = 6                                                         # u, v, r, g, b, covered


def make_wm(K, d=128, layers=3):
    import torch
    import torch.nn as nn

    class EntityWM(nn.Module):
        def __init__(self):
            super().__init__()
            self.K = K
            self.ident = nn.Embedding(K, d)
            self.inp = nn.Sequential(nn.Linear(D + 1, d), nn.GELU(), nn.Linear(d, d))
            self.evt = nn.Sequential(nn.Linear(D, d), nn.GELU(), nn.Linear(d, d))
            self.evt_tag = nn.Parameter(torch.zeros(1, 1, d))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, norm_first=True), layers)
            self.out = nn.Linear(d, D)

        def forward(self, s, e, x, logits=False):
            """s (B, K, D) normalised, e (B,), x (B, D) normalised -> (B, K, D) next state (covered prob or logit)."""
            B = len(s)
            acted = torch.nn.functional.one_hot(e.long(), K).float()[..., None]
            tok = self.inp(torch.cat([s, acted], -1)) + self.ident.weight[None]
            et = self.evt(x)[:, None] + self.ident(e.long())[:, None] + self.evt_tag
            h = self.enc(torch.cat([tok, et], 1))[:, :K]
            o = self.out(h)
            cont = s[..., :5] + o[..., :5]
            if logits:
                return cont, o[..., 5]
            return torch.cat([cont, torch.sigmoid(o[..., 5:])], -1)

    return EntityWM()


def event_input(S, e, x, tol_pos):
    """Target x (n, D) of the acted entity e (n,) in states S (n, K, D) as the model sees it under --event-pos-only:
    appearance replaced by the current one when the event moves the entity."""
    x = np.array(x, np.float32, copy=True)
    cur = np.asarray(S)[np.arange(len(x)), np.asarray(e)]
    moved = np.linalg.norm(x[:, :2] - cur[:, :2], axis=-1) > tol_pos
    x[moved, 2:5] = cur[moved, 2:5]
    return x


def h_input(t, s, g, absdiff):
    """Cost-to-go input from flattened normalised states s, g: [s, g, s - g] (+ |s - g|)."""
    return t.cat([s, g, s - g] + ([(s - g).abs()] if absdiff else []), -1)


def make_h(K, width=1024, absdiff=False):
    import torch.nn as nn

    return nn.Sequential(nn.Linear((4 if absdiff else 3) * K * D, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                         nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1), nn.Softplus())


class Scale:
    """State normalisation: pos -> [-1, 1] by the 64 px image, app -> [-1, 1]; units for errors."""

    def __init__(self, thr_pos, thr_app, tol_pos=None):
        self.thr_pos, self.thr_app = float(thr_pos), float(thr_app)
        self.tol_pos = float(tol_pos) if tol_pos is not None else self.thr_pos      # change / goal tolerance

    def norm(self, s):
        out = s.copy() if isinstance(s, np.ndarray) else s.clone()
        out[..., :2] = s[..., :2] / 32.0 - 1
        out[..., 2:5] = s[..., 2:5] * 2 - 1
        return out

    def denorm(self, z):
        out = z.copy() if isinstance(z, np.ndarray) else z.clone()
        out[..., :2] = (z[..., :2] + 1) * 32.0
        out[..., 2:5] = (z[..., 2:5] + 1) / 2
        return out


class Model:
    """World model + cost-to-go + candidate generator + goal test, on raw (px, rgb) states."""

    def __init__(self, ck, device):
        import torch

        self.K, self.dev = ck["K"], device
        self.sc = Scale(ck["thr_pos"], ck["thr_app"], ck.get("tol_pos"))
        self.wm = make_wm(self.K).to(device).eval(); self.wm.load_state_dict(ck["wm"])
        self.absdiff = bool(ck.get("h_absdiff", False))
        self.h = make_h(self.K, ck.get("h_width", 1024), self.absdiff).to(device).eval(); self.h.load_state_dict(ck["h"])
        self.proto = ck["proto"]                                   # list per identity: (P_k, D) rest-state prototypes
        self.occ_min = ck["occ_min"]; self.cover_rule = ck["cover_rule"]
        self.pos_only = bool(ck.get("event_pos_only", False))
        tol = np.asarray(ck.get("thr_app_id", np.full(self.K, ck["thr_app"])), np.float64)
        self.app_tol = np.where(np.isfinite(tol), tol, 1e9)                          # per identity
        self.torch = torch
        self.rest_support = ck.get("rest_support")

    def canonical(self, S):
        """Snap only train-supported finite-valued attributes; continuous attributes are untouched."""
        if self.rest_support is None:
            return np.asarray(S, np.float32)
        z = np.array(S, np.float32, copy=True)
        for k, fields in enumerate(self.rest_support):
            for c, values in enumerate(fields):
                if values is not None:
                    v = np.asarray(values, np.float32)
                    z[..., k, c] = v[np.abs(z[..., k, c, None] - v).argmin(-1)]
        return z

    def at_goal(self, S, G):
        dp = np.linalg.norm(S[..., :2] - G[..., :2], axis=-1) <= self.sc.tol_pos
        da = np.abs(S[..., 2:5] - G[..., 2:5]).max(-1) <= self.app_tol
        dc = (S[..., 5] > 0.5) == (G[..., 5] > 0.5)
        return (dp & da & dc).all(-1)

    def candidates(self, S, G):
        """-> list of (e, x) for one state S (K, D)."""
        out = []
        for e in range(self.K):
            if self.cover_rule and S[e, 5] > 0.5:
                continue
            xs = [G[e]] + list(self.proto[e])
            for x in xs:
                if self.rest_support is not None:
                    x = np.array(x, np.float32, copy=True)
                    for c, values in enumerate(self.rest_support[e]):
                        if values is not None:
                            v = np.asarray(values, np.float32)
                            x[c] = v[np.abs(x[c] - v).argmin()]
                moved = np.hypot(*(x[:2] - S[e, :2])) > self.sc.tol_pos
                changed = moved or np.abs(x[2:5] - S[e, 2:5]).max() > self.app_tol[e]
                if not changed:
                    continue
                if moved and self.occ_min > 0:
                    d = np.hypot(S[:, 0] - x[0], S[:, 1] - x[1]); d[e] = np.inf
                    if d.min() < self.occ_min:
                        continue
                y = x.copy(); y[5] = 0.0
                if not any(np.abs(y - z).max() < 1e-6 for _, z in out if _ == e):
                    out.append((e, y))
        return out

    def step(self, S, events):
        """S (K, D), events list of (e, x) -> successors (n, K, D)."""
        return self.step_batch(np.repeat(np.asarray(S)[None], len(events), 0), events)

    def step_batch(self, Ss, events):
        """Ss (n, K, D) aligned with events list of (e, x) -> successors (n, K, D); one forward pass."""
        t = self.torch
        with t.no_grad():
            s = t.as_tensor(self.sc.norm(np.asarray(Ss, np.float32)), device=self.dev).float()
            ee = np.array([ev[0] for ev in events])
            xx = np.stack([ev[1] for ev in events])
            if self.pos_only:
                xx = event_input(Ss, ee, xx, self.sc.tol_pos)
            e = t.as_tensor(ee, device=self.dev)
            x = t.as_tensor(self.sc.norm(xx), device=self.dev).float()
            nxt = self.sc.denorm(self.wm(s, e, x).cpu().numpy())
        nxt[..., 5] = nxt[..., 5] > 0.5
        return self.canonical(nxt)

    def heuristic(self, S, G):
        S, G = self.canonical(S), self.canonical(G)
        t = self.torch
        with t.no_grad():
            s = t.as_tensor(self.sc.norm(S), device=self.dev).float().reshape(len(S), -1)
            g = t.as_tensor(self.sc.norm(np.repeat(G[None], len(S), 0)), device=self.dev).float().reshape(len(S), -1)
            return self.h(h_input(t, s, g, self.absdiff)).squeeze(-1).cpu().numpy()

    def key(self, S):
        S = self.canonical(S)
        q = np.concatenate([np.round(S[:, :2] / (self.sc.tol_pos / 2)), np.round(S[:, 2:5] / (np.minimum(self.app_tol, 1.0)[:, None] / 4)), S[:, 5:]], -1)
        return q.astype(np.int64).tobytes()

    def plan(self, S0, G, lam=0.6, batch=64, max_expansions=20000, max_depth=30):
        """Batched weighted A*. Returns (list of (e, x) or None, info)."""
        S0, G = self.canonical(S0), self.canonical(G)
        if self.at_goal(S0, G):
            return [], {"expanded": 0}
        start = np.array(S0, np.float32); start[:, 5] = start[:, 5] > 0.5
        G = np.array(G, np.float32); G[:, 5] = G[:, 5] > 0.5
        h0 = float(self.heuristic(start[None], G)[0])
        openl = [(h0, 0, start.tobytes(), [])]
        seen = {self.key(start): 0}
        expanded, best = 0, (h0, [])
        while openl and expanded < max_expansions:
            nodes = [heapq.heappop(openl) for _ in range(min(batch, len(openl)))]
            src, evs_all, plans = [], [], []
            for f, g_, sb, plan in nodes:
                S = np.frombuffer(sb, np.float32).reshape(self.K, D)
                evs = self.candidates(S, G)
                if not evs or len(plan) >= max_depth:
                    continue
                for ev in evs:
                    src.append(S); evs_all.append(ev); plans.append(plan + [ev])
                expanded += 1
            if not evs_all:
                continue
            succ = self.step_batch(np.stack(src), evs_all).astype(np.float32)
            done = self.at_goal(succ, G[None])
            if done.any():
                j = int(np.nonzero(done)[0][0])
                return plans[j], {"expanded": expanded, "depth": len(plans[j])}
            hs = self.heuristic(succ, G)
            for sn, pl, hv in zip(succ, plans, hs):
                k = self.key(sn)
                if k in seen and seen[k] <= len(pl):
                    continue
                seen[k] = len(pl)
                heapq.heappush(openl, (lam * len(pl) + float(hv), len(pl), sn.tobytes(), pl))
                if hv < best[0]:
                    best = (float(hv), pl)
        return None, {"expanded": expanded, "best_h": best[0], "best_plan": best[1]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True, help="u_events.py output")
    ap.add_argument("--wm-steps", type=int, default=30000)
    ap.add_argument("--h-steps", type=int, default=60000)
    ap.add_argument("--h-width", type=int, default=1024)
    ap.add_argument("--h-goals", choices=("data", "walk"), default="data", help="cost-to-go goals: dataset pairs, or half from imagined walks")
    ap.add_argument("--h-walk-max", type=int, default=30, help="longest imagined walk (the cost cap)")
    ap.add_argument("--h-walk-pool", type=int, default=100000)
    ap.add_argument("--h-absdiff", action="store_true", help="cost-to-go input also gets |s - g|")
    ap.add_argument("--protos", type=int, default=8)
    ap.add_argument("--event-pos-only", action="store_true", help="moved entity: the model gets its target position only (see doc)")
    ap.add_argument("--device", default="cuda", help="cpu for smoke tests")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = a.device
    t0 = time.time()
    tr, va = dict(np.load(a.events / "events_train.npz")), dict(np.load(a.events / "events_val.npz"))
    lab = dict(np.load(a.events / "labels_train.npz"))                           # decompress once
    K = tr["before"].shape[1]
    thr_pos, thr_app = float(tr["thr_pos"]), float(tr["thr_app"])
    tol_pos = float(tr["tol_pos"]) if "tol_pos" in tr else thr_pos
    sc = Scale(thr_pos, thr_app, tol_pos)
    app_unit = thr_app if np.isfinite(thr_app) else 0.1
    # data-support rules
    idx = np.arange(len(tr["e"]))
    cover_rule = bool(tr["before"][idx, tr["e"], 5].mean() < 0.02)
    moves = np.linalg.norm(tr["target"][:, :2] - tr["before"][idx, tr["e"], :2], axis=-1) > tol_pos
    d = np.linalg.norm(tr["before"][..., :2] - tr["target"][:, None, :2], axis=-1); d[idx, tr["e"]] = np.inf
    occ_min = float(np.percentile(d.min(1)[moves], 1)) if moves.any() else 0.0
    # rest-state prototypes per identity (k-means on rest labels, deduplicated by the change thresholds)
    proto = []
    for k in range(K):
        v_ = lab["valid"][:, k]
        X = np.c_[lab["pos"][v_, k], lab["app"][v_, k], np.zeros(v_.sum())]
        if not len(X):                                                           # never observed at rest: no prototypes
            proto.append(np.zeros((0, D), np.float32)); continue
        X = X[rng.permutation(len(X))[:20000]]
        c = X[rng.choice(len(X), min(a.protos, len(X)), replace=False)].copy()
        for _ in range(30):
            lb = np.argmin(np.linalg.norm((X[:, None, :2] - c[None, :, :2]) / thr_pos, axis=-1) ** 2
                           + np.linalg.norm((X[:, None, 2:5] - c[None, :, 2:5]) / app_unit, axis=-1) ** 2, axis=1)
            c = np.stack([X[lb == j].mean(0) if (lb == j).any() else c[j] for j in range(len(c))])
        keep = []
        for x in c:
            if all(np.hypot(*(x[:2] - y[:2])) > thr_pos or np.abs(x[2:5] - y[2:5]).max() > app_unit for y in keep):
                keep.append(x)
        proto.append(np.array(keep, np.float32))
    # ---------------- world model ----------------
    wm = make_wm(K).to(dev)
    if a.event_pos_only:
        tr["target_in"] = event_input(tr["before"], tr["e"], tr["target"], tol_pos)
        va["target_in"] = event_input(va["before"], va["e"], va["target"], tol_pos)
    else:
        tr["target_in"], va["target_in"] = tr["target"], va["target"]
    T = {k: torch.as_tensor(tr[k], device=dev).float() for k in ("before", "after", "target_in")}
    E = torch.as_tensor(tr["e"], device=dev).long()
    unit = torch.tensor([tol_pos / 32.0] * 2 + [2 * app_unit] * 3, device=dev)        # change thresholds in normalised units
    opt = torch.optim.AdamW(wm.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.wm_steps))))
    for step in range(a.wm_steps):
        i = torch.randint(0, len(E), (512,), device=dev)
        s, x, y = sc.norm(T["before"][i]), sc.norm(T["target_in"][i]), sc.norm(T["after"][i])
        cont, logit = wm(s, E[i], x, logits=True)
        loss = (((cont - y[..., :5]) / unit) ** 2).mean() + F.binary_cross_entropy_with_logits(logit, y[..., 5])
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0); opt.step(); sched.step()
        if step % 5000 == 0:
            print({"wm_step": step, "loss": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 1)}, flush=True)
    wm.eval()
    with torch.no_grad():
        V = {k: torch.as_tensor(va[k], device=dev).float() for k in ("before", "after", "target_in")}
        pred = sc.denorm(wm(sc.norm(V["before"]), torch.as_tensor(va["e"], device=dev).long(), sc.norm(V["target_in"])).cpu().numpy())
    after = va["after"]; e_ = va["e"]; iv = np.arange(len(e_))
    err_pos = np.linalg.norm(pred[..., :2] - after[..., :2], axis=-1); err_app = np.abs(pred[..., 2:5] - after[..., 2:5]).max(-1)
    changed = (np.linalg.norm(after[..., :2] - va["before"][..., :2], axis=-1) > tol_pos) | (np.abs(after[..., 2:5] - va["before"][..., 2:5]).max(-1) > app_unit)
    acted = np.zeros_like(changed); acted[iv, e_] = True
    res = {"K": K, "thr_pos": thr_pos, "tol_pos": tol_pos, "thr_app": thr_app, "cover_rule": cover_rule, "occ_min": occ_min,
           "prototypes_per_identity": [len(p) for p in proto], "val_events": int(len(e_)),
           "acted_pos_err_median": float(np.median(err_pos[acted])), "acted_within_tol": float(((err_pos <= tol_pos) & (err_app <= app_unit))[acted].mean()),
           "side_effect_entities": int((changed & ~acted).sum()),
           "side_effect_within_tol": float(((err_pos <= tol_pos) & (err_app <= app_unit))[changed & ~acted].mean()) if (changed & ~acted).any() else None,
           "unchanged_within_tol": float(((err_pos <= tol_pos) & (err_app <= app_unit))[~changed].mean()),
           "covered_acc": float(((pred[..., 5] > 0.5) == (after[..., 5] > 0.5)).mean())}
    print(json.dumps(res), flush=True)
    # ---------------- cost-to-go by value iteration in the world model ----------------
    thr_app_id = np.asarray(tr["thr_app_id"], np.float64) if "thr_app_id" in tr else np.full(K, thr_app)
    model_ck = {"K": K, "thr_pos": thr_pos, "tol_pos": tol_pos, "thr_app": thr_app, "thr_app_id": thr_app_id, "wm": wm.state_dict(), "proto": proto, "occ_min": occ_min,
                "event_pos_only": bool(a.event_pos_only), "h_width": a.h_width, "h_goals": a.h_goals, "h_absdiff": bool(a.h_absdiff),
                "cover_rule": cover_rule}
    h = make_h(K, a.h_width, a.h_absdiff).to(dev); h_tgt = make_h(K, a.h_width, a.h_absdiff).to(dev); h_tgt.load_state_dict(h.state_dict())
    model_ck["h"] = h.state_dict()
    M = Model(model_ck, dev); M.wm = wm; M.h = h_tgt
    states = np.concatenate([tr["before"], tr["after"]]).astype(np.float32)
    ep_states = np.concatenate([tr["episode"], tr["episode"]])
    if a.h_goals == "walk":
        # goals by imagined random walks from dataset states: k ~ U{0..walk_max} prototype events in the world model
        pS = states[rng.integers(0, len(states), a.h_walk_pool)].copy(); pS[:, :, 5] = pS[:, :, 5] > 0.5
        pG = pS.copy(); kk = rng.integers(0, a.h_walk_max + 1, len(pS))
        for w in range(a.h_walk_max):
            rows, evs = [], []
            for b in np.nonzero(kk > w)[0]:
                c = M.candidates(pG[b], pG[b])
                if c:
                    rows.append(b); evs.append(c[rng.integers(len(c))])
            for c0 in range(0, len(rows), 8192):
                pG[rows[c0:c0 + 8192]] = M.step_batch(pG[rows[c0:c0 + 8192]], evs[c0:c0 + 8192])
        print({"h_walk_pool": len(pS), "walk_mean": float(kk.mean()), "min": round((time.time() - t0) / 60, 1)}, flush=True)
    ho = torch.optim.AdamW(h.parameters(), lr=1e-4, weight_decay=1e-5)
    hl = []
    for step in range(a.h_steps):
        i = rng.integers(0, len(states), 128)
        j = rng.integers(0, len(states), 128)
        S, G = states[i], states[j]
        if a.h_goals == "walk":
            w = rng.integers(0, len(pS), 64)
            S = np.concatenate([S[:64], pS[w]]); G = np.concatenate([G[:64], pG[w]])
        evs_all, owner = [], []
        for b in range(len(S)):
            evs = M.candidates(S[b], G[b])
            evs_all += evs; owner += [b] * len(evs)
        y = np.zeros(len(S), np.float32)
        if evs_all:
            owner = np.array(owner)
            succ = M.step_batch(S[owner], evs_all)
            done = M.at_goal(succ, G[owner])
            with torch.no_grad():
                s_n = torch.as_tensor(sc.norm(succ), device=dev).float().reshape(len(succ), -1)
                g_n = torch.as_tensor(sc.norm(G[owner]), device=dev).float().reshape(len(succ), -1)
                hv = h_tgt(h_input(torch, s_n, g_n, a.h_absdiff)).squeeze(-1).cpu().numpy()
            cost = 1 + np.where(done, 0.0, hv)
            y = np.full(len(S), 30.0, np.float32)
            np.minimum.at(y, owner, cost)
        y[M.at_goal(S, G)] = 0.0
        s_n = torch.as_tensor(sc.norm(S), device=dev).float().reshape(len(S), -1)
        g_n = torch.as_tensor(sc.norm(G), device=dev).float().reshape(len(S), -1)
        loss = ((h(h_input(torch, s_n, g_n, a.h_absdiff)).squeeze(-1) - torch.as_tensor(y, device=dev)) ** 2).mean()
        ho.zero_grad(set_to_none=True); loss.backward(); ho.step()
        if step % 1000 == 999:
            h_tgt.load_state_dict(h.state_dict())
        if step % 5000 == 0:
            hl.append({"h_step": step, "loss": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 1)}); print(hl[-1], flush=True)
    model_ck["h"] = h.state_dict()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(model_ck, a.out / "u_model.pt")
    # offline planning check: val (before -> state 2-4 events later in the same episode)
    M = Model(model_ck, dev)
    ok, depth_ratio, tried = 0, [], 0
    for i in range(0, len(va["e"]) - 4, max(1, len(va["e"]) // 60)):
        for gap in (1, 2, 3):
            if i + gap < len(va["e"]) and va["episode"][i + gap - 1] == va["episode"][i]:
                plan, info = M.plan(va["before"][i], va["after"][i + gap - 1], max_expansions=3000)
                tried += 1
                if plan is not None:
                    ok += 1; depth_ratio.append(len(plan) / gap)
    res.update({"offline_plan_found": ok / max(1, tried), "offline_plan_len_over_data_len_median": float(np.median(depth_ratio)) if depth_ratio else None,
                "h_log": hl, "minutes": round((time.time() - t0) / 60, 1)})
    (a.out / "u_model_eval.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "h_log"}, default=float), flush=True)


if __name__ == "__main__":
    main()
