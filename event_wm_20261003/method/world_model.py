#!/usr/bin/env python3
"""Components 9-15 (method/README.md): entity-set event world model, candidate events, finite-value canonicalization,
learned cost-to-go, search. Port of docs/generic_state_20261007/base_source/u_wm.py (see SOURCES in method/README.md).

State S: K entities x D = (u, v, app (A values in [0, 1]), covered). Event (e, x): act on entity e so that its state
becomes x. Same rules as u_wm:
  world model  f(S, e, x) -> S': transformer over K entity tokens + one event token; per entity a residual for (pos, app)
               and a covered logit; errors in threshold units (pos / tol_pos, app / thr_app). Effects on other entities
               (neighbouring lights, a lock, a cube that appears where it is placed) are learned.
  candidates   for each entity e not covered: its goal state and its typical rest states in the play data (prototypes).
  canonical    attributes with at most 8 distinct values in TRAIN states are snapped to them (states, targets, search keys).
  cost-to-go   h(S, G) by value iteration in the world model over the candidates (DeepCubeA style); half of the goals come
               from imagined random walks of up to --h-walk-max events; input [s, g, s - g, |s - g|].
  search       batched weighted A* (f = lam g + h); goal test: every entity whose goal state is KNOWN within tol_pos /
               thr_app and covered equal.
Differences from u_wm (method/README.md lists them):
  - D = A + 3 from the data (u_wm: D = 6, RGB appearance);
  - goal entities can be unknown (not readable or touched by the agent in the goal image, method/README.md: "goal test
    only on goal tokens judged observed"): they are ignored by the goal test, get no goal candidate, and enter the
    cost-to-go with the current value (s - g = 0);
  - prototypes of quantized appearance (events.py rule "quantized") are the most frequent distinct rest values of the
    entity (weighted by rest duration), not k-means means, which would average distinct codes into codes that never occur;
  - prototypes come from the rest runs of events.py (rests_train.npz), not per-frame rest labels;
  - --max-candidates N: when an event-support model is attached (train_support.py), only the N candidates with the
    highest support score are expanded per state (K is ~50-150 token entities, not 3-20 objects). Stages: --stage wm
    (world model) -> train_support.py -> --stage h (cost-to-go with that support) -> offline planning check.
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

from event_support import apply_event_support, features as support_features, make_support


def make_wm(K, D, d=128, layers=3):
    import torch
    import torch.nn as nn

    class EntityWM(nn.Module):
        def __init__(self):
            super().__init__()
            self.K, self.D = K, D
            self.ident = nn.Embedding(K, d)
            self.inp = nn.Sequential(nn.Linear(D + 1, d), nn.GELU(), nn.Linear(d, d))
            self.evt = nn.Sequential(nn.Linear(D, d), nn.GELU(), nn.Linear(d, d))
            self.evt_tag = nn.Parameter(torch.zeros(1, 1, d))
            self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, batch_first=True, norm_first=True), layers)
            self.out = nn.Linear(d, D)

        def forward(self, s, e, x, logits=False):
            """s (B, K, D) normalised, e (B,), x (B, D) normalised -> (B, K, D) next state (covered prob or logit)."""
            acted = torch.nn.functional.one_hot(e.long(), K).float()[..., None]
            tok = self.inp(torch.cat([s, acted], -1)) + self.ident.weight[None]
            et = self.evt(x)[:, None] + self.ident(e.long())[:, None] + self.evt_tag
            h = self.enc(torch.cat([tok, et], 1))[:, :K]
            o = self.out(h)
            cont = s[..., :D - 1] + o[..., :D - 1]
            if logits:
                return cont, o[..., D - 1]
            return torch.cat([cont, torch.sigmoid(o[..., D - 1:])], -1)

    return EntityWM()


def finite_rest_support(states, max_values=8, precision=5):
    """Near-exact finite attributes from TRAIN states (<= max_values distinct values); continuous fields stay None."""
    out = []
    for k in range(states.shape[1]):
        fields = []
        for c in range(states.shape[2]):
            values = np.unique(np.round(states[:, k, c], precision))
            fields.append(values.tolist() if len(values) <= max_values else None)
        out.append(fields)
    return out


def event_input(S, e, x, tol_pos):
    """--event-pos-only: when the event moves the acted entity, its appearance in x is replaced by the current one."""
    x = np.array(x, np.float32, copy=True)
    cur = np.asarray(S)[np.arange(len(x)), np.asarray(e)]
    moved = np.linalg.norm(x[:, :2] - cur[:, :2], axis=-1) > tol_pos
    x[moved, 2:-1] = cur[moved, 2:-1]
    return x


def h_input(t, s, g, absdiff):
    return t.cat([s, g, s - g] + ([(s - g).abs()] if absdiff else []), -1)


def make_h(K, D, width=1024, absdiff=False):
    import torch.nn as nn

    return nn.Sequential(nn.Linear((4 if absdiff else 3) * K * D, width), nn.GELU(), nn.Linear(width, width), nn.GELU(),
                         nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1), nn.Softplus())


class Scale:
    """pos -> [-1, 1] by the 64 px image, app [0, 1] -> [-1, 1]; covered unchanged."""

    def __init__(self, thr_pos, thr_app, tol_pos=None):
        self.thr_pos, self.thr_app = float(thr_pos), float(thr_app)
        self.tol_pos = float(tol_pos) if tol_pos is not None else self.thr_pos

    def norm(self, s):
        out = s.copy() if isinstance(s, np.ndarray) else s.clone()
        out[..., :2] = s[..., :2] / 32.0 - 1
        out[..., 2:-1] = s[..., 2:-1] * 2 - 1
        return out

    def denorm(self, z):
        out = z.copy() if isinstance(z, np.ndarray) else z.clone()
        out[..., :2] = (z[..., :2] + 1) * 32.0
        out[..., 2:-1] = (z[..., 2:-1] + 1) / 2
        return out


def prototypes(rests, K, A, quantized, thr_pos, app_unit, P, rng):
    """per entity (P_k, D) typical rest states. quantized: the P most frequent distinct rest values weighted by duration;
    else k-means on rest values (u_wm), deduplicated by the change thresholds."""
    out = []
    ent, dur, app = rests["entity"], rests["end"] - rests["start"] + 1, rests["app"]
    pos = rests["pos"]
    for k in range(K):
        m = ent == k
        if not m.any():
            out.append(np.zeros((0, A + 3), np.float32)); continue
        X, w = app[m], dur[m].astype(np.float64)
        if quantized:
            keys, inv = np.unique(np.round(X / (app_unit / 2)).astype(np.int64), axis=0, return_inverse=True)
            tot = np.bincount(inv.ravel(), weights=w)
            top = np.argsort(-tot)[:P]
            vals = np.stack([X[inv.ravel() == j][0] for j in top])
        else:
            Xs = X[rng.permutation(len(X))[:20000]]
            c = Xs[rng.choice(len(Xs), min(P, len(Xs)), replace=False)].copy()
            for _ in range(30):
                lb = np.argmin(np.linalg.norm((Xs[:, None] - c[None]) / app_unit, axis=-1), axis=1)
                c = np.stack([Xs[lb == j].mean(0) if (lb == j).any() else c[j] for j in range(len(c))])
            vals = []
            for x in c:
                if all(np.abs(x - y).max() > app_unit for y in vals):
                    vals.append(x)
            vals = np.array(vals)
        out.append(np.c_[np.repeat(pos[k][None], len(vals), 0), vals, np.zeros(len(vals))].astype(np.float32))
    return out


def prototypes_labels(lab, K, thr_pos, app_unit, P, rng):
    """app_unit: scalar or per entity (K,)."""
    app_unit = np.broadcast_to(np.asarray(app_unit, np.float64), (K,))
    return [_protos_entity(lab, k, thr_pos, float(app_unit[k]), P, rng) for k in range(K)]


def _protos_entity(lab, k, thr_pos, app_unit, P, rng):
    """object entities (events_objects.py): u_wm rule -- per entity k-means on its per-frame rest labels (pos, app),
    distances in change-threshold units, deduplicated by the change thresholds (a light gets its other appearance, a cube
    its typical places)."""
    v_ = lab["valid"][:, k]
    X = np.c_[lab["pos"][v_, k], lab["app"][v_, k], np.zeros(v_.sum())]
    if not len(X):
        return np.zeros((0, 6), np.float32)
    X = X[rng.permutation(len(X))[:20000]]
    c = X[rng.choice(len(X), min(P, len(X)), replace=False)].copy()
    for _ in range(30):
        lb = np.argmin(np.linalg.norm((X[:, None, :2] - c[None, :, :2]) / thr_pos, axis=-1) ** 2
                       + np.linalg.norm((X[:, None, 2:5] - c[None, :, 2:5]) / app_unit, axis=-1) ** 2, axis=1)
        c = np.stack([X[lb == j].mean(0) if (lb == j).any() else c[j] for j in range(len(c))])
    keep = []
    for x in c:
        if all(np.hypot(*(x[:2] - y[:2])) > thr_pos or np.abs(x[2:5] - y[2:5]).max() > app_unit for y in keep):
            keep.append(x)
    return np.array(keep, np.float32)


class Model:
    """World model + cost-to-go + candidate generator + goal test, on raw states (px, app in [0, 1], covered)."""

    def __init__(self, ck, device):
        import torch

        self.K, self.D, self.dev = ck["K"], ck["D"], device
        self.sc = Scale(ck["thr_pos"], ck["thr_app"], ck.get("tol_pos"))
        self.wm = make_wm(self.K, self.D).to(device).eval(); self.wm.load_state_dict(ck["wm"])
        self.absdiff = bool(ck.get("h_absdiff", False))
        self.h = make_h(self.K, self.D, ck.get("h_width", 1024), self.absdiff).to(device).eval(); self.h.load_state_dict(ck["h"])
        self.proto = ck["proto"]
        self.occ_min = ck["occ_min"]; self.cover_rule = ck["cover_rule"]
        self.pos_only = bool(ck.get("event_pos_only", False))
        tol = np.asarray(ck.get("thr_app_id", np.full(self.K, ck["thr_app"])), np.float64)
        self.app_tol = np.where(np.isfinite(tol), tol, 1e9)
        self.max_candidates = int(ck.get("max_candidates", 0))
        self.torch = torch
        support_ck = ck.get("event_support")
        self.event_support = None
        if support_ck is not None:
            self.event_support = make_support(self.K, self.D, support_ck["width"]).to(device).eval()
            self.event_support.load_state_dict(support_ck["weights"])
            self.support_threshold = np.asarray(support_ck["thresholds"], np.float32)
        self.rest_support = ck.get("rest_support")
        if self.rest_support is not None:
            n = max([len(v) for fields in self.rest_support for v in fields if v is not None] or [1])
            self.support_values = np.full((self.K, self.D, n), np.inf, np.float32)
            self.support_mask = np.zeros((self.K, self.D), bool)
            for k, fields in enumerate(self.rest_support):
                for c, values in enumerate(fields):
                    if values is not None:
                        self.support_values[k, c, :len(values)] = values
                        self.support_mask[k, c] = True
            self.support_k, self.support_c = np.indices((self.K, self.D))

    def canonical(self, S):
        if self.rest_support is None:
            return np.asarray(S, np.float32)
        z = np.asarray(S, np.float32)
        i = np.abs(z[..., None] - self.support_values).argmin(-1)
        v = self.support_values[self.support_k, self.support_c, i]
        return np.where(self.support_mask, v, z).astype(np.float32)

    def canonical_entity(self, x, e):
        if self.rest_support is None:
            return np.asarray(x, np.float32)
        z = np.asarray(x, np.float32)
        i = np.abs(z[:, None] - self.support_values[e]).argmin(-1)
        v = self.support_values[e, np.arange(self.D), i]
        return np.where(self.support_mask[e], v, z).astype(np.float32)

    def at_goal(self, S, G, known=None):
        dp = np.linalg.norm(S[..., :2] - G[..., :2], axis=-1) <= self.sc.tol_pos
        da = np.abs(S[..., 2:-1] - G[..., 2:-1]).max(-1) <= self.app_tol
        dc = (S[..., -1] > 0.5) == (G[..., -1] > 0.5)
        ok = np.where(G[..., -1] > 0.5, dc, dp & da & dc)                    # a hidden goal object: only being hidden counts
        if known is not None:
            ok = ok | ~known
        return ok.all(-1)

    def candidates(self, S, G, known=None):
        """-> list of (e, x) for one state S (K, D)."""
        out = []
        for e in range(self.K):
            if self.cover_rule and S[e, -1] > 0.5:
                continue
            xs = ([G[e]] if (known is None or known[e]) and G[e, -1] <= 0.5 else []) + list(self.proto[e])
            for x in xs:
                x = self.canonical_entity(x, e)
                moved = np.hypot(*(x[:2] - S[e, :2])) > self.sc.tol_pos
                changed = moved or np.abs(x[2:-1] - S[e, 2:-1]).max() > self.app_tol[e]
                if not changed:
                    continue
                if moved and self.occ_min > 0:
                    d = np.hypot(S[:, 0] - x[0], S[:, 1] - x[1]); d[e] = np.inf
                    if d.min() < self.occ_min:
                        continue
                y = x.copy(); y[-1] = 0.0
                if not any(np.abs(y - z).max() < 1e-6 for _, z in out if _ == e):
                    out.append((e, y))
        if self.max_candidates and self.event_support is not None and len(out) > self.max_candidates:
            sc = self.support_scores(np.repeat(np.asarray(S)[None], len(out), 0), out)
            keep = np.argsort(-sc)[:self.max_candidates]
            out = [out[i] for i in sorted(keep)]
        return out

    def _proto_table(self):
        """canonical prototypes padded to (K, P, D) with a validity mask (cached)."""
        if getattr(self, "_pt", None) is None:
            P = max([len(p) for p in self.proto] + [0])
            PT = np.zeros((self.K, P, self.D), np.float32); PM = np.zeros((self.K, P), bool)
            for e, pr in enumerate(self.proto):
                for i, x in enumerate(pr):
                    PT[e, i] = self.canonical_entity(x, e); PM[e, i] = True
            self._pt = (PT, PM)
        return self._pt

    def candidates_batch(self, S, G, known=None):
        """Vectorized candidates(): states S (B, K, D), goals G (B, K, D), known None / (K,) / (B, K) -> owner (n,),
        e (n,), x (n, D): per state the same events in the same order as candidates()."""
        S = np.asarray(S, np.float32); G = np.asarray(G, np.float32)
        B, K, D = S.shape
        PT, PM = self._proto_table()
        C = np.concatenate([self.canonical(G)[:, :, None], np.broadcast_to(PT[None], (B,) + PT.shape)], 2)   # (B, K, 1 + P, D)
        V = np.concatenate([np.ones((B, K, 1), bool), np.broadcast_to(PM[None], (B,) + PM.shape)], 2).copy()
        if known is not None:
            V[:, :, 0] &= np.broadcast_to(np.asarray(known, bool), (B, K))
        V[:, :, 0] &= ~(C[:, :, 0, -1] > 0.5)                                    # no goal target for a hidden goal object
        if self.cover_rule:
            V &= ~(S[:, :, None, -1] > 0.5)
        moved = np.hypot(C[..., 0] - S[:, :, None, 0], C[..., 1] - S[:, :, None, 1]) > self.sc.tol_pos   # as candidates()
        V &= moved | (np.abs(C[..., 2:-1] - S[:, :, None, 2:-1]).max(-1) > self.app_tol[None, :, None])
        if self.occ_min > 0:
            d = np.hypot(S[:, None, None, :, 0] - C[:, :, :, None, 0], S[:, None, None, :, 1] - C[:, :, :, None, 1])   # (B, K, 1 + P, K)
            idx = np.arange(K); d[:, idx, :, idx] = np.inf
            V &= ~(moved & (d.min(-1) < self.occ_min))
        Y = C.copy(); Y[..., -1] = 0.0
        for j in range(1, Y.shape[2]):                                          # a later copy of a kept candidate is dropped
            same = np.abs(Y[:, :, :j] - Y[:, :, j:j + 1]).max(-1) < 1e-6
            V[:, :, j] &= ~(same & V[:, :, :j]).any(-1)
        b, e, sl = np.nonzero(V)
        owner, ee, xx = b, e, Y[b, e, sl]
        if self.max_candidates and self.event_support is not None:
            cnt = np.bincount(owner, minlength=B)
            keep = np.ones(len(owner), bool)
            for b0 in np.flatnonzero(cnt > self.max_candidates):
                m = np.flatnonzero(owner == b0)
                sc = self.support_scores(np.repeat(S[b0][None], len(m), 0), list(zip(ee[m], xx[m])))
                keep[m[np.argsort(-sc)[self.max_candidates:]]] = False
            owner, ee, xx = owner[keep], ee[keep], xx[keep]
        return owner, ee, xx

    def support_scores(self, Ss, events):
        t = self.torch
        with t.no_grad():
            s = t.as_tensor(self.sc.norm(np.asarray(Ss, np.float32)), device=self.dev).float()
            e = t.as_tensor(np.array([ev[0] for ev in events]), device=self.dev)
            x = t.as_tensor(self.sc.norm(np.stack([ev[1] for ev in events])), device=self.dev).float()
            return self.event_support(support_features(t, s, e, x)).squeeze(-1).cpu().numpy()

    def step(self, S, events):
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
        nxt[..., -1] = nxt[..., -1] > 0.5
        # the event's definition: the acted entity becomes x (position, appearance; the model keeps its covered bit), and a
        # predicted change within the change thresholds is no change (the model's sub-threshold drift of untouched entities
        # broke goal tests: cube dev loop, no plan found in 20k expansions with the goal one move away)
        S_ = np.asarray(Ss, np.float32)
        same = (np.linalg.norm(nxt[..., :2] - S_[..., :2], axis=-1) <= self.sc.tol_pos) & (np.abs(nxt[..., 2:-1] - S_[..., 2:-1]).max(-1) <= self.app_tol[None])
        nxt[..., :-1] = np.where(same[..., None], S_[..., :-1], nxt[..., :-1])
        rows = np.arange(len(ee))
        nxt[rows, ee, :-1] = np.asarray([ev[1] for ev in events], np.float32)[:, :-1]
        if self.event_support is not None:
            with t.no_grad():
                x_full = t.as_tensor(self.sc.norm(np.stack([ev[1] for ev in events])), device=self.dev).float()
                scores = self.event_support(support_features(t, s, e, x_full)).sigmoid().squeeze(-1).cpu().numpy()
            nxt = apply_event_support(np.asarray(Ss), nxt, scores, self.support_threshold[ee])
        return self.canonical(nxt)

    def heuristic(self, S, G, known=None):
        S = self.canonical(S)
        G = np.broadcast_to(self.canonical(G), S.shape)
        if known is not None:
            G = np.where(known[None, :, None], G, S)                            # unknown goal entities: s - g = 0
        G = np.where(G[..., -1:] > 0.5, np.concatenate([S[..., :-1], G[..., -1:]], -1), G)   # hidden goal: only the covered bit
        t = self.torch
        with t.no_grad():
            s = t.as_tensor(self.sc.norm(S), device=self.dev).float().reshape(len(S), -1)
            g = t.as_tensor(self.sc.norm(np.ascontiguousarray(G)), device=self.dev).float().reshape(len(S), -1)
            return self.h(h_input(t, s, g, self.absdiff)).squeeze(-1).cpu().numpy()

    def key(self, S):
        S = self.canonical(S)
        q = np.concatenate([np.round(S[:, :2] / (self.sc.tol_pos / 2)), np.round(S[:, 2:-1] / (np.minimum(self.app_tol, 1.0)[:, None] / 4)), S[:, -1:]], -1)
        return q.astype(np.int64).tobytes()

    def plan(self, S0, G, known=None, lam=0.6, batch=64, max_expansions=20000, max_depth=30, avoid=()):
        """Batched weighted A*. known (K,) bool: goal entities judged observed (None = all). avoid: states the plan may not
        pass through (the closed loop's recent states, so a new plan does not start by undoing the last event).
        -> (plan or None, info)."""
        S0, G = self.canonical(S0), self.canonical(G)
        if self.at_goal(S0, G, known):
            return [], {"expanded": 0}
        start = np.array(S0, np.float32); start[:, -1] = start[:, -1] > 0.5
        G = np.array(G, np.float32); G[:, -1] = G[:, -1] > 0.5
        h0 = float(self.heuristic(start[None], G, known)[0])
        openl = [(h0, 0, start.tobytes(), [])]
        seen = {self.key(start): 0}
        for A_ in avoid:
            A_ = np.array(self.canonical(A_), np.float32); A_[:, -1] = A_[:, -1] > 0.5
            k_ = self.key(A_)
            if k_ != self.key(start):
                seen[k_] = -1
        expanded, best = 0, (h0, [])
        while openl and expanded < max_expansions:
            nodes = [heapq.heappop(openl) for _ in range(min(batch, len(openl)))]
            src, evs_all, plans = [], [], []
            nodes = [nd for nd in nodes if len(nd[3]) < max_depth]
            if not nodes:
                continue
            SB = np.stack([np.frombuffer(nd[2], np.float32).reshape(self.K, self.D) for nd in nodes])
            ow, ee_, xx_ = self.candidates_batch(SB, np.broadcast_to(G[None], SB.shape), known)
            for o_, e_, x_ in zip(ow, ee_, xx_):
                ev = (int(e_), x_)
                src.append(SB[o_]); evs_all.append(ev); plans.append(nodes[o_][3] + [ev])
            expanded += len(np.unique(ow))
            if not evs_all:
                continue
            succ = self.step_batch(np.stack(src), evs_all).astype(np.float32)
            done = self.at_goal(succ, G[None], known)
            if done.any():
                j = int(np.nonzero(done)[0][0])
                return plans[j], {"expanded": expanded, "depth": len(plans[j])}
            hs = self.heuristic(succ, G, known)
            for sn, pl, hv in zip(succ, plans, hs):
                k = self.key(sn)
                if k in seen and seen[k] <= len(pl):
                    continue
                seen[k] = len(pl)
                heapq.heappush(openl, (lam * len(pl) + float(hv), len(pl), sn.tobytes(), pl))
                if hv < best[0]:
                    best = (float(hv), pl)
        return None, {"expanded": expanded, "best_h": best[0], "best_plan": best[1]}


def stage_wm(a, dev, t0):
    import torch
    import torch.nn.functional as F

    rng = np.random.default_rng(0)
    tr, va = dict(np.load(a.events / "events_train.npz")), dict(np.load(a.events / "events_val.npz"))
    # object events (events_objects.py): only events whose acted entity is known before and after train the model, and
    # an entity's after state enters the loss only where it was observed between this event and the next
    for z in (tr, va):
        if "target_known" in z:
            keep = z["target_known"].astype(bool)
            if a.full_only and z is tr and "before_known" in z:                  # every entity seen before and after the event
                keep &= z["before_known"].astype(bool).all(1) & z["after_known"].astype(bool).all(1)
            for k_ in [k_ for k_, v_ in z.items() if isinstance(v_, np.ndarray) and v_.ndim >= 1 and len(v_) == len(keep)]:
                z[k_] = z[k_][keep]
        z["after_known"] = z["after_known"].astype(bool) if "after_known" in z else np.ones(z["after"].shape[:2], bool)
    rep_ev = json.loads((a.events / "report.json").read_text())
    K, D = tr["before"].shape[1], tr["before"].shape[2]; A = D - 3
    thr_pos, thr_app = float(tr["thr_pos"]), float(tr["thr_app"])
    tol_pos = float(tr["tol_pos"])
    sc = Scale(thr_pos, thr_app, tol_pos)
    app_unit = thr_app if np.isfinite(thr_app) else 0.1
    # object events: per-entity appearance unit (exact entities: half their smallest gap between values; events_objects.py)
    app_unit_id = np.asarray(tr["app_unit_id"], np.float64) if "app_unit_id" in tr else np.full(K, app_unit)
    app_unit_id = np.where(np.isfinite(app_unit_id), app_unit_id, app_unit)
    idx = np.arange(len(tr["e"]))
    cover_rule = bool(tr["before"][idx, tr["e"], -1].mean() < 0.02)
    moves = np.linalg.norm(tr["target"][:, :2] - tr["before"][idx, tr["e"], :2], axis=-1) > tol_pos
    d = np.linalg.norm(tr["before"][..., :2] - tr["target"][:, None, :2], axis=-1); d[idx, tr["e"]] = np.inf
    occ_min = float(np.percentile(d.min(1)[moves], 1)) if moves.any() else 0.0
    if (a.events / "rests_train.npz").exists():                                  # token entities (events.py): fixed places
        rests = dict(np.load(a.events / "rests_train.npz"))
        rests["pos"] = tr["before"][0, :, :2]
        quantized = rep_ev.get("app_threshold_rule", {}).get("rule") == "quantized"
        proto = prototypes(rests, K, A, quantized, thr_pos, app_unit, a.protos, rng)
    else:                                                                         # object entities (events_objects.py)
        proto = prototypes_labels(dict(np.load(a.events / "labels_train.npz")), K, thr_pos, app_unit_id, a.protos, rng)
    wm = make_wm(K, D).to(dev)
    tr["target_in"] = event_input(tr["before"], tr["e"], tr["target"], tol_pos) if a.event_pos_only else tr["target"]
    va["target_in"] = event_input(va["before"], va["e"], va["target"], tol_pos) if a.event_pos_only else va["target"]
    T = {k: torch.as_tensor(tr[k], device=dev).float() for k in ("before", "after", "target_in", "after_known")}
    E = torch.as_tensor(tr["e"], device=dev).long()
    unit = torch.as_tensor(np.c_[np.full((K, 2), tol_pos / 32.0), np.repeat(2 * app_unit_id[:, None], A, 1)], device=dev).float()   # (K, D - 1)
    opt = torch.optim.AdamW(wm.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.wm_steps))))
    for step in range(a.wm_steps):
        i = torch.randint(0, len(E), (a.wm_batch,), device=dev)
        s, x, y = sc.norm(T["before"][i]), sc.norm(T["target_in"][i]), sc.norm(T["after"][i])
        cont, logit = wm(s, E[i], x, logits=True)
        mk = T["after_known"][i]
        loss = ((((cont - y[..., :D - 1]) / unit) ** 2).mean(-1) * mk).sum() / mk.sum().clamp(min=1)             + (F.binary_cross_entropy_with_logits(logit, y[..., D - 1], reduction="none") * mk).sum() / mk.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(wm.parameters(), 1.0); opt.step(); sched.step()
        if step % 5000 == 0:
            print({"wm_step": step, "loss": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 1)}, flush=True)
    wm.eval()
    with torch.no_grad():
        V = {k: torch.as_tensor(va[k], device=dev).float() for k in ("before", "after", "target_in")}
        pred = np.concatenate([sc.denorm(wm(sc.norm(V["before"][s_:s_ + 2048]), torch.as_tensor(va["e"][s_:s_ + 2048], device=dev).long(),
                                             sc.norm(V["target_in"][s_:s_ + 2048])).cpu().numpy()) for s_ in range(0, len(va["e"]), 2048)])
    after, e_, iv = va["after"], va["e"], np.arange(len(va["e"]))
    err_pos = np.linalg.norm(pred[..., :2] - after[..., :2], axis=-1); err_app = np.abs(pred[..., 2:-1] - after[..., 2:-1]).max(-1)
    changed = (np.linalg.norm(after[..., :2] - va["before"][..., :2], axis=-1) > tol_pos) | (np.abs(after[..., 2:-1] - va["before"][..., 2:-1]).max(-1) > app_unit_id)
    acted = np.zeros_like(changed); acted[iv, e_] = True
    ok = (err_pos <= tol_pos) & (err_app <= app_unit_id)
    kn = va["after_known"]                                                       # scored only where the after state was observed
    changed &= kn; acted &= kn
    res = {"K": int(K), "D": int(D), "thr_pos": thr_pos, "tol_pos": tol_pos, "thr_app": thr_app, "cover_rule": cover_rule, "occ_min": occ_min,
           "prototypes_per_entity": [len(p) for p in proto], "val_events": int(len(e_)),
           "acted_within_tol": float(ok[acted].mean()), "side_effect_entities": int((changed & ~acted).sum()),
           "side_effect_within_tol": float(ok[changed & ~acted].mean()) if (changed & ~acted).any() else None,
           "unchanged_within_tol": float(ok[~changed & kn].mean()), "event_exact": float((ok | ~kn).all(1).mean()),
           "after_known_share": float(kn.mean()),
           "covered_acc": float(((pred[..., -1] > 0.5) == (after[..., -1] > 0.5)).mean())}
    print(json.dumps(res), flush=True)
    thr_app_id = np.asarray(tr["thr_app_id"], np.float64) if "thr_app_id" in tr else np.full(K, thr_app)
    model_ck = {"K": int(K), "D": int(D), "thr_pos": thr_pos, "tol_pos": tol_pos, "thr_app": thr_app, "thr_app_id": thr_app_id, "wm": wm.state_dict(),
                "proto": proto, "occ_min": occ_min, "event_pos_only": bool(a.event_pos_only), "h_width": a.h_width, "h_absdiff": bool(a.h_absdiff),
                "cover_rule": cover_rule, "max_candidates": int(a.max_candidates)}
    if a.rest_support:
        model_ck["rest_support"] = finite_rest_support(np.concatenate([tr["before"], tr["after"]]).astype(np.float32))
    model_ck["h"] = make_h(K, D, a.h_width, a.h_absdiff).state_dict()
    model_ck["train_config"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    a.out.mkdir(parents=True, exist_ok=True)
    import torch as _t
    _t.save({**model_ck, "training_stage": "wm_complete", "h_step": 0}, a.out / "wm_stage.pt")
    (a.out / "wm_eval.json").write_text(json.dumps(res, indent=1, default=float) + "\n")


def stage_h(a, dev, t0):
    import torch

    rng = np.random.default_rng(0)
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    tr, va = dict(np.load(a.events / "events_train.npz")), dict(np.load(a.events / "events_val.npz"))
    K, D = ck["K"], ck["D"]
    M = Model(ck, dev)
    sc = M.sc
    h = make_h(K, D, ck["h_width"], ck["h_absdiff"]).to(dev); h_tgt = make_h(K, D, ck["h_width"], ck["h_absdiff"]).to(dev)
    h_tgt.load_state_dict(h.state_dict()); M.h = h_tgt
    states = M.canonical(np.concatenate([tr["before"], tr["after"]]).astype(np.float32))
    if a.h_goals == "walk":
        pS = states[rng.integers(0, len(states), a.h_walk_pool)].copy(); pS[:, :, -1] = pS[:, :, -1] > 0.5
        pG = pS.copy(); kk = rng.integers(0, a.h_walk_max + 1, len(pS))
        for w in range(a.h_walk_max):
            rows, evs = [], []
            act = np.nonzero(kk > w)[0]
            for c0 in range(0, len(act), 8192):                                  # one random candidate per walk (vectorized)
                bb = act[c0:c0 + 8192]
                ow, ee_, xx_ = M.candidates_batch(pG[bb], pG[bb])
                if not len(ow):
                    continue
                cnt = np.bincount(ow, minlength=len(bb)); start = np.r_[0, np.cumsum(cnt)[:-1]]
                has = np.flatnonzero(cnt > 0)
                pick = start[has] + (rng.random(len(has)) * cnt[has]).astype(np.int64)
                rows += bb[has].tolist(); evs += [(int(ee_[q]), xx_[q]) for q in pick]
            for c0 in range(0, len(rows), 8192):
                pG[rows[c0:c0 + 8192]] = M.step_batch(pG[rows[c0:c0 + 8192]], evs[c0:c0 + 8192])
            if w % 5 == 0:
                print({"walk_step": w, "active": len(rows), "min": round((time.time() - t0) / 60, 1)}, flush=True)
        print({"h_walk_pool": len(pS), "walk_mean": float(kk.mean()), "min": round((time.time() - t0) / 60, 1)}, flush=True)
    ho = torch.optim.AdamW(h.parameters(), lr=1e-4, weight_decay=1e-5)
    hl = []
    B = a.h_batch
    for step in range(a.h_steps):
        i = rng.integers(0, len(states), B); j = rng.integers(0, len(states), B)
        S, G = states[i], states[j]
        if a.h_goals == "walk":
            w = rng.integers(0, len(pS), B // 2)
            S = np.concatenate([S[:B // 2], pS[w]]); G = np.concatenate([G[:B // 2], pG[w]])
        owner, ee_, xx_ = M.candidates_batch(S, G)
        evs_all = list(zip(ee_.tolist(), xx_))
        y = np.zeros(len(S), np.float32)
        if evs_all:
            owner = np.array(owner)
            succ = M.step_batch(S[owner], evs_all)
            done = M.at_goal(succ, G[owner])
            with torch.no_grad():
                s_n = torch.as_tensor(sc.norm(succ), device=dev).float().reshape(len(succ), -1)
                g_n = torch.as_tensor(sc.norm(G[owner]), device=dev).float().reshape(len(succ), -1)
                hv = h_tgt(h_input(torch, s_n, g_n, ck["h_absdiff"])).squeeze(-1).cpu().numpy()
            cost = 1 + np.where(done, 0.0, hv)
            y = np.full(len(S), float(a.h_walk_max), np.float32)
            np.minimum.at(y, owner, cost)
        y[M.at_goal(S, G)] = 0.0
        s_n = torch.as_tensor(sc.norm(S), device=dev).float().reshape(len(S), -1)
        g_n = torch.as_tensor(sc.norm(G), device=dev).float().reshape(len(S), -1)
        loss = ((h(h_input(torch, s_n, g_n, ck["h_absdiff"])).squeeze(-1) - torch.as_tensor(y, device=dev)) ** 2).mean()
        ho.zero_grad(set_to_none=True); loss.backward(); ho.step()
        if step % 1000 == 999:
            h_tgt.load_state_dict(h.state_dict())
        if step % 2000 == 0:
            hl.append({"h_step": step, "loss": round(loss.item(), 4), "y_mean": float(y.mean()), "candidates_per_state": len(evs_all) / len(S),
                       "min": round((time.time() - t0) / 60, 1)}); print(hl[-1], flush=True)
        if (step + 1) % 20000 == 0:
            torch.save({**ck, "h": h.state_dict(), "training_stage": "h_partial", "h_step": step + 1}, a.out / f"h_step_{step + 1}.pt")
    ck = {**ck, "h": h.state_dict(), "training_stage": "complete", "h_step": a.h_steps, "h_goals": a.h_goals}
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.out / "model.pt")
    # offline planning check: val (before -> state 1-3 events later in the same episode)
    M = Model(ck, dev)
    ok, depth_ratio, tried = 0, [], 0
    for i in range(0, len(va["e"]) - 4, max(1, len(va["e"]) // 60)):
        for gap in (1, 2, 3):
            if i + gap < len(va["e"]) and va["episode"][i + gap - 1] == va["episode"][i]:
                plan, info = M.plan(va["before"][i], va["after"][i + gap - 1], max_expansions=a.check_expansions)
                tried += 1
                if plan is not None:
                    ok += 1; depth_ratio.append(len(plan) / gap)
    res = {"offline_plan_found": ok / max(1, tried), "offline_plan_len_over_data_len_median": float(np.median(depth_ratio)) if depth_ratio else None,
           "h_log": hl, "minutes": round((time.time() - t0) / 60, 1)}
    (a.out / "model_eval.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "h_log"}, default=float), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("wm", "h"), required=True)
    ap.add_argument("--events", type=Path, required=True, help="events.py output")
    ap.add_argument("--init", type=Path, default=None, help="--stage h: model with event support (train_support.py output)")
    ap.add_argument("--full-only", action="store_true",
                    help="--stage wm: train only on TRAIN events whose every entity is known before and after (the first round of "
                         "the relabel step: the contact rule is right on .906 of fully observed held-out puzzle-3x3 presses, and "
                         "its errors come from partly observed events, which the model then explains)")
    ap.add_argument("--wm-steps", type=int, default=30000)
    ap.add_argument("--wm-batch", type=int, default=512)
    ap.add_argument("--h-steps", type=int, default=150000)
    ap.add_argument("--h-batch", type=int, default=128)
    ap.add_argument("--h-width", type=int, default=2048)
    ap.add_argument("--h-goals", choices=("data", "walk"), default="walk")
    ap.add_argument("--h-walk-max", type=int, default=30)
    ap.add_argument("--h-walk-pool", type=int, default=300000)
    ap.add_argument("--h-absdiff", action="store_true", default=True)
    ap.add_argument("--rest-support", action="store_true", default=True)
    ap.add_argument("--protos", type=int, default=8)
    ap.add_argument("--event-pos-only", action="store_true", default=True)
    ap.add_argument("--max-candidates", type=int, default=32)
    ap.add_argument("--check-expansions", type=int, default=3000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    torch.manual_seed(0)
    t0 = time.time()
    (stage_wm if a.stage == "wm" else stage_h)(a, a.device, t0)


if __name__ == "__main__":
    main()
