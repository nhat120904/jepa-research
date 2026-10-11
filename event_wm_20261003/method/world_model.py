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

from event_support import (apply_event_support, features as support_features, known_entities, make_pair_support, make_support,
                           pair_cost, pair_logits)


def make_wm(K, D, d=128, layers=3, arch="entity", rel=None):
    """arch "entity": transformer over all entity tokens with identity embeddings (any entity may depend on any other).
    arch "rel" (STRUCTURED, user decision 2026-10-10): the next state of entity k after an event on e depends only on k's
    state, e's state, the event's target x and the offset from e to k in place-spacing units (+ whether k is e, whether k
    is a place); no identity, no other entity. With identities and the global state the entity model learned
    state-dependent shortcuts (puzzle-4x5: first plans of 8 events for tasks needing at least 20 presses); an effect that
    depends on where an entity sits relative to the acted one is learned once for all entities and transfers to unseen
    global states. Fixed layouts keep pair-specific effects (one offset = one pair). rel = {"h_sp": px, "is_place": [K]}."""
    import torch
    import torch.nn as nn

    if arch == "rel":
        h_sp = float(rel["h_sp"]); is_place = torch.as_tensor(np.asarray(rel["is_place"], np.float32))
        centres = torch.arange(-3.0, 3.01, 0.5)                                   # RBF centres per axis, spacing units

        class RelWM(nn.Module):
            def __init__(self):
                super().__init__()
                self.K, self.D = K, D
                self.register_buffer("place", is_place.clone())
                self.register_buffer("cen", centres.clone())
                n_in = 3 * D + 2 * len(centres) + 3 + 2
                self.mlp = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                         nn.Linear(256, D))

            def forward(self, s, e, x, logits=False):
                """s (B, K, D) normalised, e (B,), x (B, D) normalised -> (B, K, D) next state (covered prob or logit)."""
                B = s.shape[0]
                se = s[torch.arange(B, device=s.device), e.long()]                # (B, D) the acted entity
                off = (s[..., :2] - se[:, None, :2]) * (32.0 / h_sp)               # (B, K, 2) offset in place spacings
                rbf = torch.exp(-((off[..., :, None] - self.cen) ** 2) / (2 * 0.25 ** 2)).flatten(-2)   # (B, K, 2 * n_cen)
                dist = off.norm(dim=-1, keepdim=True)
                acted = torch.nn.functional.one_hot(e.long(), K).float()[..., None]
                plc = self.place[None, :, None].expand(B, K, 1)
                inp = torch.cat([s, se[:, None].expand(B, K, D), x[:, None].expand(B, K, D), rbf, off.clamp(-4, 4) / 4, dist.clamp(max=6) / 6,
                                 acted, plc], -1)
                o = self.mlp(inp)
                cont = s[..., :D - 1] + o[..., :D - 1]
                if logits:
                    return cont, o[..., D - 1]
                return torch.cat([cont, torch.sigmoid(o[..., D - 1:])], -1)

        return RelWM()

    if arch == "rel2":
        # rel + MOVERS (G3, 2026-10-10): translation invariant (no absolute position enters), with the offsets of k from the
        # acted entity AND from its target position and the event's displacement x - e: a cube placed on another covers it
        # (offset to the target ~ 0), a cube inside a drawer moves with it (the displacement), a light toggles its
        # neighbours (offsets to e = to x, displacement 0, as rel)
        h_sp = float(rel["h_sp"]); is_place = torch.as_tensor(np.asarray(rel["is_place"], np.float32))
        centres = torch.arange(-3.0, 3.01, 0.5)

        class Rel2WM(nn.Module):
            def __init__(self):
                super().__init__()
                self.K, self.D = K, D
                self.register_buffer("place", is_place.clone())
                self.register_buffer("cen", centres.clone())
                n_in = 3 * (D - 2) + 2 + 2 * (2 * len(centres) + 2 + 1) + 2
                self.mlp = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                         nn.Linear(256, D))

            def feats(self, off):
                rbf = torch.exp(-((off[..., :, None] - self.cen) ** 2) / (2 * 0.25 ** 2)).flatten(-2)
                return [rbf, off.clamp(-4, 4) / 4, off.norm(dim=-1, keepdim=True).clamp(max=6) / 6]

            def forward(self, s, e, x, logits=False):
                B = s.shape[0]
                se = s[torch.arange(B, device=s.device), e.long()]
                sc_ = 32.0 / h_sp
                off_e = (s[..., :2] - se[:, None, :2]) * sc_
                off_x = (s[..., :2] - x[:, None, :2]) * sc_
                disp = ((x[:, :2] - se[:, :2]) * sc_).clamp(-6, 6) / 6
                acted = torch.nn.functional.one_hot(e.long(), K).float()[..., None]
                plc = self.place[None, :, None].expand(B, K, 1)
                inp = torch.cat([s[..., 2:], se[:, None, 2:].expand(B, K, D - 2), x[:, None, 2:].expand(B, K, D - 2), disp[:, None].expand(B, K, 2)]
                                + self.feats(off_e) + self.feats(off_x) + [acted, plc], -1)
                o = self.mlp(inp)
                cont = s[..., :D - 1] + o[..., :D - 1]
                if logits:
                    return cont, o[..., D - 1]
                return torch.cat([cont, torch.sigmoid(o[..., D - 1:])], -1)

        return Rel2WM()

    if arch in ("pair", "pairabs", "pairg"):
        # PAIRWISE (2026-10-10): the next state of k depends on k and the acted entity e only (their states, the target x,
        # their offsets in place spacings AND image units, the displacement x - e) plus learned identities of k and e with
        # dropout .5. No third entity enters, so no global-state shortcut (the puzzle failure of "entity"); pair identities
        # carry pair-specific effects that are not a function of the offset (scene locks: button1 -> window appearance, the
        # window sliding so its offset to the button varies; rel2 side effects .05 vs entity .38)
        h_sp = float(rel["h_sp"]); is_place = torch.as_tensor(np.asarray(rel["is_place"], np.float32))
        centres = torch.arange(-3.0, 3.01, 0.5)
        di = 16

        class PairWM(nn.Module):
            def __init__(self):
                super().__init__()
                self.K, self.D = K, D
                self.register_buffer("place", is_place.clone())
                self.register_buffer("cen", centres.clone())
                self.id_k, self.id_e = nn.Embedding(K, di), nn.Embedding(K, di)
                n_in = 3 * (D - 2) + 2 + 2 + 2 * (2 * len(centres) + 2 + 1 + 2) + 2 + 2 * di + (6 if arch == "pairabs" else 0)
                self.gated = arch == "pairg"
                self.mlp = nn.Sequential(nn.Linear(n_in, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                                         nn.Linear(256, D + (1 if self.gated else 0)))

            def feats(self, d_img):
                off = d_img * (32.0 / h_sp)                                       # place spacings
                rbf = torch.exp(-((off[..., :, None] - self.cen) ** 2) / (2 * 0.25 ** 2)).flatten(-2)
                return [rbf, off.clamp(-4, 4) / 4, off.norm(dim=-1, keepdim=True).clamp(max=6) / 6, d_img / 2]

            def forward(self, s, e, x, logits=False, gate=False):
                """pairg: an extra CHANGE logit per entity (sparse effects: an event changes few entities); at inference the
                residual applies only where it is > 0, so an entity predicted unchanged keeps its state exactly (the pairwise
                models drifted unaffected cubes by more than tol_pos: kept unchanged .73-.81 vs .86 for the entity model)."""
                B = s.shape[0]
                el = e.long()
                se = s[torch.arange(B, device=s.device), el]
                d_e = s[..., :2] - se[:, None, :2]                                # normalised image units ([-1, 1] = 64 px)
                d_x = s[..., :2] - x[:, None, :2]
                disp = x[:, :2] - se[:, :2]
                acted = torch.nn.functional.one_hot(el, K).float()[..., None]
                plc = self.place[None, :, None].expand(B, K, 1)
                ik = self.id_k.weight[None].expand(B, K, di); ie = self.id_e(el)[:, None].expand(B, K, di)
                if self.training:                                                # identity dropout: the relative pathway carries shared effects
                    ik = ik * (torch.rand(B, K, 1, device=s.device) > 0.5).float()
                    ie = ie * (torch.rand(B, 1, 1, device=s.device) > 0.5).float()
                inp = torch.cat([s[..., 2:], se[:, None, 2:].expand(B, K, D - 2), x[:, None, 2:].expand(B, K, D - 2),
                                 disp[:, None].expand(B, K, 2) / 2, (disp * (32.0 / h_sp)).clamp(-6, 6)[:, None].expand(B, K, 2) / 6]
                                + self.feats(d_e) + self.feats(d_x) + [acted, plc, ik, ie]
                                # pairabs: + absolute positions of k, e and the target (workspace, where a drawer / window sits)
                                + ([s[..., :2], se[:, None, :2].expand(B, K, 2), x[:, None, :2].expand(B, K, 2)] if arch == "pairabs" else []), -1)
                o = self.mlp(inp)
                if self.gated and not logits:
                    cont = s[..., :D - 1] + o[..., :D - 1] * (o[..., D:D + 1] > 0).float()
                    return torch.cat([cont, torch.sigmoid(o[..., D - 1:D])], -1)
                cont = s[..., :D - 1] + o[..., :D - 1]
                if logits:
                    return (cont, o[..., D - 1], o[..., D]) if (gate and self.gated) else (cont, o[..., D - 1])
                return torch.cat([cont, torch.sigmoid(o[..., D - 1:D])], -1)

        return PairWM()

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


def _cluster_positions(X, P, tol_pos, rng):
    """k-means (P) of rows X (n, D) on their positions, centres of clusters with >= 2 members, deduplicated within tol_pos."""
    if len(X) < 2:
        return []
    X = X[rng.permutation(len(X))[:20000]]
    c = X[rng.choice(len(X), min(P, len(X)), replace=False), :2].copy()
    for _ in range(30):
        lb = np.argmin(np.linalg.norm(X[:, None, :2] - c[None], axis=-1), axis=1)
        c = np.stack([X[lb == j, :2].mean(0) if (lb == j).any() else c[j] for j in range(len(c))])
    lb = np.argmin(np.linalg.norm(X[:, None, :2] - c[None], axis=-1), axis=1)
    out = []
    for j in range(len(c)):
        if (lb == j).sum() >= 2 and all(np.hypot(*(c[j] - o[:2])) > tol_pos for o in out):
            r = np.median(X[lb == j], 0).astype(np.float32); r[:2] = c[j]; r[-1] = 0.0
            out.append(r)
    return out


def hidden_goal_support(tr, proto, is_place, tol_pos, P, rng):
    """HIDDEN GOALS (2026-10-11; cube-triple task 5 = a 3-stack, scene tasks 4-5 = the cube in the closed drawer: the planner
    proposed no target for an entity hidden in the goal). From TRAIN events, for every entity that moves:
      prototypes += its PLACEMENTS (k-means of the targets of its own moving events) and its PRE-HIDING positions (k-means of
        where it was when another entity's event hid it: inside a drawer);
      COVER relations (m, j, offset, n): m's moving events that hid j, offset = m's target - j's position, the mode of those
        offsets (points within 2 tol_pos of the median, re-centred), kept when that mode holds >= 5 events and a quarter of them (a
        cube put on a cube: 38-72% within 2 px of (0.1, -2.5) px; a drawer closing over a cube: spread over 8-17 px, dropped).
    -> (proto, rel_cover)."""
    proto = [np.asarray(p, np.float32) for p in proto]
    e, B, A, X = tr["e"], tr["before"], tr["after"], tr["target"]
    bk = tr["before_known"].astype(bool) if "before_known" in tr else np.ones(B.shape[:2], bool)
    idx = np.arange(len(e))
    moved = np.linalg.norm(X[:, :2] - B[idx, e, :2], axis=-1) > tol_pos
    hide = (B[..., -1] < 0.5) & (A[..., -1] > 0.5) & bk                           # (n, K) became hidden
    tk = tr["target_known"].astype(bool) if "target_known" in tr else np.ones(len(e), bool)   # unknown targets read (0, 0)
    rel_cover = []
    for m in np.flatnonzero(~is_place):
        put = X[(e == m) & moved & tk]
        pre = B[hide[:, m] & (e != m), m]
        add = _cluster_positions(put, P, tol_pos, rng) + _cluster_positions(pre, max(2, P // 4), tol_pos, rng)
        add = [r for r in add if all(np.hypot(*(r[:2] - p[:2])) > tol_pos for p in proto[m])]
        if add:
            proto[m] = np.concatenate([proto[m].reshape(-1, 6), np.stack(add)]).astype(np.float32) if len(proto[m]) else np.stack(add)
        for j in np.flatnonzero(~is_place):
            if j == m:
                continue
            ev = (e == m) & moved & hide[:, j] & tk
            if ev.sum() < 5:
                continue
            off = X[ev, :2] - B[ev, j, :2]
            c = np.median(off, 0)
            for _ in range(5):
                near = np.linalg.norm(off - c, axis=-1) <= 2 * tol_pos
                if not near.any():
                    break
                c = np.median(off[near], 0)
            near = np.linalg.norm(off - c, axis=-1) <= 2 * tol_pos
            if near.sum() >= 5 and near.mean() >= 0.25:                         # ~3x a diffuse spread's share
                rel_cover.append((int(m), int(j), c.astype(np.float32), int(near.sum())))
    return proto, rel_cover


class Model:
    """World model + cost-to-go + candidate generator + goal test, on raw states (px, app in [0, 1], covered)."""

    def __init__(self, ck, device):
        import torch

        self.K, self.D, self.dev = ck["K"], ck["D"], device
        self.sc = Scale(ck["thr_pos"], ck["thr_app"], ck.get("tol_pos"))
        self.wm = make_wm(self.K, self.D, arch=ck.get("wm_arch", "entity"), rel=ck.get("wm_rel")).to(device).eval(); self.wm.load_state_dict(ck["wm"])
        self.absdiff = bool(ck.get("h_absdiff", False))
        self.h = make_h(self.K, self.D, ck.get("h_width", 1024), self.absdiff).to(device).eval(); self.h.load_state_dict(ck["h"])
        self.proto = ck["proto"]
        self.rel_cover = [(int(m), int(j), np.asarray(o, np.float32)) for m, j, o, *_ in ck.get("rel_cover", [])]   # m put covers j
        self.hidden_targets = {}                                                 # per plan: targets of hidden goal entities
        self.feas_weight = 0.0                                                   # plan cost per -log support (the loop sets it)
        self.feas_margin = np.zeros(self.K, np.float32)                          # per acted entity: cost below it is free
        self.occ_min = ck["occ_min"]; self.cover_rule = ck["cover_rule"]
        self.pos_only = bool(ck.get("event_pos_only", False))
        tol = np.asarray(ck.get("thr_app_id", np.full(self.K, ck["thr_app"])), np.float64)
        # floor 1e-4: exact identities have thresholds of ~2e-6, but the support values are rounded to 4 decimals while a
        # mover's reading keeps its identity colour at full precision, so every cube differed from its own prediction by ~4e-6
        # (2026-10-11 cube VAL events: predicted change sets exact .015 against raw states vs .762 against canonical ones; the
        # oracle loop saw every event as not as predicted). State changes differ by > .05.
        self.app_tol = np.maximum(np.where(np.isfinite(tol), tol, 1e9), 1e-4)
        self.pred_tol = self.app_tol.copy()                                      # prediction-vs-reading tolerance (set_prediction_tolerance)
        self.app_derived = np.zeros(self.K, bool)                                # appearance determined by the others (set_derived_appearance)
        self.relevant = np.ones((self.K, self.K), bool)                          # [e, k]: k bears on events of e (set_failure_relevance)
        self.max_candidates = int(ck.get("max_candidates", 0))
        self.torch = torch
        support_ck = ck.get("event_support")
        self.event_support = None
        self.support_kind = (support_ck or {}).get("kind", "joint")
        if support_ck is not None:
            mk = make_pair_support if self.support_kind == "pair" else make_support
            self.event_support = mk(self.K, self.D, support_ck["width"]).to(device).eval()
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

    def set_prediction_tolerance(self, events_npz, q=0.75, min_n=20, chunk=2048):
        """Appearance tolerance for comparing WM PREDICTIONS with readings (the goal test on imagined states; the loop's
        as-predicted and surprise tests): per entity, the q-quantile of the WM's appearance error on the entity when it changed
        as a SIDE effect (not the acted entity, whose target is set) in the given events; at least the event threshold app_tol,
        at most half the median appearance change of those events, so distinct appearance modes stay apart. Entities with fewer
        than min_n such events keep app_tol. (Scene: a button press recolours the drawer / window handle; the WM predicts the
        new colour to ~.004-.01 while app_tol is .0039, so every press was 'not as predicted' and goal tests failed.) -> (K,)"""
        z = np.load(events_npz)
        keep = z["target_known"].astype(bool) if "target_known" in z else np.ones(len(z["e"]), bool)
        e, b4, af, tg = z["e"][keep], z["before"][keep], z["after"][keep], z["target"][keep]
        bk = z["before_known"][keep] if "before_known" in z else np.ones(b4.shape[:2], bool)
        ak = z["after_known"][keep] if "after_known" in z else np.ones(b4.shape[:2], bool)
        pred = np.concatenate([self.step_batch(self.canonical(b4[i:i + chunk]), list(zip(e[i:i + chunk].tolist(), tg[i:i + chunk])))
                               for i in range(0, len(e), chunk)]) if len(e) else np.zeros_like(b4)
        tol = self.app_tol.copy()
        for k in range(self.K):
            d = np.abs(af[:, k, 2:-1] - b4[:, k, 2:-1]).max(-1)
            m = (e != k) & bk[:, k] & ak[:, k] & (d > self.app_tol[k])
            if m.sum() >= min_n:
                err = np.abs(pred[m, k, 2:-1] - af[m, k, 2:-1]).max(-1)
                tol[k] = float(np.clip(np.quantile(err, q), self.app_tol[k], max(self.app_tol[k], 0.5 * np.median(d[m]))))
        self.pred_tol = tol.astype(np.float32)
        return self.pred_tol

    def set_feasibility_margin(self, events_npz, q=0.9, chunk=2048):
        """Per acted entity, the q-quantile of the event cost (-log support) of GENUINE events in the given events: only the
        cost above it enters plans. Typical events then cost nothing and the search stays guided by the cost-to-go (a cost
        on every event made A* nearly breadth-first: puzzle 4x5 task 3, no plan in 20k expansions, 0/5 from 6/6). -> (K,)"""
        z = np.load(events_npz)
        keep = z["target_known"].astype(bool) if "target_known" in z else np.ones(len(z["e"]), bool)
        e, b4, tg = z["e"][keep], z["before"][keep], z["target"][keep]
        c = np.concatenate([-np.log(np.maximum(self.step_batch(self.canonical(b4[i:i + chunk]), list(zip(e[i:i + chunk].tolist(), tg[i:i + chunk])),
                                                               return_support=True)[1], 1e-6)) for i in range(0, len(e), chunk)]) if len(e) else np.zeros(0)
        self.feas_margin = np.array([float(np.quantile(c[e == k], q)) if (e == k).sum() >= 10 else float(np.quantile(c, q)) if len(c) else 0.0
                                     for k in range(self.K)], np.float32)
        return self.feas_margin

    def set_derived_appearance(self, events_dir, thr=0.5, nn=10, max_fit=8000, max_score=2000, seed=0):
        """DERIVED APPEARANCES: an entity whose appearance is predictable from the other entities' states and its own position
        (kNN regression on event before / after states, fit on TRAIN, R^2 on VAL >= thr) is a function of them: a goal or a
        prediction that matches on the others matches on it. Greedy: the most predictable appearance is flagged and leaves the
        predictors of the rest, so of two appearances that predict each other only one is flagged. Entities with a constant
        appearance are never flagged. (Scene: the drawer / window handle is red while its button is red; the WM predicts the
        new colour of the handle to ~.01 and every press looked "not as predicted"; puzzle lights are independent.) -> (K,)"""
        from scipy.spatial import cKDTree
        rng = np.random.default_rng(seed)

        def states(split, cap):
            z = np.load(Path(events_dir) / f"events_{split}.npz")
            S_ = np.concatenate([z["before"], z["after"]]); kn = np.concatenate([z["before_known"], z["after_known"]]) if "before_known" in z else np.ones(S_.shape[:2], bool)
            S_ = S_[kn.all(1)]
            return S_[rng.permutation(len(S_))[:cap]] if len(S_) > cap else S_

        Str, Sva = states("train", max_fit), states("val", max_score)
        use, flagged = np.ones(self.K, bool), np.zeros(self.K, bool)
        self.derived_r2 = {}
        for _ in range(self.K):
            r2 = {}
            for k in range(self.K):
                if flagged[k] or len(Sva) < 50 or Str[:, k, 2:-1].std(0).max() < 1e-4:
                    continue
                cols = [Str[:, k, :2]] + [Str[:, j, :2] for j in range(self.K) if j != k] + [Str[:, j, -1:] for j in range(self.K) if j != k] \
                    + [Str[:, j, 2:-1] for j in range(self.K) if j != k and use[j]]
                colv = [Sva[:, k, :2]] + [Sva[:, j, :2] for j in range(self.K) if j != k] + [Sva[:, j, -1:] for j in range(self.K) if j != k] \
                    + [Sva[:, j, 2:-1] for j in range(self.K) if j != k and use[j]]
                Ftr, Fva = np.concatenate(cols, 1), np.concatenate(colv, 1)
                sd = Ftr.std(0) + 1e-6
                _, idx = cKDTree(Ftr / sd).query(Fva / sd, k=nn)
                y, yv = Str[:, k, 2:-1], Sva[:, k, 2:-1]
                r2[k] = float(1 - ((y[idx].mean(1) - yv) ** 2).sum() / max(((yv - yv.mean(0)) ** 2).sum(), 1e-12))
            if not self.derived_r2:
                self.derived_r2 = {int(k): round(v, 3) for k, v in r2.items()}
            if not r2 or max(r2.values()) < thr:
                break
            kb = max(r2, key=r2.get)
            flagged[kb] = True; use[kb] = False
        self.app_derived = flagged
        return flagged

    def set_failure_relevance(self, events_npz, thr=0.3, chunk=2048, seed=0):
        """RELEVANCE of entity k to events on entity e, from the pairwise support: the mean cost (nats) that putting k in the
        state of another event adds to genuine VAL events of e. A failed event is then remembered against the relevant
        entities only (and e itself): changing an unrelated one (move the cube away and back, press the other button) no
        longer re-allows a locked window (scene task 2, c11). Joint support or too few events: every entity relevant. -> (K, K)"""
        rel = np.ones((self.K, self.K), bool)
        self.relevance = None
        if self.event_support is None or self.support_kind != "pair":
            self.relevant = rel
            return rel
        t, rng = self.torch, np.random.default_rng(seed)
        z = np.load(events_npz)
        keep = z["target_known"].astype(bool) if "target_known" in z else np.ones(len(z["e"]), bool)
        e_all, S_all, X_all = z["e"][keep], self.canonical(z["before"][keep]), z["target"][keep]
        kn_all = known_entities(S_all)
        score = np.zeros((self.K, self.K), np.float32)
        for e in range(self.K):
            idx = np.flatnonzero(e_all == e)
            if len(idx) < 10:
                continue
            for k in range(self.K):
                if k == e:
                    continue
                donor = rng.integers(0, len(S_all), len(idx))
                Sw = S_all[idx].copy(); Sw[:, k] = S_all[donor, k]
                kn = kn_all[idx].copy(); kn[:, k] = kn_all[donor, k]
                with t.no_grad():
                    s_ = t.as_tensor(self.sc.norm(Sw), device=self.dev).float()
                    x_ = t.as_tensor(self.sc.norm(X_all[idx]), device=self.dev).float()
                    lg = pair_logits(self.event_support, t, s_, t.as_tensor(np.full(len(idx), e), device=self.dev), x_)[:, k]
                    c = (t.nn.functional.softplus(-lg) - np.log(2)).clamp(min=0).cpu().numpy()
                score[e, k] = float(np.mean(np.where(kn[:, k], c, 0.0)))
            rel[e] = score[e] >= thr
            rel[e, e] = True
        self.relevant, self.relevance = rel, score
        return rel

    def at_goal(self, S, G, known=None):
        dp = np.linalg.norm(S[..., :2] - G[..., :2], axis=-1) <= self.sc.tol_pos
        da = (np.abs(S[..., 2:-1] - G[..., 2:-1]).max(-1) <= self.pred_tol) | self.app_derived   # imagined states: the WM's precision
        dc = (S[..., -1] > 0.5) == (G[..., -1] > 0.5)
        ok = np.where(G[..., -1] > 0.5, dc, dp & da & dc)                    # a hidden goal object: only being hidden counts
        # ... unless the cover relations place it (hidden_goal_targets, set by plan()): then hidden AND near one of those
        # places (2 tol_pos, the cover mode radius); "hidden anywhere" let the planner hide the 3-stack's blocks elsewhere
        for k, ps in self.hidden_targets.items():
            dk = np.linalg.norm(S[..., k, None, :2] - np.asarray(ps, np.float32), axis=-1).min(-1)   # (...)
            ok[..., k] = ok[..., k] & (dk <= 2 * self.sc.tol_pos)
        if known is not None:
            ok = ok | ~known
        return ok.all(-1)

    def same_target(self, x, y, e):
        """two targets of entity e within the event thresholds (position tol_pos, appearance app_tol)."""
        x, y = np.asarray(x, np.float32), np.asarray(y, np.float32)
        return bool(np.hypot(*(x[:2] - y[:2])) <= self.sc.tol_pos and np.abs(x[2:-1] - y[2:-1]).max() <= self.app_tol[e])

    def candidates(self, S, G, known=None):
        """-> list of (e, x) for one state S (K, D)."""
        out = []
        for e in range(self.K):
            if self.cover_rule and S[e, -1] > 0.5:
                continue
            xs = ([G[e]] if (known is None or known[e]) and G[e, -1] <= 0.5 else []) + list(self.proto[e])
            for x in xs:
                x = self.canonical_entity(x, e)
                if self.app_derived[e]:                                         # derived appearance: kept (candidates_batch)
                    x = x.copy(); x[2:-1] = S[e, 2:-1]
                moved = np.hypot(*(x[:2] - S[e, :2])) > self.sc.tol_pos
                changed = moved or (np.abs(x[2:-1] - S[e, 2:-1]).max() > self.app_tol[e] and not self.app_derived[e])
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

    def hidden_goal_targets(self, G, known=None, depth=None):
        """targets for the entities hidden in goal G: under a visible goal entity that covers them (cover relations m covers j:
        j sits at m's goal - offset), through chains of hidden ones (a 3-stack: the bottom block under the middle under the
        visible top) -> {entity: [(2,) positions]}."""
        G = np.asarray(G, np.float32)
        ok = np.ones(self.K, bool) if known is None else np.array(known, bool)  # a copy: &= below must not touch the caller's
        ok &= np.abs(G[:, :2]).sum(-1) > 0                                       # an unread goal entity sits at (0, 0)
        cov = {}
        for m, j, o in self.rel_cover:
            cov.setdefault(j, []).append((m, o))                                 # j can be covered by m with offset o
        hid = [k for k in range(self.K) if G[k, -1] > 0.5]
        out = {}

        def above(k, d_, seen):
            """(position, depth) where k must be for the chain above it to end at a visible goal; depth 1 = right under it."""
            res = []
            for m, o in cov.get(k, []):
                if m in seen:
                    continue
                if G[m, -1] <= 0.5:
                    if ok[m]:
                        res.append((G[m, :2] - o, d_))
                elif d_ < len(hid):
                    res += [(p - o, dd) for p, dd in above(m, d_ + 1, seen | {m})]
            return res

        for k in hid:
            ps = []
            # depth (k -> rank under the visible top, from partial goal readings: a lower side face = deeper): only chains of that
            # length; without it a 3-stack goal accepted either order of its two hidden cubes (cube task 5: wrong orders)
            cand = [p for p, dd in above(k, 1, {k}) if depth is None or k not in depth or dd == depth[k]]
            for p in cand:
                if all(np.hypot(*(p - q)) > self.sc.tol_pos for q in ps):
                    ps.append(np.asarray(p, np.float32))
            if ps:
                out[k] = ps
        return out

    def candidates_batch(self, S, G, known=None):
        """Vectorized candidates(): states S (B, K, D), goals G (B, K, D), known None / (K,) / (B, K) -> owner (n,),
        e (n,), x (n, D): per state the same events in the same order as candidates(), then (cover relations, hidden goal
        targets) the RELATIVE targets: on top of a visible entity (its position + the cover offset) and the targets of
        entities hidden in the goal (hidden_goal_targets, set by plan())."""
        S = np.asarray(S, np.float32); G = np.asarray(G, np.float32)
        B, K, D = S.shape
        PT, PM = self._proto_table()
        C = np.concatenate([self.canonical(G)[:, :, None], np.broadcast_to(PT[None], (B,) + PT.shape)], 2)   # (B, K, 1 + P, D)
        V = np.concatenate([np.ones((B, K, 1), bool), np.broadcast_to(PM[None], (B,) + PM.shape)], 2).copy()
        ex = [[] for _ in range(K)]
        for m, j, o in self.rel_cover:
            ex[m].append((S[:, j, :2] + o, (S[:, j, -1] < 0.5) & (np.abs(S[:, j, :2]).sum(-1) > 0)))
        for m, ps in self.hidden_targets.items():
            for p in ps:
                ex[m].append((np.broadcast_to(p, (B, 2)), np.ones(B, bool)))
        R = max(len(x_) for x_ in ex)
        if R:
            EX = np.repeat(S[:, :, None], R, 2).copy(); EX[..., -1] = 0.0
            EV = np.zeros((B, K, R), bool)
            for m in range(K):
                for r, (p, v) in enumerate(ex[m]):
                    EX[:, m, r, :2] = p; EV[:, m, r] = v
            C = np.concatenate([C, EX], 2); V = np.concatenate([V, EV], 2)
        if self.app_derived.any():                                               # a derived appearance is no target: only
            C = C.copy()                                                         # the position moves, the look is kept
            C[:, self.app_derived, :, 2:-1] = S[:, self.app_derived, None, 2:-1]
        if known is not None:
            V[:, :, 0] &= np.broadcast_to(np.asarray(known, bool), (B, K))
        V[:, :, 0] &= ~(C[:, :, 0, -1] > 0.5)                                    # no goal target for a hidden goal object
        if self.cover_rule:
            V &= ~(S[:, :, None, -1] > 0.5)
        moved = np.hypot(C[..., 0] - S[:, :, None, 0], C[..., 1] - S[:, :, None, 1]) > self.sc.tol_pos   # as candidates()
        # a derived appearance (set_derived_appearance) is no event target: it follows the other entities
        V &= moved | ((np.abs(C[..., 2:-1] - S[:, :, None, 2:-1]).max(-1) > self.app_tol[None, :, None]) & ~self.app_derived[None, :, None])
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
        """joint support: logits; pairwise support: minus the event cost (nats). Higher = more plausible."""
        t = self.torch
        with t.no_grad():
            s = t.as_tensor(self.sc.norm(np.asarray(Ss, np.float32)), device=self.dev).float()
            e = t.as_tensor(np.array([ev[0] for ev in events]), device=self.dev)
            x = t.as_tensor(self.sc.norm(np.stack([ev[1] for ev in events])), device=self.dev).float()
            if self.support_kind == "pair":
                kn = t.as_tensor(known_entities(np.asarray(Ss, np.float32)), device=self.dev)
                return (-pair_cost(t, pair_logits(self.event_support, t, s, e, x), kn)).cpu().numpy()
            return self.event_support(support_features(t, s, e, x)).squeeze(-1).cpu().numpy()

    def step(self, S, events):
        return self.step_batch(np.repeat(np.asarray(S)[None], len(events), 0), events)

    def step_batch(self, Ss, events, return_support=False):
        """Ss (n, K, D) aligned with events list of (e, x) -> successors (n, K, D); one forward pass. return_support: also the
        event-support probabilities (n,) (ones without a support model)."""
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
                if self.support_kind == "pair":                                  # exp(-cost): plan() takes -log of it
                    kn = t.as_tensor(known_entities(np.asarray(Ss, np.float32)), device=self.dev)
                    scores = t.exp(-pair_cost(t, pair_logits(self.event_support, t, s, e, x_full), kn)).cpu().numpy()
                else:
                    scores = self.event_support(support_features(t, s, e, x_full)).sigmoid().squeeze(-1).cpu().numpy()
            nxt = apply_event_support(np.asarray(Ss), nxt, scores, self.support_threshold[ee])
        else:
            scores = np.ones(len(ee), np.float32)
        return (self.canonical(nxt), scores) if return_support else self.canonical(nxt)

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
        qa = np.round(S[:, 2:-1] / (np.minimum(self.app_tol, 1.0)[:, None] / 4)) * ~self.app_derived[:, None]   # derived: not a state
        q = np.concatenate([np.round(S[:, :2] / (self.sc.tol_pos / 2)), qa, S[:, -1:]], -1)
        return q.astype(np.int64).tobytes()

    def plan(self, S0, G, known=None, lam=0.6, batch=64, max_expansions=20000, max_depth=30, avoid=(), forbid=(), hidden_depth=None):
        """Batched weighted A*. known (K,) bool: goal entities judged observed (None = all). avoid: states the plan may not
        pass through (the closed loop's recent states, so a new plan does not start by undoing the last event). forbid:
        (state, e) of events that failed in the closed loop: no event on e from a state within the tolerances of that state,
        at any depth (a locked drawer fails for every target; forbidding only the first event let plans insert a no-op
        button pair first). -> (plan or None, info)."""
        # forbid entries: (state (K, D), e). A node matches a failed state when every entity is within the event / prediction
        # tolerances of it (the same test as the goal test, derived appearances ignored): exact state keys missed repeats,
        # because reading noise of a few thousandths in a handle colour gave every new belief a new key (scene task 2: the
        # locked window tried 4-6 times); comparing the derived handle colour let no-op pairs (press a button twice, open and
        # close the window) change the predicted colour and re-allow the locked drawer (scene c9)
        fb_S = np.stack([np.asarray(f_[0], np.float32) for f_ in forbid]) if len(forbid) else None
        fb_e = np.array([int(f_[1]) for f_ in forbid]) if len(forbid) else None
        S0, G = self.canonical(S0), self.canonical(G)
        self.hidden_targets = self.hidden_goal_targets(G, known, hidden_depth)   # targets of entities hidden in the goal
        if self.at_goal(S0, G, known):
            return [], {"expanded": 0}
        start = np.array(S0, np.float32); start[:, -1] = start[:, -1] > 0.5
        G = np.array(G, np.float32); G[:, -1] = G[:, -1] > 0.5
        h0 = float(self.heuristic(start[None], G, known)[0])
        openl = [(h0, 0, start.tobytes(), [], 0.0, False)]
        seen = {self.key(start): 0}
        for A_ in avoid:
            A_ = np.array(self.canonical(A_), np.float32); A_[:, -1] = A_[:, -1] > 0.5
            k_ = self.key(A_)
            if k_ != self.key(start):
                seen[k_] = -1
        expanded, best = 0, (h0, [])
        while openl and expanded < max_expansions:
            nodes = []
            while openl and len(nodes) < batch:                                  # a goal is returned when it is POPPED (the
                nd = heapq.heappop(openl)                                        # cheapest path to it, with the feasibility
                if nd[5]:                                                        # cost); nodes popped before it are expanded first
                    if not nodes:
                        return nd[3], {"expanded": expanded, "depth": len(nd[3]), "cost": round(float(nd[0]), 3)}
                    heapq.heappush(openl, nd)
                    break
                nodes.append(nd)
            src, evs_all, plans, src_owner = [], [], [], []
            nodes = [nd for nd in nodes if len(nd[3]) < max_depth]
            if not nodes:
                continue
            SB = np.stack([np.frombuffer(nd[2], np.float32).reshape(self.K, self.D) for nd in nodes])
            fcs = [nd[4] for nd in nodes]                                       # accumulated feasibility cost per node
            ow, ee_, xx_ = self.candidates_batch(SB, np.broadcast_to(G[None], SB.shape), known)
            bad = None
            if fb_S is not None:                                                 # (node, entity) pairs that failed in the loop
                same = ((np.linalg.norm(SB[:, None, :, :2] - fb_S[None, :, :, :2], axis=-1) <= self.sc.tol_pos)
                        & ((np.abs(SB[:, None, :, 2:-1] - fb_S[None, :, :, 2:-1]).max(-1) <= self.pred_tol) | self.app_derived)
                        & ((SB[:, None, :, -1] > 0.5) == (fb_S[None, :, :, -1] > 0.5)))
                hit = (same | ~self.relevant[fb_e][None]).all(-1)                # (nodes, failures): the relevant entities only
                bad = np.zeros((len(SB), self.K), bool)
                for j_ in range(len(fb_e)):
                    bad[hit[:, j_], fb_e[j_]] = True
            for o_, e_, x_ in zip(ow, ee_, xx_):
                if bad is not None and bad[o_, int(e_)]:
                    continue                                                    # an event on e failed from this state
                ev = (int(e_), x_)
                src.append(SB[o_]); evs_all.append(ev); plans.append(nodes[o_][3] + [ev]); src_owner.append(o_)
            expanded += len(np.unique(ow))
            if not evs_all:
                continue
            succ, sup = self.step_batch(np.stack(src), evs_all, return_support=True)
            succ = succ.astype(np.float32)
            done = self.at_goal(succ, G[None], known)
            fc = [fcs[o_] for o_ in src_owner] if self.feas_weight > 0 else None
            if fc is not None:                                                   # feasibility cost above the per-entity margin
                fx = self.feas_weight * np.maximum(0.0, -np.log(np.maximum(sup, 1e-6)) - self.feas_margin[np.array([ev[0] for ev in evs_all])])
                fall = np.array(fc) + fx
            else:
                fall = np.zeros(len(plans))
            free = done & (fall <= 1e-9)
            if free.any():                                                       # a goal reached at no feasibility cost: the first
                j = int(np.nonzero(free)[0][0])                                  # one generated (greedy, as without the cost)
                return plans[j], {"expanded": expanded, "depth": len(plans[j])}
            hs = self.heuristic(succ, G, known)
            # FEASIBILITY COST: -log of the event support, so plans avoid events the support model finds implausible in their
            # context without forbidding them (closing a LOCKED drawer: support .00 with the 60k-step support model, unlocked
            # .69; its 2% quantile threshold fell to ~0 and stopped abstaining)
            for n_, (sn, pl, hv) in enumerate(zip(succ, plans, hs)):
                k = self.key(sn)
                f_ = float(fall[n_])
                gc = lam * len(pl) + f_                                          # path cost; a state is kept at its CHEAPEST path
                if k in seen and seen[k] <= gc + 1e-9:                           # (by length alone, a locked-drawer close reached
                    continue                                                     # 'closed' first and pruned the unlock path to it)
                seen[k] = gc
                g_ = bool(done[n_])
                heapq.heappush(openl, (gc + (0.0 if g_ else float(hv)), len(pl), sn.tobytes(), pl, f_, g_))
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
    rel = None
    if a.wm_arch in ("rel", "rel2", "pair", "pairabs", "pairg"):                 # data-derived layout constants (make_wm doc)
        P_ = np.concatenate([tr["before"], tr["after"]])[..., :2]
        Kn = np.concatenate([tr["before_known"], tr["after_known"]]).astype(bool) if "before_known" in tr else np.ones(P_.shape[:2], bool)
        med = np.array([np.median(P_[Kn[:, k], k], 0) if Kn[:, k].any() else np.zeros(2) for k in range(K)])
        spread = np.array([np.percentile(np.linalg.norm(P_[Kn[:, k], k] - med[k], axis=-1), 95) if Kn[:, k].any() else np.inf for k in range(K)])
        is_place = spread <= tol_pos                                               # an entity that never moves
        pm = med[is_place]
        if len(pm) >= 2:
            dd = np.linalg.norm(pm[:, None] - pm[None], axis=-1); dd[np.arange(len(pm)), np.arange(len(pm))] = np.inf
            h_sp = float(np.median(dd.min(1)))                                     # median nearest-neighbour place spacing
        else:
            h_sp = float(thr_pos)                                                  # no layout: one object width
        rel = {"h_sp": h_sp, "is_place": is_place.tolist()}
    rel_cover = []
    if a.hidden_protos and rel is not None:
        proto, rel_cover = hidden_goal_support(tr, proto, np.asarray(rel["is_place"], bool), tol_pos, a.protos, rng)
        print({"hidden_goal_support": {"protos_per_entity": [len(p) for p in proto],
                                       "cover_relations": [(m, j, np.round(o, 2).tolist(), n) for m, j, o, n in rel_cover]}}, flush=True)
        print({"wm_arch": a.wm_arch, "h_sp": round(h_sp, 2), "places": int(is_place.sum()), "K": int(K)}, flush=True)
    wm = make_wm(K, D, arch=a.wm_arch, rel=rel).to(dev)
    tr["target_in"] = event_input(tr["before"], tr["e"], tr["target"], tol_pos) if a.event_pos_only else tr["target"]
    va["target_in"] = event_input(va["before"], va["e"], va["target"], tol_pos) if a.event_pos_only else va["target"]
    T = {k: torch.as_tensor(tr[k], device=dev).float() for k in ("before", "after", "target_in", "after_known")}
    # covered-bit TRANSITIONS (becoming hidden / appearing) are 1-3% of the scored elements (cube 859 / 18582, scene 1328 /
    # 63449): with --cov-balance they weigh N_still / N_flip (at most 50), so hiding is not drowned (unweighted: stackings
    # predicted to hide the lower cube in 21% of VAL)
    flip = (tr["before"][..., -1] > 0.5) != (tr["after"][..., -1] > 0.5)
    scored = tr["after_known"].astype(bool) | (tr["after"][..., -1] > 0.5)
    ratio = float(np.clip((scored & ~flip).sum() / max(int((scored & flip).sum()), 1), 1.0, 50.0) ** a.cov_balance_power) if a.cov_balance else 1.0
    T["cov_w"] = torch.as_tensor(np.where(flip, ratio, 1.0), device=dev).float()
    print({"cov_balance": a.cov_balance, "flip_weight": round(ratio, 2)}, flush=True)
    E = torch.as_tensor(tr["e"], device=dev).long()
    CH = torch.as_tensor(((np.linalg.norm(tr["after"][..., :2] - tr["before"][..., :2], axis=-1) > tol_pos)
                          | (np.abs(tr["after"][..., 2:-1] - tr["before"][..., 2:-1]).max(-1) > app_unit_id)).astype(np.float32), device=dev)
    unit = torch.as_tensor(np.c_[np.full((K, 2), tol_pos / 32.0), np.repeat(2 * app_unit_id[:, None], A, 1)], device=dev).float()   # (K, D - 1)
    opt = torch.optim.AdamW(wm.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 1000) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.wm_steps))))
    for step in range(a.wm_steps):
        i = torch.randint(0, len(E), (a.wm_batch,), device=dev)
        s, x, y = sc.norm(T["before"][i]), sc.norm(T["target_in"][i]), sc.norm(T["after"][i])
        mk = T["after_known"][i]
        # the covered bit of an entity hidden after the event is an observation too (it vanished with no agent near: the
        # covered rule of events_objects.py); masking it with after_known meant no hiding was ever learned (2026-10-11: the WM
        # predicted the cube hidden in 0 of 205 TRAIN drawer-closing events, cube-triple stacking 0 of 43 VAL)
        mk_cov = mk if a.cov_known_only else torch.clamp(mk + (T["after"][i][..., D - 1] > 0.5).float(), max=1.0)
        if a.wm_arch == "pairg":                                                 # + the sparse change gate
            cont, logit, glogit = wm(s, E[i], x, logits=True, gate=True)
            gate_loss = (F.binary_cross_entropy_with_logits(glogit, CH[i], reduction="none") * mk).sum() / mk.sum().clamp(min=1)
        else:
            cont, logit = wm(s, E[i], x, logits=True)
            gate_loss = 0.0
        loss = ((((cont - y[..., :D - 1]) / unit) ** 2).mean(-1) * mk).sum() / mk.sum().clamp(min=1)             + (F.binary_cross_entropy_with_logits(logit, y[..., D - 1], reduction="none") * mk_cov * T["cov_w"][i]).sum() / (mk_cov * T["cov_w"][i]).sum().clamp(min=1) + gate_loss
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
                "cover_rule": cover_rule, "max_candidates": int(a.max_candidates), "wm_arch": a.wm_arch, "wm_rel": rel,
                "rel_cover": [(int(m), int(j), np.asarray(o, np.float32), int(n)) for m, j, o, n in rel_cover]}
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
    ap.add_argument("--wm-arch", choices=("entity", "rel", "rel2", "pair", "pairabs", "pairg"), default="entity",
                    help="entity: transformer over all entities with identities; rel: structured relative-offset effect model (make_wm doc)")
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
    ap.add_argument("--no-hidden-protos", dest="hidden_protos", action="store_false",
                    help="ablation: no placement / pre-hiding prototypes and no cover relations (hidden_goal_support; before 2026-10-11)")
    ap.add_argument("--no-cov-balance", dest="cov_balance", action="store_false",
                    help="ablation: covered-bit transitions unweighted in the WM loss (default: weighted (N_still / N_flip, at most 50) "
                         "** --cov-balance-power; power .5 had the best VAL hiding F1 of 0 / .5 / 1 on cube-triple .31 / .40 / .26 and "
                         "scene - / .46 / .13, 2026-10-11)")
    ap.add_argument("--cov-balance-power", type=float, default=0.5)
    ap.add_argument("--cov-known-only", action="store_true",
                    help="ablation: the covered-bit loss only where the after state is known (before 2026-10-11: hiding never learned)")
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
