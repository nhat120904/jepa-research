#!/usr/bin/env python3
"""Interaction events and a compact discrete state from scene memory v2, in the format of the event-WM pipeline
(build_events.py): events_{split}.npz with codes (state bits per frame), t_start, t, e, seg_start, before, after,
episode; vocab.npy (one row per event type) and tmaps.npy (8 x 8 target map per type, for skill v3).

Label-free and identical for every family (thresholds are 2-means splits fitted on TRAIN):
  memory       fresh = SeeThrough codes (learned space) taken when p > .5 for k frames; confirmed = the same, only outside
               the dilated agent mask; confirmed changes backdated to first sight in the fresh memory (as sm2_diag 'conf')
  dynamic      tokens that hold more than one state on TRAIN: among tokens with a confirmed change, 2-means split of the
               log time share of their second most frequent confirmed code (a change count alone misses tokens the
               arm often covers: few confirmed changes, yet two states half of the time each)
  values       per dynamic token, the codes held for a share of TRAIN time above the 2-means split of log shares pooled over
               dynamic tokens (at least the most frequent one); other codes map to the nearest kept value in FSQ space
  bits         per dynamic token: 1 bit for 2 values, otherwise one-hot over its values
  interactions temporal clusters of backdated confirmed changes on dynamic tokens; gap = 2-means split of log gaps > 5
               between consecutive onsets on TRAIN. before = confirmed state at the first onset - 1, after = confirmed state
               at the last confirmation (the agent has left); unseen tokens take their first confirmed value (offline)
  objects      dynamic tokens grouped by co-change over TRAIN interactions (Jaccard, average linkage, cut at the 2-means
               split of the pairwise similarities, as g_entities.py)
  type e       --types pattern (default): the set of changed objects; valid as an event identity only where effects do
               not depend on the state (Lights Out; build_events.py's XOR types rest on the same property).
               --types acted: the changed object with the largest AgentNet action sensitivity |F(x, a + d) - F(x, a - d)|
               over the 6 frames up to the first onset (puzzle VAL: .73 of single presses vs .26 chance; a contact cue
               for families with state-dependent effects). Vocabulary = types whose TRAIN count is above the 2-means
               split of log counts (rare -> e = -1, as build_events.py)
PRIVILEGED (scoring only, report.json): reference segments, and for puzzle the pressed button (deepest button joint)
against the acted object through a majority map learned on TRAIN.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from sm2_model import AgentNet, SeeThrough, agent_masks, memory_rule, see_codes, two_means_threshold
from sm2_train import load_split, to_t
from sm_model import FSQ

GRID = 16


def split_or(x, fallback):
    if len(x) < 10:
        return fallback, 0.0
    thr, sep, _ = two_means_threshold(np.asarray(x, np.float64))
    return (thr if sep > 2.0 else fallback), sep


def see_cache(a, split, n_ep, dev):
    """SeeThrough codes (uint16) and probabilities (uint8, p x 255) for every frame of the split, cached in --out."""
    fc, fp = a.out / f"see_codes_{split}.npy", a.out / f"see_prob_{split}.npy"
    if fc.exists() and fp.exists():
        return np.load(fc, mmap_mode="r"), np.load(fp, mmap_mode="r")
    sk = torch.load(a.run / "seethru_learned.pt", map_location=dev, weights_only=False)
    m = SeeThrough(sk["nd"], sk["nl"], sk["width"]).to(dev).eval(); m.load_state_dict(sk["model"])
    obs = load_split(a.cache, split, n_ep)[0]
    C = np.lib.format.open_memmap(fc, "w+", np.uint16, (len(obs), 256))
    P = np.lib.format.open_memmap(fp, "w+", np.uint8, (len(obs), 256))
    for s in range(0, len(obs), 1024):
        c, p = see_codes(m, to_t(obs[s:s + 1024], dev))
        C[s:s + len(c)] = c; P[s:s + len(c)] = np.round(p * 255)
    C.flush(); P.flush()
    return np.load(fc, mmap_mode="r"), np.load(fp, mmap_mode="r")


def episode_memories(codes, prob, idx, bits, segp, is_start, k, dilate):
    """-> fresh memory, confirmed memory (T, 256) and confirmed changes as rows (t_on, token, t_conf), local times."""
    codes = np.asarray(codes, np.int64)
    low = np.asarray(prob) <= 127
    mf, _ = memory_rule(codes, low, k)
    agent = agent_masks(idx, bits, segp, is_start, r=dilate)
    mc, chc, cf = memory_rule(codes, low | agent, k, return_confirmed=True)
    rows = []
    for t in np.flatnonzero(chc.any(1)):
        for i in np.flatnonzero(chc[t]):
            old = np.flatnonzero(cf[:t, i])
            lo = old[-1] + 1 if len(old) else 0
            seen = np.flatnonzero(mf[lo:t + 1, i] == mc[t, i])
            rows.append((lo + seen[0] if len(seen) else t, i, t))
    return mf, mc, np.array(rows, np.int64).reshape(-1, 3)


def clusters(onset_rows, gap):
    """rows (t_on, token, t_conf) of one episode -> list of row-index arrays, clustered by onset time (gap)."""
    if not len(onset_rows):
        return []
    o = np.argsort(onset_rows[:, 0], kind="stable")
    t = onset_rows[o, 0]
    cut = np.flatnonzero(np.diff(t) > gap) + 1
    return np.split(o, cut)


def backfill(mem):
    """unknown (-1) entries take the token's first known value (offline; the token was not seen before)."""
    out = mem.copy()
    for i in range(mem.shape[1]):
        k = np.flatnonzero(mem[:, i] >= 0)
        if len(k):
            out[:k[0], i] = mem[k[0], i]
    return out


class Encoder:
    """dynamic tokens' codes -> state bits."""

    def __init__(self, dyn, values, fsq_vals):
        self.dyn, self.values = dyn, values                                     # values: list of kept codes per dyn token
        self.fsq = fsq_vals                                                      # (15625, 6)
        self.width = [1 if len(v) == 2 else len(v) for v in values]
        self.bits = int(sum(self.width))

    def value_index(self, codes):
        """codes (T, n_dyn) -> index of the nearest kept value per dynamic token (T, n_dyn); unknown (-1) -> 0."""
        out = np.zeros(codes.shape, np.int64)
        for j, v in enumerate(self.values):
            c = codes[:, j]
            d = ((self.fsq[np.maximum(c, 0)][:, None, :] - self.fsq[np.asarray(v)][None]) ** 2).sum(-1)
            out[:, j] = np.where(c >= 0, d.argmin(1), 0)
        return out

    def encode(self, codes):
        vi = self.value_index(codes)
        cols = []
        for j, w in enumerate(self.width):
            cols.append(vi[:, j:j + 1] == 1 if w == 1 else np.eye(w, dtype=bool)[vi[:, j]])
        return np.concatenate(cols, 1).astype(np.uint8)


def objects_by_cochange(cl_sets, dyn):
    """cl_sets: list of sets of dynamic-token positions (indices into dyn) changed per TRAIN interaction."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = len(dyn)
    inc = np.zeros((len(cl_sets), n), bool)
    for r, s in enumerate(cl_sets):
        inc[r, list(s)] = True
    co = inc.T.astype(np.float64) @ inc.astype(np.float64)
    cnt = np.diag(co)
    jac = co / np.maximum(cnt[:, None] + cnt[None] - co, 1)
    np.fill_diagonal(jac, 1.0)
    sims = jac[np.triu_indices(n, 1)]
    cut, sep = split_or(sims[sims > 0], 0.5)
    lab = fcluster(linkage(squareform(1 - jac, checks=False), "average"), t=1 - cut, criterion="distance") - 1
    return lab, {"pairwise_split": float(cut), "ashman_D": float(sep), "objects": int(lab.max() + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="sm2_train.py --out directory")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--family", choices=("cube", "puzzle", "scene"), required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--types", choices=("pattern", "acted"), default="pattern")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    dev = a.device
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    fsq_vals = FSQ().values(torch.arange(5 ** 6)).numpy()
    rep = {"run": str(a.run), "family": a.family, "k": a.k, "dilate": a.dilate}

    def split_data(split, n_ep):
        obs, act, st, en, is_start, is_end = load_split(a.cache, split, n_ep)
        C, P = see_cache(a, split, n_ep, dev)
        bits = np.load(a.run / f"teacher_{split}.npy", mmap_mode="r"); segp = np.load(a.run / f"seg_{split}.npy", mmap_mode="r")
        return st, en, is_start, C, P, bits, segp

    # ---- pass 1 on TRAIN: change counts, code shares, onset gaps
    st, en, is_start, C, P, tb, ts = split_data("train", a.episodes)
    nchg = np.zeros(256, np.int64)
    share = np.zeros((256, 5 ** 6), np.int64)
    ep_rows, gaps = [], []
    for s0, e0 in zip(st, en):
        idx = np.arange(s0, e0 + 1)
        mf, mc, rows = episode_memories(C[s0:e0 + 1], P[s0:e0 + 1], idx, tb, ts, is_start, a.k, a.dilate)
        np.add.at(nchg, rows[:, 1], 1)
        known = mc >= 0
        np.add.at(share, (np.nonzero(known)[1], mc[known]), 1)
        ep_rows.append(rows)
        if len(rows):
            g = np.diff(np.unique(rows[:, 0]))
            gaps.append(g[g > 5])
    tot = share.sum(1)
    second = np.sort(share, 1)[:, -2] / np.maximum(tot, 1)
    cand = np.flatnonzero((nchg > 0) & (second > 0))
    dthr, dsep = split_or(np.log10(second[cand]), np.log10(0.05))
    dyn = cand[np.log10(second[cand]) > dthr]
    gaps = np.concatenate(gaps)
    gthr, gsep = split_or(np.log10(gaps), np.log10(15))
    gap = float(10 ** gthr)
    shares = share[dyn] / np.maximum(share[dyn].sum(1, keepdims=True), 1)
    pooled = np.log10(shares[shares > 0])
    vthr, vsep = split_or(pooled, np.log10(0.05))
    values = []
    for j in range(len(dyn)):
        keep = np.flatnonzero(np.log10(np.maximum(shares[j], 1e-12)) > vthr)
        if not len(keep):
            keep = np.array([shares[j].argmax()])
        values.append(keep[np.argsort(-shares[j, keep])].tolist())
    enc = Encoder(dyn, values, fsq_vals)
    rep["dynamic"] = {"tokens": int(len(dyn)), "second_share_split": float(10 ** dthr), "ashman_D": float(dsep),
                      "values_per_token": np.bincount([len(v) for v in values]).tolist(), "value_share_split": float(10 ** vthr),
                      "value_ashman_D": float(vsep), "state_bits": enc.bits}
    rep["interaction_gap"] = {"frames": gap, "ashman_D": float(gsep), "gaps": int(len(gaps))}
    print(json.dumps(rep), flush=True)

    # ---- objects by co-change over TRAIN interactions
    pos = {int(t): j for j, t in enumerate(dyn)}
    cl_sets = []
    for rows in ep_rows:
        r = rows[np.isin(rows[:, 1], dyn)]
        for c in clusters(r, gap):
            cl_sets.append({pos[int(i)] for i in r[c, 1]})
    lab, orep = objects_by_cochange(cl_sets, dyn)
    rep["objects"] = orep
    obj_tokens = [dyn[lab == o] for o in range(lab.max() + 1)]
    print("objects", orep, "sizes", np.bincount(lab).tolist(), flush=True)

    # ---- pass 2: state bits and interactions per split
    net = None
    if a.types == "acted":
        ck = torch.load(a.run / "agent_net.pt", map_location=dev, weights_only=False)
        net = AgentNet(ck["act_dim"], ck["width"]).to(dev).eval(); net.load_state_dict(ck["net"])

    @torch.no_grad()
    def sensitivity(obs, act, is_start, t_idx, d=0.5):
        x = to_t(obs[t_idx], dev)
        at = torch.as_tensor(act[t_idx], device=dev)
        ap = torch.as_tensor(act[np.maximum(t_idx - 1, 0)], device=dev); ap[torch.as_tensor(is_start[t_idx], device=dev)] = 0
        one = torch.ones(len(t_idx), device=dev)
        S = torch.zeros(len(t_idx), 1, 64, 64, device=dev)
        for j in range(at.shape[1]):
            e = torch.zeros_like(at); e[:, j] = d
            S += (net(x, at + e, ap, one) - net(x, at - e, ap, one)).abs().mean(1, keepdim=True)
        return torch.nn.functional.avg_pool2d(S, 4).flatten(1).sum(0).cpu().numpy()

    out = {}
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        obs, act, st, en, is_start, is_end = load_split(a.cache, split, n_ep)
        _, _, _, C, P, tb, ts = split_data(split, n_ep)
        N = int(en[-1] + 1)
        codes_bits = np.zeros((N, enc.bits), np.uint8)
        ev = {k: [] for k in ("t_start", "t", "t_end_onset", "acted", "seg_start", "episode", "n_changed", "n_objects")}
        pats, before, after = [], [], []
        for epi, (s0, e0) in enumerate(zip(st, en)):
            idx = np.arange(s0, e0 + 1)
            mf, mc, rows = episode_memories(C[s0:e0 + 1], P[s0:e0 + 1], idx, tb, ts, is_start, a.k, a.dilate)
            codes_bits[s0:e0 + 1] = enc.encode(backfill(mf)[:, dyn])
            cb = enc.encode(backfill(mc)[:, dyn])
            r = rows[np.isin(rows[:, 1], dyn)]
            seg = 0
            for c in clusters(r, gap):
                t_on, t_end, t_done = r[c, 0].min(), r[c, 0].max(), r[c, 2].max()
                changed = np.zeros(256, bool); changed[r[c, 1]] = True
                objs = [o for o, tk in enumerate(obj_tokens) if changed[tk].any()]
                acted = -1
                if net is not None and objs:
                    S = sensitivity(obs, act, is_start, s0 + np.arange(max(t_on - 6, 0), t_on + 1))
                    acted = objs[int(np.argmax([S[obj_tokens[o]].mean() for o in objs]))]
                pats.append(tuple(objs))
                ev["t_start"].append(s0 + t_on); ev["t"].append(s0 + t_done); ev["t_end_onset"].append(s0 + t_end)
                ev["acted"].append(acted); ev["seg_start"].append(s0 + seg); ev["episode"].append(epi)
                ev["n_changed"].append(int(changed.sum())); ev["n_objects"].append(len(objs))
                before.append(cb[max(t_on - 1, 0)]); after.append(cb[min(t_done, len(idx) - 1)])
                seg = min(t_done + 1, len(idx) - 1)
        out[split] = (codes_bits, {k: np.array(v, np.int64) for k, v in ev.items()}, pats, np.array(before, np.uint8).reshape(-1, enc.bits),
                      np.array(after, np.uint8).reshape(-1, enc.bits))
    # vocabulary on TRAIN: types (patterns or acted objects) whose count is above the 2-means split of log counts
    keys_tr = out["train"][2] if a.types == "pattern" else [(o,) for o in out["train"][1]["acted"]]
    uniq, cnt = np.unique(np.array([",".join(map(str, k)) for k in keys_tr]), return_counts=True)
    lc = np.log10(cnt)
    cthr, csep = split_or(lc, np.log10(10))
    keep = [u for u, c in zip(uniq, lc) if c > cthr and u not in ("", "-1")]
    to_e = {u: i for i, u in enumerate(keep)}
    tmaps = np.zeros((len(keep), 64), np.float32)
    for e, u in enumerate(keep):
        m = np.zeros(256, bool)
        for o in map(int, u.split(",")):
            m[obj_tokens[o]] = True
        tmaps[e] = m.reshape(8, 2, 8, 2).any((1, 3)).ravel()
    rep["vocab"] = {"types": a.types, "size": int(len(keep)), "count_split": float(10 ** cthr), "ashman_D": float(csep),
                    "train_counts": sorted(cnt[lc > cthr].tolist(), reverse=True), "dropped_events": int(cnt[lc <= cthr].sum())}
    np.save(a.out / "vocab.npy", np.arange(len(keep)))
    np.save(a.out / "tmaps.npy", tmaps)
    np.save(a.out / "dyn_tokens.npy", dyn)
    (a.out / "encoder.json").write_text(json.dumps({"dyn": dyn.tolist(), "values": values, "objects": [t.tolist() for t in obj_tokens],
                                                     "types": a.types, "vocab": keep, "gap": gap, "k": a.k, "dilate": a.dilate}) + "\n")
    for split, (cbits, ev, pats, b, af) in out.items():
        keys = pats if a.types == "pattern" else [(o,) for o in ev["acted"]]
        e = np.array([to_e.get(",".join(map(str, k)), -1) for k in keys], np.int64)
        np.savez_compressed(a.out / f"events_{split}.npz", codes=cbits, t_start=ev["t_start"], t=ev["t"], t_end_onset=ev["t_end_onset"],
                            e=e, acted_object=ev["acted"], seg_start=ev["seg_start"], before=b, after=af, episode=ev["episode"],
                            n_changed=ev["n_changed"], n_objects=ev["n_objects"])
        rep[f"events_{split}"] = {"events": int(len(e)), "per_episode": float(len(e) / (ev["episode"].max() + 1 if len(e) else 1)),
                                  "in_vocab": float((e >= 0).mean()) if len(e) else None,
                                  "changed": float(np.mean([(x != y).any() for x, y in zip(b, af)])) if len(b) else None}
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
