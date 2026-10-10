#!/usr/bin/env python3
"""Component 8 (method/README.md): events from the entity table of memory_entities.py, one rule set for every family.

Port of docs/generic_state_20261007/base_source/u_events.py, --per-frame mode (see SOURCES in method/README.md); the
rules are the same:
  1. rest runs per entity: >= m observations whose appearance stays within r_app of the run mean; an observation needs
     the entity readable and free of the agent; unobserved frames do not break a run (u_events.rest_runs);
  2. consecutive rests are one rest unless they differ by more than thr_app; otherwise the gap between them is a change,
     timed per frame: departure = last readable frame still at the before-rest state, arrival = first readable frame from
     which the reading stays at the after-rest state (u_events.transition);
  3. grouping (--group): `overlap` (u_events) = changes whose [departure, arrival] windows overlap (slack --frame-gap) form
     one event; `span` (default, see below) = an event takes every change arriving within G frames of its first arrival.
     Entities whose reading at the two frames around the event differs are added (u_events: changes missed by the rest
     runs); the ACTED entity is the changing entity closest to the agent at the event's first frame, then the most
     central one, then the longest change;
  4. entity state = (u, v, app (A values), covered); covered = never readable during a static interval (no change in
     progress) (u_events.state_at). STATE_RULE: the value of an entity at frame t is the rest value whose change began at or
     before t (rest values switch only at the change windows), a readable reading at t taking precedence (as u_events:
     SeeThrough readings through the transparent arm decode puzzle lights 100% right and cover 73% of light-frames, the
     agent-free ones only 36%); before = value at the event's first frame - 1, after = value at its last frame + 1. u_events read the
     nearest rest on the requested side instead, which for an entity hidden by the agent returns a rest reached only
     after a LATER interaction (puzzle VAL check 2026-10-08, PRIVILEGED per-light decoders: readings of observed tokens
     are 100% right, but labels from the nearest rest give all 20 lights right in 34% (before) / 42% (after) of events and
     the toggled set right in 21%).
Differences from u_events, all forced by fixed-place token entities:
  - positions are constant, so rests and changes are defined by appearance only;
  - appearance thresholds: the token codes are quantized and stable (>= 99% of consecutive agent-free readings are
    identical, local check 2026-10-08), so the u_events split of NON-ZERO frame-to-frame differences separates 1-step from
    larger real code changes instead of noise from change. With quantized readings (share of exact zeros >= .99) the noise
    radius is a quarter of the quantization step q: r_app = q / 4, thr_app = 2 r_app (any code change); otherwise the
    u_events rule (2-means split of log non-zero differences, thr_app = 2 r_app);
  - rest runs vectorized over entities (same cur / cand logic); per-frame rest labels are not materialized (K up to ~150 x
    1M frames); rest runs are saved instead (rests_{split}.npz, for prototypes);
  - (as u_events) the transition reads every frame; rest observations need the entity readable (SeeThrough p > .5) and free
    of the agent. A readability test inside the transition (tried 2026-10-08) doubled nothing but the window length: on
    puzzle VAL 48% of the frames between two rests are not readable, yet their argmax code is the old or the new rest code
    77% of the time; windows p50 / p90 16 / 29 frames with the test, 1 / 13 without, and long windows chain consecutive
    presses into one event;
  - agent distance at token resolution (undilated segmenter tokens);
  - grouping by span (complete linkage on arrival times) instead of overlap chains: around the agent the token memory
    records many changes (the arm and a held cube over a token give readings that match neither rest), so overlapping
    windows chain several interactions into one event (cube VAL: events of 80-125 frames with 20-30 changed tokens around
    two true moves). G = the 2-means split of log gaps between consecutive change arrivals in TRAIN episodes, the same
    rule for every family (`--span` fixes it). Local comparison 2026-10-08 (VAL, PRIVILEGED sm2_diag reference, w15 span
    recall / precision, overlap -> span 16): puzzle .76 / .99 -> .94 / 1.0, scene .64 / .84 -> .91 / .92, cube .46 / .87 ->
    .97 / .91; events per episode puzzle 27.5 -> 32.7 (29.6 presses), scene 20.8 -> 29.8 (18.7), cube 10.4 -> 33.0 (11.8).
Output per split: events_{split}.npz (t_start, t, e, before (N, K, D), after, target = after[e], seg_start, knock = several
entities moved (never, for fixed places), episode, n_changed, thr_pos, tol_pos, thr_app), rests_{split}.npz (entity,
start, end, mean app), report.json (thresholds + PRIVILEGED diagnostics against the simulator, scoring only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from utils import bimodal_split, episode_bounds, save_json, two_means_threshold


def load(d: Path, split: str):
    z = dict(np.load(d / f"entities_{split}.npz"))
    z["app"] = np.load(d / f"app_{split}.npy", mmap_mode="r")
    return z


def rest_runs_vec(app, ok, s0, r_app, m):
    """Rest runs of every entity in one episode: app (T, K, A) float, ok (T, K) bool (local frames) -> list per entity of
    [start, end, sum_app, n] in global frames. u_events.rest_runs (constant positions), vectorized over entities."""
    T, K, A = app.shape
    runs = [[] for _ in range(K)]
    cv = np.zeros(K, bool); cs = np.zeros(K, np.int64); ce = np.zeros(K, np.int64); csum = np.zeros((K, A)); cn = np.zeros(K)
    dv = np.zeros(K, bool); ds = np.zeros(K, np.int64); de = np.zeros(K, np.int64); dsum = np.zeros((K, A)); dn = np.zeros(K)
    for t in range(T):
        vis = ok[t]
        if not vis.any():
            continue
        c = app[t]
        near_c = vis & cv & (np.abs(c - csum / np.maximum(cn, 1)[:, None]).max(1) <= r_app)
        if near_c.any():
            ce[near_c] = t; csum[near_c] += c[near_c]; cn[near_c] += 1; dv[near_c] = False
        rest = vis & ~near_c
        near_d = rest & dv & (np.abs(c - dsum / np.maximum(dn, 1)[:, None]).max(1) <= r_app)
        if near_d.any():
            de[near_d] = t; dsum[near_d] += c[near_d]; dn[near_d] += 1
            prom = near_d & (dn >= m)
            for k in np.flatnonzero(prom & cv):
                runs[k].append([s0 + cs[k], s0 + ce[k], csum[k].copy(), cn[k]])
            cv[prom] = True; cs[prom] = ds[prom]; ce[prom] = de[prom]; csum[prom] = dsum[prom]; cn[prom] = dn[prom]; dv[prom] = False
        new = rest & ~near_d
        if new.any():
            dv[new] = True; ds[new] = t; de[new] = t; dsum[new] = c[new]; dn[new] = 1
    for k in np.flatnonzero(cv):
        runs[k].append([s0 + cs[k], s0 + ce[k], csum[k].copy(), cn[k]])
    return runs


def transition(app_k, r0, r1, r_app):
    """u_events.transition: app_k (T, A) and the runs r0, r1 in episode-local frames -> (departure + 1, arrival), local."""
    s = np.arange(r0[1], r1[0] + 1)

    def at(r):
        return np.abs(app_k[s] - r[2] / r[3]).max(-1) <= r_app

    stay = np.flip(np.cumprod(np.flip(at(r1)))).astype(bool)
    arr = int(s[stay.argmax()]) if stay.any() else int(r1[0])
    pre = s[(s < arr) & at(r0)]
    dep = int(pre.max()) if len(pre) else int(r0[1])
    return dep + 1, arr


def episode_changes(z, s0, e0, ok, scale, r_app, thr_app, m):
    """One episode -> (app (T, K, A) float, merged rest runs per entity, changes (t0, t1, k, before-rest end) sorted), global
    frames: rest runs, merging of rests within thr_app, per-frame transition timing (u_events)."""
    app_ep = np.asarray(z["app"][s0:e0 + 1], np.float32) * scale
    K = app_ep.shape[1]
    runs = rest_runs_vec(app_ep, ok[s0:e0 + 1], s0, r_app, m)
    merged, changes = [], []
    for k in range(K):
        mk = []
        for r_ in runs[k]:
            if mk and np.abs(r_[2] / r_[3] - mk[-1][2] / mk[-1][3]).max() <= thr_app:
                mk[-1] = [mk[-1][0], r_[1], mk[-1][2] + r_[2], mk[-1][3] + r_[3]]
                continue
            mk.append(list(r_))
        merged.append(mk)
        for i in range(1, len(mk)):
            r0 = [mk[i - 1][0] - s0, mk[i - 1][1] - s0, mk[i - 1][2], mk[i - 1][3]]
            r1 = [mk[i][0] - s0, mk[i][1] - s0, mk[i][2], mk[i][3]]
            t0, t1 = transition(app_ep[:, k], r0, r1, r_app)
            changes.append((s0 + t0, s0 + t1, k, mk[i - 1][1]))
    changes.sort()
    return app_ep, merged, changes


def group_changes(changes, mode, frame_gap, span):
    """changes (t0, t1, k, .) of one episode sorted by t0 -> list of groups. overlap: u_events chains of overlapping windows
    (slack frame_gap); span: complete linkage on arrival times t1 (an event takes every change arriving within `span`
    frames of its first arrival)."""
    groups = []
    if mode == "overlap":
        i = 0
        while i < len(changes):
            grp = [changes[i]]; t1 = changes[i][1]; j = i + 1
            while j < len(changes) and changes[j][0] <= t1 + frame_gap:
                grp.append(changes[j]); t1 = max(t1, changes[j][1]); j += 1
            i = j
            groups.append(grp)
        return groups
    g = []
    for c in sorted(changes, key=lambda c: (c[1], c[0])):
        if g and c[1] - g[0][1] > span:
            groups.append(g); g = []
        g.append(c)
    if g:
        groups.append(g)
    return groups


def app_thresholds(z, starts, ends, scale, rng, pairs=200_000):
    """r_app, thr_app from consecutive readable, agent-free frame pairs of the same episode (see module doc)."""
    n = int(z["n"])
    ok = z["area"].astype(bool) & z["free"]
    last = np.zeros(n, bool); last[ends] = True
    f0 = np.flatnonzero(~last[:-1])
    f0 = f0[rng.permutation(len(f0))[:pairs]]
    f0.sort()
    a0 = np.asarray(z["app"][f0], np.float32) * scale; a1 = np.asarray(z["app"][f0 + 1], np.float32) * scale
    both = ok[f0] & ok[f0 + 1]
    d = np.abs(a1 - a0).max(-1)[both]
    zero = float((d == 0).mean()) if len(d) else 1.0
    nz = d[d > 0]
    if zero >= 0.99 and len(nz):
        q = float(nz.min())
        return q / 4, q / 2, {"rule": "quantized", "zero_share": zero, "step": q}
    if len(nz) >= 10:
        r_log, D1 = bimodal_split(np.log(nz + 1e-6))
        r_app = float(np.exp(r_log)) if np.isfinite(r_log) else float(np.percentile(d, 99))
        return r_app, 2 * r_app, {"rule": "u_events", "zero_share": zero, "ashman_D": D1}
    return np.inf, np.inf, {"rule": "constant", "zero_share": zero}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=Path, required=True, help="memory_entities.py output dir")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--frame-gap", type=int, default=2, help="--group overlap: slack in frames between overlapping windows")
    ap.add_argument("--group", choices=("span", "overlap"), default="span")
    ap.add_argument("--rest-obs", choices=("free", "readable"), default="free",
                    help="rest observations: readable and free of the agent (u_events contact-free rule), or readable only")
    ap.add_argument("--span", type=float, default=16.0,
                    help="--group span: G in frames, the same for every family (0 = 2-means split of TRAIN arrival gaps). DEV CHOICE "
                         "2026-10-08: the 2-means split gives 4.4-4.7 frames (it separates the 1-4 frame staggering of one object's tokens "
                         "from everything else; puzzle 62 events per episode for 29.6 presses) and a label-free repeatability criterion "
                         "prefers ever smaller events, so G = 16 was chosen on VAL with PRIVILEGED metrics (span 4..48 swept on all "
                         "three families); report it as such")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    tr = load(a.entities, "train")
    scale = float(tr["app_scale"]); pos = tr["pos"].astype(np.float64); K = len(pos); A = tr["app"].shape[2]; D = A + 3
    thr_pos = float(tr["thr_pos"])
    st_tr, en_tr = episode_bounds(a.cache, "train", a.episodes)
    r_app, thr_app, arep = app_thresholds(tr, st_tr, en_tr, scale, rng)
    r_pos, tol_pos = 1.0, 2.0                                                  # constant positions (u_events fallback)
    rep = {"entities": K, "app_dims": A, "thr_pos": thr_pos, "r_pos": r_pos, "tol_pos": tol_pos, "r_app": r_app, "thr_app": thr_app,
           "app_threshold_rule": arep, "m": a.m, "frame_gap": a.frame_gap}
    print(json.dumps(rep), flush=True)
    tok_uv = np.stack([np.arange(256) % 16 * 4 + 1.5, np.arange(256) // 16 * 4 + 1.5], 1)
    if a.group == "span" and a.span <= 0:
        ok_tr = tr["area"].astype(bool) & (tr["free"] if a.rest_obs == "free" else True)
        gaps = []
        for s0, e0 in zip(st_tr, en_tr):
            ch = episode_changes(tr, s0, e0, ok_tr, scale, r_app, thr_app, a.m)[2]
            arr = np.unique([c[1] for c in ch])
            if len(arr) > 1:
                g = np.diff(arr); gaps.append(g[g >= 1])
        gaps = np.concatenate(gaps)
        th, Dg, _ = two_means_threshold(np.log(gaps))
        a.span = float(np.exp(th))
        rep["span_rule"] = {"gaps": int(len(gaps)), "split_frames": a.span, "ashman_D": Dg, "gap_p25_p50_p75": np.percentile(gaps, [25, 50, 75]).tolist()}
    rep.update(group=a.group, span=a.span, rest_obs=a.rest_obs)
    print(json.dumps({k: rep[k] for k in ("group", "span")} | ({"span_rule": rep["span_rule"]} if "span_rule" in rep else {})), flush=True)
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        z = tr if split == "train" else load(a.entities, split)
        starts, ends = episode_bounds(a.cache, split, n_ep)
        n = int(ends[-1] + 1)
        rd = z["area"].astype(bool)
        ok = rd & z["free"] if a.rest_obs == "free" else rd
        ev = {k: [] for k in ("t_start", "t", "e", "seg_start", "episode", "n_changed")}
        before, after = [], []
        rest_rows = []
        for epi, (s0, e0) in enumerate(zip(starts, ends)):
            T = e0 - s0 + 1
            app_ep, merged, changes = episode_changes(z, s0, e0, ok, scale, r_app, thr_app, a.m)
            for k in range(K):
                rest_rows += [(k, r_[0], r_[1], *(r_[2] / r_[3])) for r_ in merged[k]]
            # static intervals (no change in progress) of this episode -> covered per entity
            in_ch = np.zeros(T, bool)
            for t0, t1, *_ in changes:
                in_ch[t0 - s0:t1 - s0 + 1] = True
            iv = []
            t = 0
            while t < T:
                if in_ch[t]:
                    t += 1; continue
                u = t
                while u + 1 < T and not in_ch[u + 1]:
                    u += 1
                if u - t + 1 >= a.m:
                    iv.append((t, u, ~rd[s0 + t:s0 + u + 1].any(0)))
                t = u + 1
            ep_events = []
            for grp in group_changes(changes, a.group, a.frame_gap, a.span):
                t1 = max(g[1] for g in grp)
                c0 = min(g[0] for g in grp)
                b0, b1 = max(c0 - 1, s0), min(t1 + 1, e0)
                ks_ = {g[2] for g in grp}
                readable = rd[b0] & rd[b1]
                raw = np.abs(app_ep[b1 - s0] - app_ep[b0 - s0]).max(-1) > thr_app
                ks_ = sorted(ks_ | set(np.flatnonzero(raw & readable).tolist()))
                cen = {k: sum(np.hypot(*(pos[k] - pos[j_])) for j_ in ks_ if j_ != k) for k in ks_}
                am = np.unpackbits(z["agent"][c0])[:256].astype(bool)
                if am.any():
                    dist = {k: float(np.min(np.hypot(*(tok_uv[am] - pos[k]).T))) for k in ks_}
                else:
                    dist = {k: np.inf for k in ks_}
                dur = {k: max([g[1] - g[0] for g in grp if g[2] == k] or [0]) for k in ks_}
                acted = min(ks_, key=lambda k: (dist[k], cen[k], -dur[k]))
                ep_events.append((c0, t1, acted, len(ks_)))

            # value of entity k over time: its i-th rest value from the start of the change into it (the change window
            # [t0, t1], u_events timing) until the start of the change out of it; before the first rest, the first value
            vals = [np.array([r_[2] / r_[3] for r_ in mk]) for mk in merged]
            t0s = [[] for _ in range(K)]
            for c in changes:
                t0s[c[2]].append(c[0] - s0)
            t0s = [np.sort(np.array(x)) for x in t0s]

            def state_at(t, side):
                """state of every entity at local frame t: the value whose change started at or before t (STATE_RULE in the
                module doc); a readable reading at t takes precedence."""
                s = np.zeros((K, D), np.float32)
                s[:, :2] = pos
                for k in range(K):
                    if len(vals[k]):
                        s[k, 2:2 + A] = vals[k][min(int(np.searchsorted(t0s[k], t, side="right")), len(vals[k]) - 1)]
                    if rd[s0 + t, k]:
                        s[k, 2:2 + A] = app_ep[t, k]
                if iv:
                    mid = np.array([(x[0] + x[1]) / 2 for x in iv])
                    s[:, D - 1] = iv[int(np.argmin(np.abs(mid - t)))][2]
                return s

            prev_end = 0
            for c0, t1, acted, nch in ep_events:
                ev["t_start"].append(c0); ev["t"].append(t1); ev["e"].append(acted); ev["episode"].append(epi)
                ev["seg_start"].append(min(s0 + prev_end, c0)); ev["n_changed"].append(nch)
                before.append(state_at(max(c0 - 1 - s0, 0), -1)); after.append(state_at(min(t1 + 1 - s0, T - 1), 1))
                prev_end = t1 + 1 - s0
        E_ = {k: np.array(v, np.int64) for k, v in ev.items()}
        before = np.array(before, np.float32).reshape(-1, K, D); after = np.array(after, np.float32).reshape(-1, K, D)
        np.savez_compressed(a.out / f"events_{split}.npz", t_start=E_["t_start"], t=E_["t"], e=E_["e"], before=before, after=after,
                            target=after[np.arange(len(E_["e"])), E_["e"]], seg_start=E_["seg_start"], knock=np.zeros(len(E_["e"]), bool),
                            episode=E_["episode"], n_changed=E_["n_changed"], thr_pos=thr_pos, tol_pos=tol_pos, thr_app=thr_app)
        rr = np.array(rest_rows, np.float32).reshape(-1, 3 + A)
        np.savez_compressed(a.out / f"rests_{split}.npz", entity=rr[:, 0].astype(np.int64), start=rr[:, 1].astype(np.int64),
                            end=rr[:, 2].astype(np.int64), app=rr[:, 3:])
        r = {"events": int(len(E_["e"])), "per_episode": float(len(E_["e"]) / len(starts)), "episodes": int(len(starts)),
             "changed_entities_pct": np.percentile(E_["n_changed"], [50, 90, 99]).tolist() if len(E_["e"]) else None,
             "acted_entities_used": int(len(np.unique(E_["e"]))), "rest_runs": int(len(rr))}
        r.update(privileged(a.cache, split, starts, ends, n, E_, z["tokens"]))
        rep[split] = r
        print(split, json.dumps(r), flush=True)
    save_json(a.out / "report.json", rep)


def privileged(cache: Path, split: str, starts, ends, n, E_, tokens):
    """PRIVILEGED scoring only (simulator state never enters the method): event times vs object motion / button toggles,
    one-to-one matching (u_events), and for puzzle the acted token -> pressed button purity."""
    ts, te = E_["t_start"], E_["t"]
    refm = np.zeros(n - 1, bool)
    q = np.load(cache / f"{split}_qpos.npy", mmap_mode="r") if (cache / f"{split}_qpos.npy").exists() else None
    b = np.load(cache / f"{split}_button_states.npy", mmap_mode="r") if (cache / f"{split}_button_states.npy").exists() else None
    for s0, e0 in zip(starts, ends):
        if q is not None and "puzzle" not in cache.name:
            refm[s0:e0] |= np.abs(np.diff(np.asarray(q[s0:e0 + 1, 14:]), axis=0)).max(1) > 2e-3
        if b is not None:
            refm[s0:e0] |= (np.asarray(b[s0 + 1:e0 + 1]) != np.asarray(b[s0:e0])).any(1)
    d_ = np.diff(np.r_[0, refm.astype(np.int8), 0]); rs_, re_ = np.nonzero(d_ == 1)[0] + 1, np.nonzero(d_ == -1)[0]
    out = {}
    if not (len(rs_) and len(ts)):
        return out

    def one_to_one(R0, R1, slack=5):
        nref = np.array([((R0 <= y + slack) & (R1 >= x - slack)).sum() for x, y in zip(ts, te)])
        hit = [np.nonzero((ts <= y + slack) & (te >= x - slack))[0] for x, y in zip(R0, R1)]
        return {"refs_per_episode": round(len(R0) / len(starts), 2),
                "events_with_0_1_2plus_refs": [int((nref == 0).sum()), int((nref == 1).sum()), int((nref >= 2).sum())],
                "refs_in_clean_one_to_one": round(float(np.mean([len(h) == 1 and nref[h[0]] == 1 for h in hit])), 3)}

    out["privileged_recall"] = float(np.mean([np.any((ts <= y + 15) & (te >= x - 15)) for x, y in zip(rs_, re_)]))
    out["privileged_precision"] = float(np.mean([np.any((rs_ <= y + 15) & (re_ >= x - 15)) for x, y in zip(ts, te)]))
    out["privileged_one_to_one"] = one_to_one(rs_, re_)
    if "cube" in cache.name and q is not None:
        sl = [c for c in (14, 21, 28, 35) if c + 3 <= q.shape[1]]
        disp = np.array([max(np.linalg.norm(np.asarray(q[min(y + 1, n - 1), c:c + 2]) - np.asarray(q[x, c:c + 2])) for c in sl) for x, y in zip(rs_, re_)])
        big = disp >= 0.02
        if big.any():
            out["privileged_one_to_one_moves_2cm"] = one_to_one(rs_[big], re_[big])
    if "puzzle" in cache.name and q is not None:
        qq = np.asarray(q[:n, 14:34])
        pressed = np.where(qq.min(1) < -0.015, qq.argmin(1), -1)
        pb = []
        for x, y in zip(ts, te):
            w = pressed[max(x - 12, 0):y + 3]
            u, c = np.unique(w[w >= 0], return_counts=True)
            pb.append(int(u[c.argmax()]) if len(u) else -1)
        pb = np.array(pb); e = E_["e"]; half = E_["episode"] < E_["episode"].max() / 2
        mp = {}
        for k in np.unique(e[half & (pb >= 0)]):
            u, c = np.unique(pb[half & (pb >= 0) & (e == k)], return_counts=True); mp[k] = u[c.argmax()]
        tst = ~half & (pb >= 0)
        out["privileged_acted_token_to_pressed_button"] = float(np.mean([mp.get(k, -2) == p for k, p in zip(e[tst], pb[tst])])) if tst.any() else None
        out["privileged_events_without_press"] = float((pb < 0).mean())
    return out


if __name__ == "__main__":
    main()
