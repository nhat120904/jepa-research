#!/usr/bin/env python3
"""Unified backend, step 1: events from the generic entity table -- one rule set for every task family.

Input: entities_{split}.npz (per frame and identity: pos px, app RGB/255, visible area; agent mask) from
the front end (sam2_tracks.py). Per identity and episode:
  1. rest runs: >= m visible frames whose pos stays within r_pos px and app within r_app of the run mean
     (invisible frames -- occlusion, being covered -- do not break a run); a frame counts as a rest
     observation only while no agent pixel is within half an object width (objects rest free of the agent:
     a cube held still in the gripper is in transit);
  2. consecutive rests are one rest unless they differ by more than tol_pos in position or thr_app in
     appearance; otherwise the gap between them is a change of that identity;
       r_pos = noise / motion split (2-means, Ashman's D > 2) of log |d pos| between consecutive sampled
       frames; tol_pos = 2 r_pos (two rests within r_pos of the same place differ by at most 2 r_pos);
       thr_pos = one object width (sqrt(6) x median mask spread), kept for occupancy and prototypes;
       r_app = 2-means split of log |d app| frame-to-frame, used only if bimodal (Ashman's D > 2) --
       otherwise appearance never changes in this task family; thr_app = 2 r_app (as tol_pos = 2 r_pos);
  3. changes of several identities that overlap (within `gap` frames) form one event; the ACTED entity
     is the changed identity most central among the changed ones (effects are local around their cause;
     this decides only when >= 3 identities change), ties broken by the agent mask: closest to it when
     the change starts (the one the agent touched); the others are effects the world model must predict;
  4. state of identity k = (pos u, pos v, app r, g, b, covered); covered = never visible during a static
     interval (no move in progress) -- an object on top hides it.
Output per split: events_{split}.npz (t_start, t, e, before (N, K, 6), after, target = after[e], seg_start,
knock = several entities moved, episode) and labels_{split}.npz (per frame rest state + validity, for the
reader); report.json with thresholds and PRIVILEGED diagnostics (event match to qpos / button toggles).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sfa_code import two_means_threshold


def rest_runs(pos, app, vis, a, b, r_pos, r_app, m):
    """Rest runs of one identity in frames [a, b): [start, end, sum_pos, sum_app, n_visible]."""
    runs, cur, cand = [], None, None

    def near(run, p, c):
        return np.hypot(*(p - run[2] / run[4])) <= r_pos and np.abs(c - run[3] / run[4]).max() <= r_app

    for t in range(a, b):
        if not vis[t]:
            continue
        p, c = pos[t], app[t]
        if cur is not None and near(cur, p, c):
            cur[1] = t; cur[2] = cur[2] + p; cur[3] = cur[3] + c; cur[4] += 1; cand = None
            continue
        if cand is not None and near(cand, p, c):
            cand[1] = t; cand[2] = cand[2] + p; cand[3] = cand[3] + c; cand[4] += 1
            if cand[4] >= m:
                if cur is not None:
                    runs.append(cur)
                cur, cand = cand, None
        else:
            cand = [t, t, p.astype(np.float64).copy(), c.astype(np.float64).copy(), 1]
    if cur is not None:
        runs.append(cur)
    return runs


def bimodal_split(x):
    """2-means split of x; (threshold, D) with D = Ashman's separation; inf threshold if not bimodal."""
    thr, sep, _ = two_means_threshold(x)
    return (thr if sep > 2.0 else np.inf), sep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", type=Path, required=True, help="front-end dir (entities_*.npz, discover.json)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--no-contact-free", dest="contact_free", action="store_false",
                    help="ablation: count rest observations also while the agent touches the identity")
    ap.add_argument("--thresholds-from", type=Path, default=None,
                    help="events dir whose report.json fixes r_pos / tol_pos / r_app / thr_app (self-training rounds keep "
                         "the change definitions of the front-end round)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    disc = json.loads((a.entities / "discover.json").read_text())
    thr_pos = float(np.sqrt(6) * np.median([t["spread_median"] for t in disc["table"]]))
    E = {s: np.load(a.entities / f"entities_{s}.npz") for s in ("train", "val")}
    # appearance noise / change thresholds from train (frame-to-frame changes of visible identities)
    tr = E["train"]
    vis = tr["area"] >= 1
    term_tr = np.load(a.cache / "train_terminals.npy"); ep_tr = np.concatenate([[0], np.cumsum(term_tr[:-1])])
    pf = np.nonzero(tr["processed"])[0]                                          # consecutive sampled frames, same episode
    f0, f1 = pf[:-1], pf[1:]; f0, f1 = f0[ep_tr[f0] == ep_tr[f1]], f1[ep_tr[f0] == ep_tr[f1]]
    both = vis[f0] & vis[f1]
    dpos = np.linalg.norm(tr["pos"][f1] - tr["pos"][f0], axis=-1)[both]
    if (dpos > 0).sum() >= 10:
        r_pos_log, D0 = bimodal_split(np.log(dpos[dpos > 0] + 1e-6))
        r_pos = float(np.exp(r_pos_log)) if np.isfinite(r_pos_log) else 1.0
    else:                                                                        # only place identities: position never changes
        r_pos, D0 = 1.0, 0.0
    tol_pos = 2.0 * r_pos                      # two rests within r_pos of one place differ by <= 2 r_pos
    dapp = np.abs(tr["app"][f1] - tr["app"][f0]).max(-1)[both]
    if (dapp > 0).sum() >= 10:
        r_app_log, D1 = bimodal_split(np.log(dapp[dapp > 0] + 1e-6))
        r_app = float(np.exp(r_app_log)) if np.isfinite(r_app_log) else float(np.percentile(dapp, 99))
    else:                                                                        # no identity ever changes appearance
        r_app, D1 = np.inf, 0.0
    thr_app = None
    if a.thresholds_from is not None:
        fx = json.loads((a.thresholds_from / "report.json").read_text())
        thr_pos, r_pos, tol_pos, r_app, thr_app = fx["thr_pos"], fx["r_pos"], fx["tol_pos"], fx["r_app"], fx["thr_app"]
        D0 = D1 = D2 = None
    rep = {"thr_pos": thr_pos, "r_pos": r_pos, "r_pos_D": D0, "tol_pos": tol_pos, "r_app": r_app, "r_app_D": D1, "m": a.m,
           "thresholds_from": str(a.thresholds_from) if a.thresholds_from else None}
    if thr_app is not None:
        rep.update(thr_app=thr_app)
    for split in ("train", "val"):
        z = E[split]
        pos, app, area, agent, processed = z["pos"].astype(np.float64), z["app"].astype(np.float64), z["area"], z["agent"], z["processed"]
        vis = area >= 1
        n, K = area.shape
        # rest observations need the object free of the agent: unobserved while an agent pixel is within half an
        # object width of it (a cube held still in the gripper is in transit, a light under the finger unread)
        vis_rest = vis.copy()
        if a.contact_free:
            vg, ug = np.mgrid[0:64, 0:64]
            for t in np.nonzero(processed & vis.any(1))[0]:
                am = np.unpackbits(agent[t], axis=-1)[:, :64].astype(bool)
                if am.any():
                    d = np.hypot(ug[am][:, None] - pos[t, :, 0][None], vg[am][:, None] - pos[t, :, 1][None]).min(0)
                    vis_rest[t] &= d > thr_pos / 2
        term = np.load(a.cache / f"{split}_terminals.npy")
        ep_of = np.concatenate([[0], np.cumsum(term[:-1])]).astype(np.int64)
        starts = np.r_[0, np.nonzero(term)[0] + 1]; ends = np.r_[np.nonzero(term)[0] + 1, n]
        starts, ends = starts[starts < ends], ends[starts < ends]                 # a terminal last frame adds an empty episode
        eps = [e for e in range(min(len(starts), len(ends))) if processed[starts[e]]]
        runs = {(e, k): rest_runs(pos[:, k], app[:, k], vis_rest[:, k], starts[e], ends[e], r_pos, r_app, a.m) for e in eps for k in range(K)}
        if thr_app is None:
            # rest-to-rest appearance change: two rests of one state lie within r_app of it, so they differ by at
            # most 2 r_app (as tol_pos = 2 r_pos). A 2-means split of rest-to-rest differences fails when rests are
            # read steadily: the "no change" mode vanishes and the split falls between weak and strong changes.
            thr_app, D2 = (2.0 * r_app, None) if np.isfinite(r_app) else (np.inf, None)
            rep.update(thr_app=thr_app)
            print({"thr_pos": round(thr_pos, 2), "r_pos": round(r_pos, 3), "tol_pos": round(tol_pos, 3), "D_pos": round(D0, 2), "r_app": round(r_app, 4),
                   "thr_app": thr_app, "D_frame": round(D1, 2)}, flush=True)
        labp = np.zeros((n, K, 2), np.float32); laba = np.zeros((n, K, 3), np.float32); valid = np.zeros((n, K), bool)
        changes = []                                                             # (t0, t1, k, e)
        for (e, k), rs in runs.items():
            merged = []
            for r_ in rs:
                if merged:
                    pm, am = merged[-1][2] / merged[-1][4], merged[-1][3] / merged[-1][4]
                    if np.hypot(*(r_[2] / r_[4] - pm)) <= tol_pos and np.abs(r_[3] / r_[4] - am).max() <= thr_app:
                        merged[-1] = [merged[-1][0], r_[1], merged[-1][2] + r_[2], merged[-1][3] + r_[3], merged[-1][4] + r_[4]]
                        continue
                merged.append(list(r_))
            for i, r_ in enumerate(merged):
                labp[r_[0]:r_[1] + 1, k] = r_[2] / r_[4]; laba[r_[0]:r_[1] + 1, k] = r_[3] / r_[4]; valid[r_[0]:r_[1] + 1, k] = True
                if i:
                    changes.append((merged[i - 1][1] + 1, r_[0] - 1, k, e))
        changes.sort()
        # covered: never visible during a static interval (no change of any identity in progress)
        in_ch = np.zeros(n, bool)
        for t0, t1, _, _ in changes:
            in_ch[t0:t1 + 1] = True
        chg = np.r_[True, (in_ch[1:] != in_ch[:-1]) | (ep_of[1:] != ep_of[:-1])]
        st = np.nonzero(chg)[0]; ln = np.diff(np.r_[st, n])
        vsum = np.add.reduceat(vis.astype(np.int64), st, axis=0)
        ok = (~in_ch[st]) & (ln >= a.m) & processed[st]
        run_of = np.cumsum(chg) - 1
        cov = (vsum == 0)[run_of]; cov_valid = np.repeat(ok[:, None], K, 1)[run_of]
        # group overlapping changes into events; acted entity by agent contact
        v_, u_ = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        events, i = [], 0
        while i < len(changes):
            grp = [changes[i]]; t1 = changes[i][1]; j = i + 1
            while j < len(changes) and changes[j][3] == changes[i][3] and changes[j][0] <= t1 + a.gap:
                grp.append(changes[j]); t1 = max(t1, changes[j][1]); j += 1
            i = j
            # acted entity: the changed identity most central among the changed ones (effects are local around
            # their cause; decides only when >= 3 identities change), ties -> closest to the agent at change start
            ks_ = sorted({g[2] for g in grp})
            p_pre = {k: labp[max(min(g[0] for g in grp if g[2] == k) - 1, 0), k] for k in ks_}
            cen = {k: sum(np.hypot(*(p_pre[k] - p_pre[j])) for j in ks_ if j != k) for k in ks_}
            cmin = min(cen.values())
            central = {k for k in ks_ if cen[k] <= cmin + 1e-6}
            best, bd = None, np.inf
            for (c0, c1, k, e_) in grp:
                if k not in central:
                    continue
                p0 = labp[max(c0 - 1, 0), k]
                d = np.inf
                for t in range(max(c0 - 5, starts[e_]), min(c0 + 10, c1 + 1)):   # from the last rest frame on
                    am = np.unpackbits(agent[t], axis=-1)[:, :64].astype(bool)
                    if am.any():
                        d = min(d, float(np.min(np.hypot(u_[am] - p0[0], v_[am] - p0[1]))))
                # closest to the agent; ties (e.g. no agent seen) -> the longest change
                if best is None or d < bd or (d == bd and (c1 - c0) > (best[1] - best[0])):
                    best, bd = (c0, c1, k), d
            events.append((min(g[0] for g in grp), t1, best[2], len(grp)))

        def state_at(t):
            s = np.zeros((K, 6), np.float32)
            e0 = starts[ep_of[t]]; e1 = ends[ep_of[t]]
            for k in range(K):
                vv = np.nonzero(valid[e0:e1, k])[0]
                if len(vv):
                    tt = e0 + vv[np.abs(vv + e0 - t).argmin()]
                    s[k, :2] = labp[tt, k]; s[k, 2:5] = laba[tt, k]
                cc = np.nonzero(cov_valid[e0:e1, k])[0]
                if len(cc):
                    s[k, 5] = cov[e0 + cc[np.abs(cc + e0 - t).argmin()], k]
            return s

        ts = np.array([x[0] for x in events]); te = np.array([x[1] for x in events]); ee = np.array([x[2] for x in events])
        before = np.stack([state_at(max(t - 1, 0)) for t in ts]); after = np.stack([state_at(min(t + 1, n - 1)) for t in te])
        ep = ep_of[ts]
        seg = starts[ep].copy()
        same = np.r_[False, ep[1:] == ep[:-1]]
        seg[same] = te[:-1][same[1:]] + 1
        seg = np.minimum(seg, ts)
        moved = (np.linalg.norm(after[..., :2] - before[..., :2], axis=-1) > tol_pos).sum(1)
        np.savez_compressed(a.out / f"events_{split}.npz", t_start=ts, t=te, e=ee, before=before, after=after,
                            target=after[np.arange(len(ee)), ee], seg_start=seg, knock=moved > 1, episode=ep,
                            n_changed=np.array([x[3] for x in events]), thr_pos=thr_pos, tol_pos=tol_pos, thr_app=thr_app)
        np.savez_compressed(a.out / f"labels_{split}.npz", pos=labp, app=laba, valid=valid, cov=cov, cov_valid=cov_valid)
        r = {"events": int(len(ee)), "per_episode": float(len(ee) / max(1, len(eps))), "episodes": len(eps),
             "changed_entities_pct": np.percentile([x[3] for x in events], [50, 90, 99]).tolist() if events else None,
             "acted_entity_counts": np.bincount(ee, minlength=K).tolist(), "label_valid_frac_processed": float(valid[processed].mean()),
             "covered_frac": float(cov[cov_valid].mean()) if cov_valid.any() else None}
        # PRIVILEGED diagnostics: event times vs qpos object motion / button toggles
        refm = np.zeros(n - 1, bool)
        same_ep = ep_of[1:] == ep_of[:-1]
        if (a.cache / f"{split}_qpos.npy").exists():
            q = np.load(a.cache / f"{split}_qpos.npy", mmap_mode="r")
            for e in eps:
                s0, s1 = starts[e], ends[e]
                refm[s0:s1 - 1] |= np.abs(np.diff(np.asarray(q[s0:s1, 14:]), axis=0)).max(1) > 2e-3
        if (a.cache / f"{split}_button_states.npy").exists():
            b = np.load(a.cache / f"{split}_button_states.npy", mmap_mode="r")
            for e in eps:
                s0, s1 = starts[e], ends[e]
                refm[s0:s1 - 1] |= (np.asarray(b[s0 + 1:s1]) != np.asarray(b[s0:s1 - 1])).any(1)
        refm &= same_ep
        d_ = np.diff(np.r_[0, refm.astype(np.int8), 0]); rs_, re_ = np.nonzero(d_ == 1)[0] + 1, np.nonzero(d_ == -1)[0]
        if len(rs_) and len(ts):
            r["privileged_recall"] = float(np.mean([np.any((ts <= y + 15) & (te >= x - 15)) for x, y in zip(rs_, re_)]))
            r["privileged_precision"] = float(np.mean([np.any((rs_ <= y + 15) & (re_ >= x - 15)) for x, y in zip(ts, te)]))
            r["privileged_ref_events"] = int(len(rs_))
            if "cube" in a.cache.name and (a.cache / f"{split}_qpos.npy").exists():
                # picks and places: reference intervals where some cube's net xy displacement >= 2 cm (half a cube)
                sl = [c for c in (14, 21, 28, 35) if c + 3 <= q.shape[1]]
                disp = np.array([max(np.linalg.norm(np.asarray(q[min(y + 1, n - 1), c:c + 2]) - np.asarray(q[x, c:c + 2])) for c in sl) for x, y in zip(rs_, re_)])
                big = disp >= 0.02
                r["privileged_recall_moves_2cm"] = float(np.mean([np.any((ts <= y + 15) & (te >= x - 15)) for x, y in zip(rs_[big], re_[big])])) if big.any() else None
                r["privileged_ref_moves_2cm"] = int(big.sum())
        rep[split] = r
        print(split, json.dumps(r), flush=True)
    (a.out / "report.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
