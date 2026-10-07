#!/usr/bin/env python3
"""Cube direction A, step 2: label-free move events and rest-state pseudo-labels from object tracks.

Input: cube_discover.py tracks (per frame and object: centroid (u, v) in px and visible mass). No
simulator state is an input. Per object and episode:
  1. rest runs = >= m visible frames whose centroid stays within r px of the run's running mean
     (invisible frames -- shadow, gripper -- do not break a run; a short visible blip elsewhere,
     e.g. during a carry, never reaches m frames);
  2. consecutive runs further apart than `thr` px are a move; closer runs are one rest (partial
     occlusion shifts a centroid slightly). thr = one object width L: an object displaced by less
     than its own width still overlaps its old footprint (OGBench's 4 cm success radius is the same
     notion). L = sqrt(6) x the median pixel spread of the object clusters (a uniform square of side L
     has spread L / sqrt(6)). Reported for reference: displacement between runs <= `adj` frames apart
     (no carry fits in so few frames, so pure noise). 2-means on all displacements failed (it cut
     through the broad move distribution: thr 8.4 px, recall .71, job 57121);
  3. a move of object k spans from the end of its previous rest to the start of its next rest.
Pseudo-labels: inside a rest (from its first to its last visible frame) the object's state is the
rest position, also on frames where it is not visible -- the cube analogue of the puzzle's
segment-majority labels.
Coverage bit (also constant between events): in every static interval (no move of any object in
progress, >= m frames) an object whose colour track never sees it is covered -- another object on top
hides its top face, while shadows hide it only part of the time. (Visible fraction per interval, val,
PRIVILEGED check: covered cubes < 1% in 97% of intervals, free cubes > 70% in 85%, never seen 1.3%.
A 2-means threshold on the fraction, job 57130, landed at .56 and mislabelled shadowed free cubes.)
Output per split mirrors cube_events.py; state = K x (u, v, covered).
Diagnostic only (PRIVILEGED): recall/precision against the qpos moves of cube_events.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np



def rest_runs(uv, vis, a, b, r, m):
    """Rest runs of one object in frames [a, b): list of [start, end, sum_xy, n_visible]."""
    runs, cur, cand = [], None, None
    for t in range(a, b):
        if not vis[t]:
            continue
        p = uv[t]
        if cur is not None and np.hypot(*(p - cur[2] / cur[3])) <= r:
            cur[1] = t; cur[2] = cur[2] + p; cur[3] += 1; cand = None
            continue
        if cand is not None and np.hypot(*(p - cand[2] / cand[3])) <= r:
            cand[1] = t; cand[2] = cand[2] + p; cand[3] += 1
            if cand[3] >= m:
                if cur is not None:
                    runs.append(cur)
                cur, cand = cand, None
        else:
            cand = [t, t, p.astype(np.float64).copy(), 1]
    if cur is not None:
        runs.append(cur)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", type=Path, required=True, help="cube_discover.py output dir")
    ap.add_argument("--cache", type=Path, required=True, help="terminals only (qpos only with --ref-events)")
    ap.add_argument("--r", type=float, default=1.0, help="px; rest-run radius")
    ap.add_argument("--m", type=int, default=5, help="visible frames to accept a rest run (debounce)")
    ap.add_argument("--adj", type=int, default=2, help="frames; run pairs this close are physically not a move")
    ap.add_argument("--ref-events", type=Path, default=None, help="PRIVILEGED cube_events.py dir, diagnostic only")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    disc = json.loads((a.discover / "discover.json").read_text())
    runs_all, thr = {}, None
    for split in ("train", "val"):
        z = np.load(a.discover / f"tracks_{split}.npz")
        uv, mass = z["uv"].astype(np.float64), z["mass"]
        term = np.load(a.cache / f"{split}_terminals.npy")
        first = np.r_[0, np.nonzero(term)[0] + 1]
        last = np.r_[np.nonzero(term)[0] + 1, len(term)]
        first, last = first[first < len(term)], last[: len(first[first < len(term)])]
        K = uv.shape[1]
        runs_all[split] = [[rest_runs(uv[:, k], mass[:, k] >= 3, s, e, a.r, a.m) for s, e in zip(first, last)] for k in range(K)]
        if split == "train":                                   # threshold from train only
            pairs = [(np.hypot(*(r1[2] / r1[3] - r0[2] / r0[3])), r1[0] - r0[1]) for k in range(K)
                     for ep in runs_all[split][k] for r0, r1 in zip(ep[:-1], ep[1:])]
            disp, gap = np.array(pairs).T
            spread = [disc["table"][c]["spread_median"] for g in disc["groups"] for c in g]
            thr = float(np.sqrt(6) * np.mean(spread))
            noise = np.percentile(disp[gap <= a.adj], [50, 90, 99]).round(2).tolist()
            print({"run_pairs": len(disp), "adjacent_pairs": int((gap <= a.adj).sum()), "adjacent_disp_pct_50_90_99": noise,
                   "thr_px": round(thr, 2), "move_frac": round(float((disp > thr).mean()), 3)}, flush=True)
    rep = {"thr_px": thr, "thr_rule": "object width", "adjacent_disp_pct_50_90_99": noise, "r": a.r, "m": a.m}
    for split in ("train", "val"):
        z = np.load(a.discover / f"tracks_{split}.npz")
        n, K = z["mass"].shape
        term = np.load(a.cache / f"{split}_terminals.npy")
        ep_of = np.concatenate([[0], np.cumsum(term[:-1])]).astype(np.int64)
        first = np.r_[0, np.nonzero(term)[0] + 1][: ep_of[-1] + 1]
        pos = np.zeros((n, K, 2), np.float32); valid = np.zeros((n, K), bool)
        moves = []                                                  # (t_start, t_end, k, before_k, after_k)
        for k in range(K):
            for e, runs in enumerate(runs_all[split][k]):
                merged = []
                for r_ in runs:                                     # merge rests closer than thr
                    if merged and np.hypot(*(r_[2] / r_[3] - merged[-1][2] / merged[-1][3])) <= thr:
                        merged[-1] = [merged[-1][0], r_[1], merged[-1][2] + r_[2], merged[-1][3] + r_[3]]
                    else:
                        merged.append(list(r_))
                for i, r_ in enumerate(merged):
                    c = r_[2] / r_[3]
                    pos[r_[0]:r_[1] + 1, k] = c; valid[r_[0]:r_[1] + 1, k] = True
                    if i:
                        p = merged[i - 1]
                        moves.append((p[1] + 1, r_[0] - 1, k, p[2] / p[3], c))
        moves.sort()

        # coverage bit per static interval (no move of any object in progress)
        in_move = np.zeros(n, bool)
        for mv in moves:
            in_move[mv[0]:mv[1] + 1] = True
        change = np.r_[True, (in_move[1:] != in_move[:-1]) | (ep_of[1:] != ep_of[:-1])]
        starts = np.nonzero(change)[0]
        lens = np.diff(np.r_[starts, n])
        vis_sum = np.add.reduceat((z["mass"] >= 3).astype(np.int64), starts, axis=0)
        ok = np.repeat(((~in_move[starts]) & (lens >= a.m))[:, None], K, 1)
        run_of = np.cumsum(change) - 1
        cov = (vis_sum == 0)[run_of]; cov_valid = ok[run_of]
        rep.setdefault("coverage", {})[split] = {"static_intervals": int(ok[:, 0].sum()),
                                                 "covered_rate": float((vis_sum == 0)[ok].mean())}

        def state_at(t):
            """Each object's (u, v, covered) at frame t (nearest labelled frame of the same episode)."""
            s = np.zeros((K, 3), np.float32)
            e0 = first[ep_of[t]]; e1 = first[ep_of[t] + 1] if ep_of[t] + 1 < len(first) else n
            for k in range(K):
                v = np.nonzero(valid[e0:e1, k])[0]
                if valid[t, k]:
                    s[k, :2] = pos[t, k]
                elif len(v):
                    s[k, :2] = pos[e0 + v[np.abs(v + e0 - t).argmin()], k]
                c = np.nonzero(cov_valid[e0:e1, k])[0]
                if len(c):
                    s[k, 2] = cov[e0 + c[np.abs(c + e0 - t).argmin()], k]
            return s

        ts = np.array([mv[0] for mv in moves]); te = np.array([mv[1] for mv in moves]); kk = np.array([mv[2] for mv in moves])
        before = np.stack([state_at(max(t - 1, 0)) for t in ts]); after = np.stack([state_at(min(t + 1, n - 1)) for t in te])
        before[np.arange(len(moves)), kk, :2] = np.stack([mv[3] for mv in moves])
        after[np.arange(len(moves)), kk, :2] = np.stack([mv[4] for mv in moves])
        others = np.ones((len(moves), K), bool); others[np.arange(len(moves)), kk] = False
        knock = (np.linalg.norm(after[..., :2] - before[..., :2], axis=-1) * others).max(1) > thr
        ep = ep_of[ts]
        seg = first[ep].copy()
        same = np.r_[False, ep[1:] == ep[:-1]]
        seg[same] = te[:-1][same[1:]] + 1
        seg = np.minimum(seg, ts)
        np.savez_compressed(a.out / f"cube_events_{split}.npz", t_start=ts, t=te, k=kk, before=before, after=after,
                            target_xy=after[np.arange(len(moves)), kk, :2], seg_start=seg, knock=knock, episode=ep,
                            lo=np.array([0, 0, 0], np.float32), hi=np.array([64, 64, 1], np.float32), thr_px=thr, n_bin=1)
        np.savez_compressed(a.out / f"labels_{split}.npz", pos=pos, valid=valid, cov=cov, cov_valid=cov_valid)
        r = {"moves": int(len(moves)), "per_episode": float(len(moves) / (ep_of[-1] + 1)), "knock_frac": float(knock.mean()),
             "covered_frac_labelled": float(cov[cov_valid].mean()), "cov_valid_frac": cov_valid.mean(0).round(4).tolist(),
             "moved_object_covered_before": float(before[np.arange(len(moves)), kk, 2].mean()),
             "label_valid_frac": valid.mean(0).round(4).tolist(), "move_len_pct": np.percentile(te - ts, [5, 50, 95]).tolist()}
        if a.ref_events is not None:                                # PRIVILEGED diagnostic
            ref = np.load(a.ref_events / f"cube_events_{split}.npz")
            obj2cube = {d["object"]: d["cube"] for d in disc["privileged_diagnostic"] if d}
            used = np.zeros(len(ref["t"]), bool); hits, offs = 0, []
            for t0, t1, k in zip(ts, te, kk):
                cand = np.nonzero((ref["k"] == obj2cube.get(int(k), -1)) & ~used & (ref["t_start"] <= t1 + 30) & (ref["t"] >= t0 - 30))[0]
                if len(cand):
                    j = cand[np.abs(ref["t"][cand] - t1).argmin()]
                    used[j] = True; hits += 1; offs.append(int(t1 - ref["t"][j]))
            dist = np.linalg.norm(ref["after"][:, :, :2] - ref["before"][:, :, :2], axis=-1)[np.arange(len(ref["k"])), ref["k"]]
            r["recall_by_true_xy_dist_cm"] = {f"{lo_}-{hi_}": float(used[(dist >= lo_ / 100) & (dist < hi_ / 100)].mean())
                                              for lo_, hi_ in ((0, 4), (4, 8), (8, 16), (16, 100))}
            q = np.load(a.cache / f"{split}_qpos.npy", mmap_mode="r")
            xyz = np.stack([np.asarray(q[:, s_:s_ + 3]) for s_ in (14, 21, 28, 35)[:K]], 1)
            tcov = np.zeros((n, K), bool)
            for i_ in range(K):
                for j_ in range(K):
                    if i_ != j_:
                        tcov[:, i_] |= (xyz[:, j_, 2] > xyz[:, i_, 2] + 0.02) & (np.linalg.norm(xyz[:, i_, :2] - xyz[:, j_, :2], axis=-1) < 0.03)
            agree = [float((cov[:, k_] == tcov[:, obj2cube[k_]])[cov_valid[:, k_]].mean()) for k_ in range(K) if k_ in obj2cube]
            r["coverage_label_agreement_with_qpos"] = agree
            r["privileged_diagnostic"] = {"recall": float(used.mean()), "precision": float(hits / max(1, len(ts))),
                                          "end_offset_pct": np.percentile(offs, [5, 50, 95]).tolist() if offs else None}
        rep[split] = r
        print(split, json.dumps(r), flush=True)
    (a.out / "cube_events_report.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
