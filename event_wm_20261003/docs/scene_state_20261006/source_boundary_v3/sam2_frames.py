#!/usr/bin/env python3
"""Unified front end v2: SAM 2 segments of sparse frames + identity rules -> generic entity table.

No tracker (job 57245: video tracking kept only ~60% of cube frames and was slow). Every `stride`-th frame:
SAM 2.1 point-grid proposals (finest valid mask per point, <= max_frac of the image) -> segments with
centroid, area, mean colour. Identities, one rule set for every task family:
  CLUSTERS   greedy RGB clustering of all train segments (merge radius = 3 x the median within-segment colour
             deviation); train and val segments labelled by the nearest centre within that radius;
  VARIANTS   one object shows several clusters (lit / shaded faces, tinted under the agent): clusters whose
             colour-flip link (B appears where A was, in the next sampled frame, while A has left the frame)
             is in the upper group of a bimodal 2-means split are merged into one type (free,
             single-instance clusters only: not a fixed-place thing, not a background / agent colour);
  AGENT      a type whose largest segment per sampled frame moves by more than one object width between
             consecutive sampled frames in most frames (> 1/2; objects rest until the agent moves them);
  ANCHORING  an object type is LOCATION-anchored if most of its segments sit at fixed places (density
             peeling, radius thr_pos / 2, places occupied in >= 30% of the sampled frames): one identity per
             place, and the location-anchored types that share a place are merged; its position is constant
             and its state is the appearance of the place (colour of the uncovered part of the place disc,
             pixels weighted by their temporal variance over train frames -- where the state shows; unobserved while agent segments cover most of it) -- a light, a button, a handle's rest place. Other single-instance types are COLOUR-anchored: one
             identity per type (union of its segments near the one picked by continuity: nearest the
             previous position within one object width, else the largest); position is the state and the
             appearance is constant (the colour is the identity) -- cube, drawer.
So a cube is identified by its colour and a light by its place, with no domain switch. Output in the
u_events.py format: entities_{split}.npz (pos, app, area; agent = packed union of agent segments;
processed = sampled frames), discover.json (identities, spread; PRIVILEGED identity -> cube matching).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from sam2_entities import MODEL
from sfa_code import two_means_threshold


def proposals_batch(model, proc, frames, res, grid, iou_min, max_frac, dev, chunk=256):
    """SAM 2 point-grid proposals for a batch of 64x64 frames, all on the GPU: one image encoding per frame,
    all points in one decoder pass, low-resolution masks pooled to 64x64, finest valid mask per point,
    greedy NMS (mask IoU > 0.5). Returns per frame a list of 64x64 boolean masks."""
    import torch
    import torch.nn.functional as F
    from PIL import Image

    imgs = [Image.fromarray(f).resize((res, res), Image.BICUBIC) for f in frames]
    g = (np.arange(grid) + 0.5) * res / grid
    pts = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    inp = proc(images=imgs, return_tensors="pt").to(dev)
    scale = 1024.0 / res                                                          # processor maps points to the 1024 px input
    P = len(pts)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        emb = model.get_image_embeddings(inp["pixel_values"])
        all_m, all_s = [], []
        for b in range(len(frames)):
            eb = [e_[b:b + 1] for e_ in emb]
            mb, sb = [], []
            for c in range(0, P, chunk):
                ip = torch.as_tensor(pts[c:c + chunk] * scale, device=dev).float()[None, :, None, :]
                il = torch.ones(ip.shape[:3], dtype=torch.long, device=dev)
                out = model(image_embeddings=eb, input_points=ip, input_labels=il, multimask_output=True)
                pm = out.pred_masks.float()[0]                                      # (p, 3, h, w)
                f_ = pm.shape[-1] // 64
                mm = (pm > 0).float().view(-1, 1, *pm.shape[-2:])
                m64 = F.avg_pool2d(mm, f_) > 0.5
                anyp = F.max_pool2d(mm, f_) > 0
                empty = m64.flatten(1).sum(1) == 0
                m64[empty] = anyp[empty]
                mb.append(m64.view(-1, 3, 64, 64)); sb.append(out.iou_scores.float()[0])
            all_m.append(torch.cat(mb)); all_s.append(torch.cat(sb))
    m64 = torch.stack(all_m); sc = torch.stack(all_s)                              # (B, P, 3, 64, 64), (B, P, 3)
    B = m64.shape[0]
    area = m64.flatten(3).sum(-1).float()                                        # (B, P, 3)
    valid = (sc >= iou_min) & (area >= 1) & (area <= max_frac * 4096)
    big = torch.where(valid, area, torch.full_like(area, 1e9))
    pick = big.argmin(-1)                                                        # finest valid mask per point
    okp = valid.any(-1)
    res_masks = []
    for b in range(B):
        idx = torch.nonzero(okp[b]).squeeze(-1)
        if len(idx) == 0:
            res_masks.append([]); continue
        M = m64[b, idx, pick[b, idx]].flatten(1).float()                         # (n, 4096)
        s_ = sc[b, idx, pick[b, idx]]
        order = torch.argsort(s_, descending=True)
        M = M[order]
        inter = M @ M.T
        ar = M.sum(1)
        iou = inter / (ar[:, None] + ar[None] - inter).clamp(min=1)
        keep = []
        sup = torch.zeros(len(M), dtype=torch.bool, device=M.device)
        iou_c = iou.cpu().numpy()
        for i in range(len(M)):
            if sup[i]:
                continue
            keep.append(i)
            sup |= torch.as_tensor(iou_c[i] > 0.5, device=M.device)
        res_masks.append([M[i].view(64, 64).bool().cpu().numpy() for i in keep])
    return res_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--train-episodes", type=int, default=150)
    ap.add_argument("--val-episodes", type=int, default=30)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--frame-batch", type=int, default=4)
    ap.add_argument("--iou-min", type=float, default=0.75)
    ap.add_argument("--max-frac", type=float, default=0.02)
    ap.add_argument("--merge-mult", type=float, default=3.0, help="colour merge radius in units of the median within-segment colour deviation")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], help="segment only these splits (identity stage needs both)")
    ap.add_argument("--train-ep-start", type=int, default=0, help="first train episode (sharded segmentation)")
    ap.add_argument("--segments", type=Path, default=None, help="reuse segments_{split}.npz of an earlier run (no SAM 2)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    v_, u_ = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    segs = {}                                                                   # split -> list of dicts per segment
    meta = {}
    if a.segments is None:
        import torch  # noqa: F401
        from transformers import Sam2Model, Sam2Processor
        dev = "cuda"
        proc = Sam2Processor.from_pretrained(MODEL)
        img_model = Sam2Model.from_pretrained(MODEL).to(dev).eval()
    for split, n_ep in (("train", a.train_episodes), ("val", a.val_episodes)):
        if split not in a.splits:
            continue
        term = np.load(a.cache / f"{split}_terminals.npy")
        if a.segments is not None:
            z = np.load(a.segments / f"segments_{split}.npz")
            meta[split] = {"n": len(term), "frames": z["frames"]}
            segs[split] = [{"t": int(t), "pos": p_, "area": int(ar), "col": c, "col_dev": float(cd), "spread": float(sp), "mask": mk}
                           for t, p_, ar, c, cd, sp, mk in zip(z["t"], z["pos"], z["area"], z["col"], z["col_dev"], z["spread"], z["mask"])]
            continue
        obs = np.load(a.cache / f"{split}_observations.npy", mmap_mode="r")
        starts = np.r_[0, np.nonzero(term)[0] + 1]; ends = np.r_[np.nonzero(term)[0] + 1, len(term)]
        starts, ends = starts[starts < ends], ends[starts < ends]
        e0 = a.train_ep_start if split == "train" else 0
        frames = np.concatenate([np.arange(starts[e], ends[e], a.stride) for e in range(e0, min(e0 + n_ep, len(starts)))])
        meta[split] = {"n": len(term), "frames": frames}
        rows = []
        bs = a.frame_batch
        for c0 in range(0, len(frames), bs):
          tb = frames[c0:c0 + bs]
          frs = np.asarray(obs[tb])
          for t, fr, masks in zip(tb, frs, proposals_batch(img_model, proc, frs, a.res, a.grid, a.iou_min, a.max_frac, dev)):
            for m in masks:
                ar = m.sum()
                cu, cv = (m * u_).sum() / ar, (m * v_).sum() / ar
                pix = fr[m].astype(np.float32)
                rows.append({"t": int(t), "pos": np.array([cu, cv], np.float32), "area": int(ar), "col": pix.mean(0) / 255.0,
                             "col_dev": float(np.linalg.norm(pix - pix.mean(0), axis=-1).mean() / 255.0),
                             "spread": float(np.sqrt(((m * ((u_ - cu) ** 2 + (v_ - cv) ** 2)).sum()) / ar)), "mask": np.packbits(m, axis=-1)})
          if (c0 // bs) % 50 == 0:
            print(split, c0, len(frames), len(rows), round((time.time() - t0) / 60, 1), "min", flush=True)
        segs[split] = rows
        np.savez(a.out / f"segments_{split}.npz", frames=frames, t=np.array([r["t"] for r in rows]), pos=np.stack([r["pos"] for r in rows]),
                 area=np.array([r["area"] for r in rows]), col=np.stack([r["col"] for r in rows]), col_dev=np.array([r["col_dev"] for r in rows]),
                 spread=np.array([r["spread"] for r in rows]), mask=np.stack([r["mask"] for r in rows]))
    if set(a.splits) != {"train", "val"}:
        print({"segments_only": a.splits, "minutes": round((time.time() - t0) / 60, 1)}, flush=True)
        return
    # COLOUR CLUSTERS (train segments): greedy RGB clustering, merge radius = 3 x the median within-segment
    # colour deviation; every segment (train and val) is then labelled by the nearest final centre, -1 beyond
    # the radius (one rule for both splits). (A radius from IoU-matched segment pairs was tried: rendering is
    # deterministic, so it collapses to ~0.01 and shatters the clusters.)
    tr = segs["train"]
    C = np.stack([r["col"] for r in tr]).astype(np.float64)
    merge = a.merge_mult * float(np.median([r["col_dev"] for r in tr]))
    m_D = None
    centres = []
    for i in np.argsort(-C.sum(1)):
        if centres:
            cc = np.stack([c[0] / c[1] for c in centres]); d = np.linalg.norm(cc - C[i], axis=-1); j = int(d.argmin())
            if d[j] < merge:
                centres[j][0] += C[i]; centres[j][1] += 1; continue
        centres.append([C[i].copy(), 1])
    cent = np.stack([c[0] / c[1] for c in centres])

    def label(Cx):
        if not len(Cx):
            return np.zeros(0, int)
        d = np.linalg.norm(Cx[:, None] - cent[None], axis=-1)
        lb = d.argmin(1); lb[d.min(1) >= merge] = -1
        return lb

    lab_raw = label(C)
    thr_pos = float(np.sqrt(6) * np.median([r["spread"] for r in tr]))
    frames_tr = meta["train"]["frames"]
    term = np.load(a.cache / "train_terminals.npy"); ep_of = np.concatenate([[0], np.cumsum(term[:-1])])
    seg_t = np.array([r["t"] for r in tr]); seg_p = np.stack([r["pos"] for r in tr]); seg_a = np.array([r["area"] for r in tr])
    fidx = {t: i for i, t in enumerate(frames_tr)}
    seg_f = np.array([fidx[t] for t in seg_t])

    def places(P, F, n_frames, r, min_occ=0.3, res=4):
        """Fixed places of a set of segment centroids P (sampled-frame index F): density peeling on a 1/res px
        grid -- take the densest disc of radius r, place = mean of its points, remove them, repeat; a place is
        kept while it is occupied in >= min_occ of the sampled frames (the presence rule used for types).
        -> (kept centres (n, 2), fraction of the points at kept places, member indices per place)."""
        from scipy.ndimage import convolve
        g = 64 * res
        rr = int(np.ceil(r * res)); yy, xx = np.mgrid[-rr:rr + 1, -rr:rr + 1]; disc = (np.hypot(xx, yy) <= r * res).astype(np.float64)
        ij = np.clip(np.round(P * res).astype(int), 0, g - 1)
        left = np.ones(len(P), bool); out, members = [], []; at = 0
        while left.any():
            H = np.zeros((g, g)); np.add.at(H, (ij[left, 1], ij[left, 0]), 1.0)
            D = convolve(H, disc, mode="constant")
            y0, x0 = np.unravel_index(D.argmax(), D.shape)
            c = np.array([x0, y0], np.float64) / res
            inside = left & (np.hypot(*(P - c).T) <= r)
            if not inside.any() or len(np.unique(F[inside])) < min_occ * n_frames:
                break
            out.append(P[inside].mean(0)); members.append(np.nonzero(inside)[0]); at += inside.sum(); left &= ~inside
        return (np.array(out) if out else np.zeros((0, 2))), at / max(1, len(P)), members

    def pick(cands, P, A, prev):
        """Continuity: the candidate nearest the previous position if within one object width, else the largest."""
        if prev is not None:
            dd = [np.hypot(*(P[c] - prev)) for c in cands]
            if min(dd) <= thr_pos:
                return cands[int(np.argmin(dd))]
        return max(cands, key=lambda c: A[c])
    # COLOUR VARIANTS: one object can show several colour clusters (lit / shaded faces, tinted under the agent).
    # flip(A -> B) = P(a B segment within thr_pos / 2 of p in the next sampled frame | an A segment at p and no A
    # segment anywhere in that frame); link(A, B) = max of both directions. Clusters are merged when their link
    # falls in the upper group of a 2-means split of all links, used only if bimodal (Ashman's D > 2).
    # Clusters seen several times per frame (agent parts, lights) never leave the frame, so they never merge.
    nc = len(cent)
    by_f = {}
    for i in np.nonzero(lab_raw >= 0)[0]:
        by_f.setdefault(seg_f[i], []).append(i)
    gone, flip = np.zeros(nc), np.zeros((nc, nc))
    for f, ii in by_f.items():
        nxt = by_f.get(f + 1)
        if nxt is None or ep_of[frames_tr[f + 1]] != ep_of[frames_tr[f]]:
            continue
        ln, pn = lab_raw[nxt], seg_p[nxt]
        for i in ii:
            A = lab_raw[i]
            if (ln == A).any():
                continue
            gone[A] += 1
            dn = np.hypot(*(pn - seg_p[i]).T)
            for B in set(ln[dn <= thr_pos / 2]):
                flip[A, B] += 1
    rate = flip / np.maximum(gone[:, None], 1)
    rate[gone < 30] = 0.0                                                       # too few events to estimate
    link = np.maximum(rate, rate.T)
    # only free clusters merge: a cluster sitting at fixed places (a bar, a light) is a different kind of
    # identity from a mover, even when a shaded mover takes its colour; place variants merge by place later
    # and only single-instance clusters (one segment per frame where present): an object's colour variants are
    # never several things at once; a background or agent colour (many pieces per frame) is revealed or
    # passes where an object was, which is not a variant
    fixed = np.zeros(nc, bool); single = np.zeros(nc, bool)
    for k in range(nc):
        sk = lab_raw == k
        cnt_k = np.bincount(seg_f[sk], minlength=len(frames_tr))
        if sk.any() and (cnt_k > 0).mean() >= 0.05:
            fixed[k] = places(seg_p[sk], seg_f[sk], len(frames_tr), thr_pos / 2)[1] >= 0.5
            single[k] = np.median(cnt_k[cnt_k > 0]) <= 1
    iu = np.triu_indices(nc, 1)
    ok_k = ~fixed & single
    free_pair = ok_k[iu[0]] & ok_k[iu[1]]
    iu = (iu[0][free_pair], iu[1][free_pair])
    lv = link[iu] if len(iu[0]) else np.zeros(1)
    c2 = np.array([lv.min(), lv.max()], np.float64)
    for _ in range(50):
        lb2 = np.abs(lv[:, None] - c2[None]).argmin(1)
        c2 = np.array([lv[lb2 == j].mean() if (lb2 == j).any() else c2[j] for j in range(2)])
    thr_link = float(c2.mean())
    lo, hi = lv[lv <= thr_link], lv[lv > thr_link]
    D_link = float(abs(c2[1] - c2[0]) / (np.sqrt(0.5 * (lo.var() + (hi.var() if len(hi) else 0))) + 1e-9))
    parent = list(range(nc))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    if D_link > 2:
        for A, B in zip(*iu):
            if link[A, B] > thr_link:
                ra, rb = root(A), root(B); parent[max(ra, rb)] = min(ra, rb)
    rep = np.array([root(k) for k in range(nc)])
    lab = np.where(lab_raw >= 0, rep[np.maximum(lab_raw, 0)], -1)
    cent_rgb = np.stack([C[lab == k].mean(0) if (lab == k).any() else cent[k] for k in range(nc)])
    present_raw = [float((np.bincount(seg_f[lab_raw == k], minlength=len(frames_tr)) > 0).mean()) for k in range(nc)]
    raw = [{"cluster": k, "rgb": (cent[k] * 255).round().astype(int).tolist(), "present": round(present_raw[k], 3), "merged_into": int(rep[k]),
            "fixed": bool(fixed[k]), "single": bool(single[k])}
           for k in range(nc) if present_raw[k] >= 0.05]
    flips = [{"A": int(A), "B": int(B), "link": round(float(link[A, B]), 3)} for A, B in zip(*iu) if link[A, B] >= 0.05]
    types = []
    for k in sorted(set(lab[lab >= 0].tolist())):
        sel = np.nonzero(lab == k)[0]
        cnt = np.bincount(seg_f[sel], minlength=len(frames_tr))
        if (cnt > 0).mean() < 0.3:                                              # rare colours = segmentation noise
            continue
        # one segment of this type per sampled frame (continuity rule `pick`) -> motion between consecutive
        # sampled frames
        by_ft = {}
        for i in sel:
            by_ft.setdefault(seg_f[i], []).append(i)
        big_f, big_p, prev, prev_ep = [], [], None, None
        for f in sorted(by_ft):
            if ep_of[frames_tr[f]] != prev_ep:
                prev, prev_ep = None, ep_of[frames_tr[f]]
            i = pick(by_ft[f], seg_p, seg_a, prev)
            big_f.append(f); big_p.append(seg_p[i]); prev = seg_p[i]
        big_f, big_p = np.array(big_f), np.array(big_p)
        cons = (np.diff(big_f) == 1) & (ep_of[frames_tr[big_f[1:]]] == ep_of[frames_tr[big_f[:-1]]])
        mv = np.hypot(*(big_p[1:] - big_p[:-1]).T)[cons] > thr_pos
        # agent statistic: the type's largest segment per sampled frame (the main body of a multi-part agent)
        lg = np.array([max(by_ft[f], key=lambda i: seg_a[i]) for f in big_f])
        mv_big = np.hypot(*(seg_p[lg[1:]] - seg_p[lg[:-1]]).T)[cons] > thr_pos
        types.append({"type": int(k), "clusters": [int(c) for c in np.nonzero(rep == k)[0]], "present": float((cnt > 0).mean()),
                      "count": int(np.median(cnt[cnt > 0])), "moving_frac": float(mv.mean()) if len(mv) else 1.0,
                      "big_move": float(mv_big.mean()) if len(mv_big) else 1.0})
    # identity anchoring: a type is LOCATION-anchored if most of its segments sit at fixed places (`places`)
    # -- lights, buttons; location-anchored types that
    # share a place are one identity (its colour is its state); otherwise COLOUR-anchored (cube, drawer).
    # AGENT: the type's largest segment moves by more than one object width between consecutive sampled
    # frames in most frames (> 1/2) -- objects rest until the agent moves them.
    anchored, colour_ids = [], []
    for t_ in types:
        k = t_["type"]
        sel = lab == k
        cc, frac_at, mem = places(seg_p[sel], seg_f[sel], len(frames_tr), thr_pos / 2)
        seg_sp = np.array([r["spread"] for r in tr])[sel]
        rad = [float(np.sqrt(6) / 2 * np.median(seg_sp[m])) for m in mem]     # half-width of the object at the place
        t_["anchor"] = "location" if frac_at >= 0.5 else "colour"
        t_["places"], t_["frac_at_places"] = len(cc), round(frac_at, 3)
        t_["agent"] = bool(t_["big_move"] > 0.5)
        if t_["agent"]:
            continue
        if t_["anchor"] == "location":
            anchored += [(c, k, r_) for c, r_ in zip(cc, rad)]
        elif t_["count"] <= 1:                                              # an identity is one thing
            colour_ids.append(k)
        else:
            t_["multi_instance_free"] = True                                # background, or same-colour objects: not resolved
    loc_ids = []
    # places within thr_pos of each other are one identity; its disc is the largest member place (the whole
    # object); where in the disc the state shows is learned below (pixel weights)
    for c, k, r_ in sorted(anchored, key=lambda x: -x[2]):
        for L_ in loc_ids:
            if np.hypot(*(L_["centre"] - c)) <= thr_pos:
                L_["types"].add(k); break
        else:
            loc_ids.append({"centre": c, "types": {k}, "radius": r_})
    idents = [{"anchor": "colour", "types": {k}, "centre": None, "colour": cent_rgb[k].astype(np.float32)} for k in colour_ids] + \
             [{"anchor": "location", "types": L_["types"], "centre": L_["centre"], "radius": L_["radius"]} for L_ in loc_ids]
    K = len(idents)
    agent_types = {t_["type"] for t_ in types if t_["agent"]}

    vv, uu_ = np.mgrid[0:64, 0:64]
    discs = {j: np.hypot(uu_ - idn["centre"][0], vv - idn["centre"][1]) <= max(idn["radius"], 1.0) for j, idn in enumerate(idents) if idn["centre"] is not None}
    # PIXEL WEIGHTS of a place: temporal variance of each disc pixel over train frames where the agent does not
    # cover it -- the state of a place shows where its pixels change (a light's dot, not its tile)
    pweights = {}
    if discs:
        byt_tr = {}
        for i, r in enumerate(tr):
            byt_tr.setdefault(r["t"], []).append(i)
        obs_w = np.load(a.cache / "train_observations.npy", mmap_mode="r")
        acc = {j: [np.zeros((D_.sum(), 3)), np.zeros((D_.sum(), 3)), np.zeros(D_.sum())] for j, D_ in discs.items()}
        for t in sorted(byt_tr)[:: max(1, len(byt_tr) // 2000)]:
            am = np.zeros((64, 64), bool)
            for i in byt_tr[t]:
                if lab[i] in agent_types:
                    am |= np.unpackbits(tr[i]["mask"], axis=-1)[:, :64].astype(bool)
            img = np.asarray(obs_w[t], np.float64) / 255.0
            for j, D_ in discs.items():
                px, ok = img[D_], ~am[D_]
                acc[j][0][ok] += px[ok]; acc[j][1][ok] += px[ok] ** 2; acc[j][2][ok] += 1
        for j, (s1, s2, c) in acc.items():
            c = np.maximum(c, 1)[:, None]
            w = np.clip(s2 / c - (s1 / c) ** 2, 0, None).sum(-1)
            pweights[j] = w if w.sum() > 0 else np.ones_like(w)

    def assign(rows, labels, split):
        n = meta[split]["n"]
        term_s = np.load(a.cache / f"{split}_terminals.npy"); ep_s = np.concatenate([[0], np.cumsum(term_s[:-1])])
        P_r = np.stack([r["pos"] for r in rows]); A_r = np.array([r["area"] for r in rows])
        last = [None] * K; last_ep = None
        pos = np.zeros((n, K, 2), np.float32); app = np.zeros((n, K, 3), np.float32); area = np.zeros((n, K), np.int16)
        agent = np.zeros((n, 64, 8), np.uint8); processed = np.zeros(n, bool)
        processed[meta[split]["frames"]] = True
        byt = {}
        for i, r in enumerate(rows):
            byt.setdefault(r["t"], []).append(i)
        obs_s = np.load(a.cache / f"{split}_observations.npy", mmap_mode="r")
        for t, ii in sorted(byt.items()):
            if ep_s[t] != last_ep:
                last, last_ep = [None] * K, ep_s[t]
            for i in ii:
                if labels[i] in agent_types:
                    agent[t] |= rows[i]["mask"]
            am = np.unpackbits(agent[t], axis=-1)[:, :64].astype(bool)
            img = None
            for j, idn in enumerate(idents):
                if idn["centre"] is None:
                    # colour-anchored: union of the identity's segments within thr_pos of the one picked by
                    # continuity (faces of one object); its colour is its identity -> constant appearance
                    cand = [i for i in ii if labels[i] in idn["types"]]
                    if not cand:
                        continue
                    i = pick(cand, P_r, A_r, last[j])
                    uu = [c for c in cand if np.hypot(*(rows[c]["pos"] - rows[i]["pos"])) <= thr_pos]
                    w = np.array([rows[c]["area"] for c in uu], np.float64)
                    pos[t, j] = (w[:, None] * np.stack([rows[c]["pos"] for c in uu])).sum(0) / w.sum()
                    app[t, j] = idn["colour"]; area[t, j] = int(w.sum()); last[j] = rows[i]["pos"]
                else:
                    # location-anchored: its place is its identity -> constant position; its state is the
                    # appearance of the place (mean colour of the place disc), unobserved while the agent covers it
                    D_, w = discs[j], pweights[j]
                    ok = ~am[D_]                                                     # uncovered pixels of the place
                    if w[ok].sum() * 2 <= w.sum():                                   # most of the weight covered: unobserved
                        continue
                    if img is None:
                        img = np.asarray(obs_s[t], np.float32) / 255.0
                    px = img[D_][ok]
                    pos[t, j] = idn["centre"]; app[t, j] = (w[ok][:, None] * px).sum(0) / w[ok].sum(); area[t, j] = int(ok.sum())
        return pos, app, area, agent, processed

    for split in ("train", "val"):
        rows = segs[split]
        if split == "train":
            labels = lab
        else:
            lr = label(np.stack([r["col"] for r in rows]).astype(np.float64)) if rows else np.zeros(0, int)
            labels = np.where(lr >= 0, rep[np.maximum(lr, 0)], -1)
        pos, app, area, agent, processed = assign(rows, labels, split)
        np.savez(a.out / f"entities_{split}.npz", pos=pos, app=app, area=area, agent=agent, processed=processed)
        np.savez(a.out / f"tracks_{split}.npz", uv=pos, mass=area)
    disc = {"objects": K, "source": "sam2_frames", "stride": a.stride, "groups": [[i] for i in range(K)],
            "table": [{"cluster": i, "spread_median": float(np.median([r["spread"] for r in tr]))} for i in range(K)],
            "types": types, "raw_clusters": raw, "flip_links": flips, "flip_merge": {"thr": thr_link, "D": D_link}, 
            "identities": [{"anchor": d["anchor"], "types": sorted(int(x) for x in d["types"]),
                            "centre": None if d["centre"] is None else np.round(d["centre"], 2).tolist(),
                            "radius": None if d["centre"] is None else round(d["radius"], 2)} for d in idents],
            "colour_merge": merge, "colour_merge_D": m_D, "thr_pos": thr_pos, "segments": {s: len(r) for s, r in segs.items()}}
    # PRIVILEGED diagnostic (cube envs): identity -> cube by centroid fit on val sampled frames
    diag = []
    qp = a.cache / "val_qpos.npy"
    if qp.exists() and "cube" in a.cache.name:
        q = np.load(qp, mmap_mode="r"); z = np.load(a.out / "entities_val.npz")
        fr = meta["val"]["frames"]
        slices = [s for s in (14, 21, 28, 35) if s + 3 <= q.shape[1]]
        xyz = np.stack([np.asarray(q[fr, s:s + 3]) for s in slices], 1)
        uvv, mv = z["pos"][fr], z["area"][fr]
        half = len(fr) // 2
        for k in range(K):
            vis = mv[:, k] >= 1
            best = None
            for j in range(len(slices)):
                f = np.c_[np.ones(len(fr)), uvv[:, k], uvv[:, k] ** 2]
                trm, tem = vis & (np.arange(len(fr)) < half), vis & (np.arange(len(fr)) >= half)
                if trm.sum() < 30 or tem.sum() < 30:
                    continue
                W = np.linalg.lstsq(f[trm], xyz[trm, j, :2], rcond=None)[0]
                err = float(np.median(np.linalg.norm(f[tem] @ W - xyz[tem, j, :2], axis=-1)) * 100)
                if best is None or err < best["median_err_cm"]:
                    best = {"object": k, "cube": j, "median_err_cm": round(err, 2), "visible_frac": round(float(vis.mean()), 3)}
            diag.append(best)
    disc["privileged_diagnostic"] = diag
    disc["minutes"] = round((time.time() - t0) / 60, 1)
    (a.out / "discover.json").write_text(json.dumps(disc, indent=1, default=float) + "\n")
    print(json.dumps({k: v for k, v in disc.items() if k not in ("table",)}, default=float), flush=True)


if __name__ == "__main__":
    main()
