#!/usr/bin/env python3
"""Unified front end at scale: SAM 2 entity tracks for many episodes -> per-identity tracks in the format
of cube_discover.py (uv, visible mass), so the existing event / reader / planner / skill stages can run.

Per episode (whole episode): proposals + video propagation as in sam2_entities.py; agent entities removed
by the slowness rule; per object entity: centroid, visible area, median colour over the episode.
Identity across episodes, one rule for every task family:
  1. TYPES: agglomerative clustering of the entities' median colours; two entities of one episode are
     never the same entity, and the merge threshold is 3 x the median within-track colour deviation;
  2. INSTANCES: a type with several entities per episode (identical lights, two buttons) is split by
     position (k-means on mean position, k = median number of entities of that type per episode).
So a cube is identified by its colour and a light by its location, with no domain switch.
Writes <out>/entities_{train,val}.npz -- the generic entity table of the unified backend: per frame and
identity pos (px), app (mean RGB / 255), area (visible px); agent = packed union mask of the agent
entities per frame; processed = frames of tracked episodes -- plus tracks_{split}.npz (uv, mass; the
cube_discover.py format) and <out>/discover.json (identities, spreads; PRIVILEGED identity -> cube matching).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from sam2_entities import MODEL, proposals
from sfa_code import two_means_threshold


def track_episode(frames, img_model, proc, vproc, vmodel, a, dev):
    import torch

    T = len(frames)
    masks0 = proposals(img_model, proc, frames[0], a.res, a.grid, a.iou_min, a.max_frac, dev)
    N = len(masks0)
    if N == 0:
        return None
    sess = vproc.init_video_session(video=list(frames), inference_device=dev, video_storage_device="cpu", dtype=torch.bfloat16)
    vproc.add_inputs_to_inference_session(sess, frame_idx=0, obj_ids=list(range(1, N + 1)), input_masks=[m.astype(np.float32) for m in masks0])
    v, u = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    area = np.zeros((T, N), np.float32); cen = np.zeros((T, N, 2), np.float32); col = np.zeros((T, N, 3), np.float32)
    spread = np.zeros((T, N), np.float32)
    packed = np.zeros((T, N, 64, 8), np.uint8)
    with torch.no_grad():
        for out in vmodel.propagate_in_video_iterator(sess, start_frame_idx=0):
            pm = vproc.post_process_masks([out.pred_masks.float()], original_sizes=[[64, 64]], binarize=True)[0]
            M = pm.reshape(N, -1, 64, 64)[:, 0].cpu().numpy()
            t = out.frame_idx
            packed[t] = np.packbits(M, axis=-1)
            ar = M.sum((1, 2)).astype(np.float32)
            area[t] = ar
            cen[t] = np.stack([(M * u).sum((1, 2)), (M * v).sum((1, 2))], -1) / np.maximum(ar, 1)[:, None]
            col[t] = np.einsum("nhw,hwc->nc", M.astype(np.float32), frames[t].astype(np.float32)) / np.maximum(ar, 1)[:, None]
            var = (M * ((u[None] - cen[t, :, 0, None, None]) ** 2 + (v[None] - cen[t, :, 1, None, None]) ** 2)).sum((1, 2)) / np.maximum(ar, 1)
            spread[t] = np.sqrt(var)
    vis = area >= 1
    both = vis[1:] & vis[:-1]
    step = np.linalg.norm(cen[1:] - cen[:-1], axis=-1)
    movf = np.array([(step[both[:, k], k] > 0.5).mean() if both[:, k].any() else 1.0 for k in range(N)])
    return {"area": area, "cen": cen, "col": col, "spread": spread, "movf": movf, "vis": vis, "packed": packed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--train-episodes", type=int, default=150)
    ap.add_argument("--val-episodes", type=int, default=30)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--iou-min", type=float, default=0.75)
    ap.add_argument("--max-frac", type=float, default=0.02)
    ap.add_argument("--kind", default="triple")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor

    dev = "cuda"
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    proc = Sam2Processor.from_pretrained(MODEL)
    img_model = Sam2Model.from_pretrained(MODEL).to(dev).eval()
    vproc = Sam2VideoProcessor.from_pretrained(MODEL)
    vmodel = Sam2VideoModel.from_pretrained(MODEL).to(dev, dtype=torch.bfloat16).eval()
    raw = {}
    for split, n_ep in (("train", a.train_episodes), ("val", a.val_episodes)):
        obs = np.load(a.cache / f"{split}_observations.npy", mmap_mode="r")
        term = np.load(a.cache / f"{split}_terminals.npy")
        starts = np.r_[0, np.nonzero(term)[0] + 1]; ends = np.r_[np.nonzero(term)[0] + 1, len(term)]
        raw[split] = []
        for e in range(min(n_ep, len(starts))):
            r = track_episode(np.asarray(obs[starts[e]:ends[e]]), img_model, proc, vproc, vmodel, a, dev)
            raw[split].append((starts[e], ends[e], r))
            if e % 10 == 0:
                print(split, e, None if r is None else len(r["movf"]), round((time.time() - t0) / 60, 1), "min", flush=True)
    # agent entities: slowness rule over all entities of all episodes
    allm = np.concatenate([r["movf"] for sp in raw.values() for _, _, r in sp if r is not None])
    thr, sep, _ = two_means_threshold(np.log(allm + 1e-3))
    # object entities -> descriptors
    ents = []
    for sp, lst in raw.items():
        for ei, (s0, s1, r) in enumerate(lst):
            if r is None:
                continue
            for k in range(len(r["movf"])):
                if np.log(r["movf"][k] + 1e-3) > thr or r["vis"][:, k].sum() < 20:
                    continue
                v_ = r["vis"][:, k]
                ents.append({"split": sp, "ep": ei, "k": k, "col": np.median(r["col"][v_, k], 0),
                             "col_dev": float(np.median(np.linalg.norm(r["col"][v_, k] - np.median(r["col"][v_, k], 0), axis=-1))),
                             "pos": np.median(r["cen"][v_, k], 0)})
    C = np.stack([e_["col"] for e_ in ents]); epkey = np.array([hash((e_["split"], e_["ep"])) for e_ in ents])
    merge = 3.0 * float(np.median([e_["col_dev"] for e_ in ents]))
    # TYPES: greedy agglomerative clustering on colour with a cannot-link inside an episode
    labels = -np.ones(len(ents), int); centres = []
    for i in np.argsort([-np.linalg.norm(c_) for c_ in C]):
        best, bd = -1, merge
        for t_, (cc, _) in enumerate(centres):
            d = np.linalg.norm(C[i] - cc / max(1, _))
            if d < bd:
                best, bd = t_, d
        if best < 0:
            centres.append([C[i].copy(), 1]); labels[i] = len(centres) - 1
        else:
            centres[best][0] += C[i]; centres[best][1] += 1; labels[i] = best
    # INSTANCES: split types with several entities per episode by position
    ident = -np.ones(len(ents), int); idinfo = []
    for t_ in range(len(centres)):
        sel = np.nonzero(labels == t_)[0]
        if len(sel) == 0:
            continue
        per_ep = np.bincount(np.unique(epkey[sel], return_inverse=True)[1])
        kk = int(np.median(per_ep))
        if kk <= 1:
            ident[sel] = len(idinfo); idinfo.append({"type": int(t_), "instance": 0, "n": int(len(sel))})
            continue
        P = np.stack([ents[i]["pos"] for i in sel])
        cc = P[np.random.default_rng(0).choice(len(P), kk, replace=False)]
        for _ in range(50):
            lab = np.linalg.norm(P[:, None] - cc[None], axis=-1).argmin(1)
            cc = np.stack([P[lab == j].mean(0) if (lab == j).any() else cc[j] for j in range(kk)])
        for j in range(kk):
            ident[sel[lab == j]] = len(idinfo); idinfo.append({"type": int(t_), "instance": int(j), "n": int((lab == j).sum())})
    # keep identities present in at least half of the episodes (rare ones are segmentation noise)
    n_eps = len({(e_["split"], e_["ep"]) for e_ in ents})
    keep_ids = [i for i, d in enumerate(idinfo) if d["n"] >= 0.5 * n_eps]
    remap = {old: new for new, old in enumerate(keep_ids)}
    K = len(keep_ids)
    for split, lst in raw.items():
        obs_n = len(np.load(a.cache / f"{split}_terminals.npy"))
        uv = np.zeros((obs_n, K, 2), np.float32); mass = np.zeros((obs_n, K), np.int16)
        for i, e_ in enumerate(ents):
            if e_["split"] != split or ident[i] not in remap:
                continue
            s0, s1, r = lst[e_["ep"]]
            j = remap[ident[i]]
            uv[s0:s1, j] = r["cen"][:, e_["k"]]; mass[s0:s1, j] = r["area"][:, e_["k"]].astype(np.int16)
        np.savez(a.out / f"tracks_{split}.npz", uv=uv, mass=mass)
        app = np.zeros((obs_n, K, 3), np.float32); agent = np.zeros((obs_n, 64, 8), np.uint8); processed = np.zeros(obs_n, bool)
        for i, e_ in enumerate(ents):
            if e_["split"] != split or ident[i] not in remap:
                continue
            s0, s1, r = lst[e_["ep"]]
            app[s0:s1, remap[ident[i]]] = r["col"][:, e_["k"]] / 255.0
        for s0, s1, r in lst:
            if r is None:
                continue
            processed[s0:s1] = True
            ag = np.log(r["movf"] + 1e-3) > thr
            if ag.any():
                agent[s0:s1] = np.bitwise_or.reduce(r["packed"][:, ag], axis=1)
        np.savez(a.out / f"entities_{split}.npz", pos=uv, app=app, area=mass, agent=agent, processed=processed)
    spreads = []
    for i, e_ in enumerate(ents):
        if ident[i] in remap:
            r = raw[e_["split"]][e_["ep"]][2]
            spreads.append(float(np.median(r["spread"][r["vis"][:, e_["k"]], e_["k"]])))
    disc = {"objects": K, "source": "sam2_tracks", "groups": [[i] for i in range(K)],
            "table": [{"cluster": i, "spread_median": float(np.median(spreads))} for i in range(K)],
            "identities": [idinfo[i] for i in keep_ids], "agent_thr_moving_frac": float(np.exp(thr)), "colour_merge": merge,
            "processed_episodes": {sp: len(lst) for sp, lst in raw.items()}}
    # PRIVILEGED diagnostic: identity -> cube matching on val (cube domains only)
    qp = a.cache / "val_qpos.npy"
    diag = []
    if qp.exists() and "cube" in a.cache.name:
        q = np.load(qp, mmap_mode="r")
        z = np.load(a.out / "tracks_val.npz")
        nv = sum(s1 - s0 for s0, s1, _ in raw["val"])
        slices = [s for s in (14, 21, 28, 35) if s + 3 <= q.shape[1]]
        xyz = np.stack([np.asarray(q[:nv, s:s + 3]) for s in slices], 1)
        uvv, mv = z["uv"][:nv], z["mass"][:nv]
        half = nv // 2
        for k in range(K):
            vis = mv[:, k] >= 1
            best = None
            for j in range(len(slices)):
                f = np.c_[np.ones(nv), uvv[:, k], uvv[:, k] ** 2]
                tr, te = vis & (np.arange(nv) < half), vis & (np.arange(nv) >= half)
                if tr.sum() < 50 or te.sum() < 50:
                    continue
                Wq = np.linalg.lstsq(f[tr], xyz[tr, j, :2], rcond=None)[0]
                err = float(np.median(np.linalg.norm(f[te] @ Wq - xyz[te, j, :2], axis=-1)) * 100)
                if best is None or err < best["median_err_cm"]:
                    best = {"object": k, "cube": j, "median_err_cm": err, "visible_frac": float(vis.mean())}
            diag.append(best)
    disc["privileged_diagnostic"] = diag
    disc["minutes"] = round((time.time() - t0) / 60, 1)
    (a.out / "discover.json").write_text(json.dumps(disc, indent=1) + "\n")
    print(json.dumps({k: v for k, v in disc.items() if k not in ("table",)}), flush=True)


if __name__ == "__main__":
    main()
