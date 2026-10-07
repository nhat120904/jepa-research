#!/usr/bin/env python3
"""Unified front end: entities = SAM 2 segments tracked through the episode; one rule set for all tasks.

Per episode (first `T` frames):
  1. proposals on frame 0: SAM 2.1 point grid on the frame upsampled to 512 px; per point the FINEST mask
     with predicted IoU >= iou_min (so individual lights rather than the whole panel); keep masks covering
     at most `max_frac` of the image (drops floor, panel, wall); de-duplicate (mask IoU > 0.5);
  2. SAM 2 video propagation of every proposal through the T frames (identity kept by the tracker);
  3. per frame and entity: visible area, centroid (px), mean colour;
  4. agent entities = entities whose centroid moves in most frames (2-means on log moving fraction);
     the rest are object entities.
PRIVILEGED diagnostics only (pre-registered criteria of the unified state component):
  puzzle / scene buttons: best object entity per light, accuracy of reading its state from mean colour;
  cube: best object entity per cube, centroid error (px and cm via the cube projection), tracked fraction;
  events: object-entity state changes (centroid or colour jump; 2-means on log change) vs privileged events.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from sfa_code import two_means_threshold

MODEL = "facebook/sam2.1-hiera-small"


def proposals(img_model, proc, frame, res, grid, iou_min, max_frac, dev):
    import torch
    from PIL import Image

    img = Image.fromarray(frame).resize((res, res), Image.BICUBIC)
    g = (np.arange(grid) + 0.5) * res / grid
    pts = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
    cand, score = [], []
    for c in range(0, len(pts), 64):
        inp = proc(images=img, input_points=[[[[float(x), float(y)]] for x, y in pts[c:c + 64]]], return_tensors="pt").to(dev)
        with torch.no_grad():
            out = img_model(**inp, multimask_output=True)
        post = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0] > 0      # (P, 3, H, W)
        sc = out.iou_scores.cpu()[0]
        for p in range(post.shape[0]):
            ok = [m for m in range(3) if sc[p, m] >= iou_min and 4 <= post[p, m].sum() <= max_frac * res * res]
            if ok:
                m = min(ok, key=lambda m_: int(post[p, m_].sum()))                                    # finest valid
                cand.append(post[p, m].numpy()); score.append(float(sc[p, m]))
    keep = []
    for j in np.argsort(score)[::-1]:
        if all((cand[j] & cand[k]).sum() / max(1, (cand[j] | cand[k]).sum()) <= 0.5 for k in keep):
            keep.append(j)
    f = res // 64
    masks64 = []
    for j in keep:
        m = cand[j].reshape(64, f, 64, f)
        m64 = m.mean((1, 3)) > 0.5
        if not m64.any():
            m64 = m.any((1, 3))
        masks64.append(m64)
    return masks64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--grid", type=int, default=32)
    ap.add_argument("--iou-min", type=float, default=0.75)
    ap.add_argument("--max-frac", type=float, default=0.02)
    ap.add_argument("--project", type=Path, default=None, help="projection.npz (PRIVILEGED cube check)")
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
    vo = np.load(a.cache / "val_observations.npy", mmap_mode="r")
    vt = np.load(a.cache / "val_terminals.npy")
    starts = np.r_[0, np.nonzero(vt)[0] + 1][: a.episodes]
    bs = np.load(a.cache / "val_button_states.npy", mmap_mode="r") if (a.cache / "val_button_states.npy").exists() else None
    q = np.load(a.cache / "val_qpos.npy", mmap_mode="r") if (a.cache / "val_qpos.npy").exists() else None
    W = None
    if a.project is not None:
        from cube_projection import feats
        W = np.load(a.project)["W"]
    v, u = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    per_ep, light_acc, cube_err, cube_track, ev_rec, ev_prec, n_ent, n_obj, viz = [], [], [], [], [], [], [], [], []
    for e, s0 in enumerate(starts):
        frames = np.asarray(vo[s0:s0 + a.T])
        T = len(frames)
        masks0 = proposals(img_model, proc, frames[0], a.res, a.grid, a.iou_min, a.max_frac, dev)
        N = len(masks0)
        sess = vproc.init_video_session(video=list(frames), inference_device=dev, video_storage_device="cpu", dtype=torch.bfloat16)
        vproc.add_inputs_to_inference_session(sess, frame_idx=0, obj_ids=list(range(1, N + 1)), input_masks=[m.astype(np.float32) for m in masks0])
        M = np.zeros((T, N, 64, 64), bool)
        with torch.no_grad():
            for out in vmodel.propagate_in_video_iterator(sess, start_frame_idx=0):
                pm = vproc.post_process_masks([out.pred_masks.float()], original_sizes=[[64, 64]], binarize=True)[0]
                M[out.frame_idx] = pm.reshape(N, -1, 64, 64)[:, 0].cpu().numpy()
        area = M.sum((2, 3)).astype(np.float32)                                  # (T, N)
        cen = np.stack([(M * u).sum((2, 3)), (M * v).sum((2, 3))], -1) / np.maximum(area, 1)[..., None]
        col = np.einsum("tnhw,thwc->tnc", M.astype(np.float32), frames.astype(np.float32)) / np.maximum(area, 1)[..., None]
        vis = area >= 1
        both = vis[1:] & vis[:-1]
        step = np.linalg.norm(cen[1:] - cen[:-1], axis=-1)
        movf = np.array([(step[both[:, k], k] > 0.5).mean() if both[:, k].any() else 1.0 for k in range(N)])
        thr, sep, _ = two_means_threshold(np.log(movf + 1e-3)) if N > 2 else (np.log(0.5), 0.0, 0.0)
        obj = np.log(movf + 1e-3) <= thr
        n_ent.append(N); n_obj.append(int(obj.sum()))
        rec = {"episode": int(e), "entities": N, "object_entities": int(obj.sum()), "moving_frac": np.round(movf, 3).tolist()}
        if bs is not None:
            b = np.asarray(bs[s0:s0 + T]).astype(int)
            accs = []
            for i in range(b.shape[1]):
                best = 0.5
                for k in np.nonzero(obj)[0]:
                    ok = vis[:, k]
                    if ok.sum() < 20 or b[ok, i].min() == b[ok, i].max():
                        continue
                    for ch in range(3):                                           # best colour channel + threshold
                        x = col[ok, k, ch]
                        t_, _, _ = two_means_threshold(x)
                        acc = max(((x > t_) == b[ok, i]).mean(), ((x <= t_) == b[ok, i]).mean())
                        best = max(best, acc)
                accs.append(best)
            rec["light_acc"] = np.round(accs, 3).tolist(); light_acc.append(accs)
        if q is not None and W is not None:
            xyz = np.stack([np.asarray(q[s0:s0 + T, s:s + 3]) for s in (14, 21, 28) if s + 3 <= q.shape[1]], 1)
            uv = (feats(xyz.reshape(-1, 3)) @ W).reshape(T, -1, 2)
            errs, tr = [], []
            for c in range(uv.shape[1]):
                best = (np.inf, 0.0)
                for k in np.nonzero(obj)[0]:
                    ok = vis[:, k]
                    if ok.sum() < 20:
                        continue
                    d = np.linalg.norm(cen[ok, k] - uv[ok, c], axis=-1)
                    if np.median(d) < best[0]:
                        best = (float(np.median(d)), float((np.linalg.norm(cen[:, k] - uv[:, c], axis=-1)[vis[:, k]] < 2).sum() / T))
                errs.append(best[0]); tr.append(best[1])
            rec["cube_err_px"] = np.round(errs, 2).tolist(); rec["cube_tracked_frac"] = np.round(tr, 3).tolist()
            cube_err.append(errs); cube_track.append(tr)
        # events from object-entity state changes
        if obj.any():
            dc = np.zeros(T); dcol = np.zeros(T)
            o = np.nonzero(obj)[0]
            stepo = np.where(both[:, o], step[:, o], 0.0)
            colo = np.where(both[:, o][..., None], np.abs(col[1:, o] - col[:-1, o]), 0.0).max(-1)
            dc[1:] = stepo.max(1); dcol[1:] = colo.max(1) / 32.0
            ch_ = np.maximum(dc, dcol)
            nz = ch_[ch_ > 0]
            if len(nz) > 10:
                cthr, _, _ = two_means_threshold(np.log(nz))
                on = np.log(np.maximum(ch_, 1e-6)) > cthr
                ev = []
                t = 1
                while t < T:
                    if not on[t]:
                        t += 1; continue
                    s_ = e_ = t
                    while t < T and (on[t] or t - e_ <= 5):
                        if on[t]:
                            e_ = t
                        t += 1
                    ev.append(s_)
                ev = np.array(ev)
                ref = None
                if bs is not None and q is None:
                    b = np.asarray(bs[s0:s0 + T]); ref = (b[1:] != b[:-1]).any(1)
                elif q is not None:
                    qq = np.asarray(q[s0:s0 + T]); ref = np.abs(np.diff(qq[:, 14:], axis=0)).max(1) > 2e-3
                    if bs is not None:
                        b = np.asarray(bs[s0:s0 + T]); ref |= (b[1:] != b[:-1]).any(1)
                if ref is not None and len(ev):
                    d_ = np.diff(np.r_[0, ref.astype(np.int8), 0]); st_, en_ = np.nonzero(d_ == 1)[0] + 1, np.nonzero(d_ == -1)[0]
                    if len(st_):
                        cand = np.r_[st_, en_]
                        ev_rec.append(float(np.mean([(np.abs(ev - x).min() <= 15) or (np.abs(ev - y).min() <= 15) for x, y in zip(st_, en_)])))
                        ev_prec.append(float(np.mean([np.abs(cand - s_).min() <= 15 for s_ in ev])))
                        rec["events"] = int(len(ev)); rec["ref_events"] = int(len(st_))
        per_ep.append(rec)
        if e < 3:
            pal = np.random.default_rng(1).integers(40, 255, (N, 3))
            for t in (0, T // 3, 2 * T // 3, T - 1):
                img = frames[t].astype(np.float32).copy()
                for k in range(N):
                    img[M[t, k]] = 0.4 * img[M[t, k]] + 0.6 * (pal[k] if obj[k] else np.array([255, 255, 255]))
                viz.append(np.concatenate([frames[t], img.astype(np.uint8)], 1))
        print(json.dumps({k: rec[k] for k in rec if k != "moving_frac"}), round((time.time() - t0) / 60, 1), "min", flush=True)
    res = {"env": a.cache.name, "episodes": len(per_ep), "T": a.T, "entities_mean": float(np.mean(n_ent)), "object_entities_mean": float(np.mean(n_obj))}
    if light_acc:
        la = np.array(light_acc)
        res["criterion1_light_acc_min_mean"] = [float(la.mean(0).min()), float(la.mean())]
    if cube_err:
        ce = np.array(cube_err)
        res["criterion2_cube_err_px_median"] = np.round(np.median(ce, 0), 2).tolist()
        res["cube_tracked_frac_mean"] = np.round(np.mean(cube_track, 0), 3).tolist()
    if ev_rec:
        res["criterion4_event_recall"] = float(np.mean(ev_rec)); res["criterion4_event_precision"] = float(np.mean(ev_prec))
    res["minutes"] = round((time.time() - t0) / 60, 1)
    res["per_episode"] = per_ep
    (a.out / f"entities_{a.cache.name}.json").write_text(json.dumps(res, indent=1) + "\n")
    from PIL import Image
    if viz:
        img = np.concatenate(viz, 0)
        Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(a.out / f"entities_{a.cache.name}.png")
    print(json.dumps({k: v_ for k, v_ in res.items() if k != "per_episode"}), flush=True)


if __name__ == "__main__":
    main()
