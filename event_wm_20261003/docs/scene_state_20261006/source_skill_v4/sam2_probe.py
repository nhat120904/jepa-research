#!/usr/bin/env python3
"""Front-end fallback probe: does SAM 2 segment the small things of every task family?

Per frame (64x64 upsampled to `res`): a grid of point prompts -> SAM 2.1 masks (best of 3 per point by
predicted IoU) -> keep predicted IoU >= `iou_min`, drop near-duplicates (mask IoU > 0.7, higher score
kept) and masks larger than half the image -> segments. Saves overlays and segment statistics; for
cube frames also reports (PRIVILEGED, via the fitted qpos->pixel projection of the cube A run) whether
each cube centre falls inside some small segment.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

MODEL = "facebook/sam2.1-hiera-small"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--grid", type=int, default=24)
    ap.add_argument("--iou-min", type=float, default=0.80)
    ap.add_argument("--project", type=Path, default=None, help="projection.npz (PRIVILEGED cube check)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor

    dev = "cuda"
    a.out.mkdir(parents=True, exist_ok=True)
    proc = Sam2Processor.from_pretrained(MODEL)
    model = Sam2Model.from_pretrained(MODEL).to(dev).eval()
    vo = np.load(a.cache / "val_observations.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = np.sort(rng.integers(0, len(vo), a.frames))
    g = (np.arange(a.grid) + 0.5) * a.res / a.grid
    pts = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)                         # (P, 2) x, y
    stats, overlays, cube_hits = [], [], []
    if a.project is not None and (a.cache / "val_qpos.npy").exists():
        from cube_projection import feats
        zp = np.load(a.project); W = zp["W"]
        q = np.load(a.cache / "val_qpos.npy", mmap_mode="r")
    for n_, i in enumerate(idx):
        img = Image.fromarray(np.asarray(vo[i])).resize((a.res, a.res), Image.BICUBIC)
        masks, scores = [], []
        for c in range(0, len(pts), 64):
            inp = proc(images=img, input_points=[[[[float(x), float(y)]] for x, y in pts[c:c + 64]]], return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model(**inp, multimask_output=True)
            post = proc.post_process_masks(out.pred_masks.cpu(), inp["original_sizes"].cpu())[0]   # (P, 3, H, W)
            sc = out.iou_scores.cpu()[0]                                                       # (P, 3)
            best = sc.argmax(-1)
            for p in range(len(best)):
                masks.append(post[p, best[p]].numpy() > 0); scores.append(float(sc[p, best[p]]))
        order = np.argsort(scores)[::-1]
        keep = []
        for j in order:
            mk = masks[j]
            if scores[j] < a.iou_min or mk.sum() < 4 or mk.mean() > 0.5:
                continue
            if any((mk & masks[k]).sum() / max(1, (mk | masks[k]).sum()) > 0.7 for k in keep):
                continue
            keep.append(j)
        areas = [int(masks[j].sum()) for j in keep]
        stats.append({"frame": int(i), "segments": len(keep), "area_px_pct": np.percentile(areas, [10, 50, 90]).tolist() if areas else None})
        if a.project is not None and (a.cache / "val_qpos.npy").exists():
            xyz = np.stack([np.asarray(q[i, s:s + 3]) for s in (14, 21, 28) if s + 3 <= q.shape[1]])
            uv = feats(xyz) @ W * a.res / 64.0
            hits = []
            for u, v in uv:
                ui, vi = int(np.clip(u, 0, a.res - 1)), int(np.clip(v, 0, a.res - 1))
                hits.append(bool(any(masks[j][vi, ui] and masks[j].mean() < 0.02 for j in keep)))
            cube_hits.append(hits)
        if n_ < 6:
            base = np.asarray(img).astype(np.float32)
            col = np.zeros_like(base)
            for k, j in enumerate(keep):
                col[masks[j]] = rng.integers(40, 255, 3)
            overlays.append(np.concatenate([base, 0.45 * base + 0.55 * col], 1).astype(np.uint8))
    res = {"env": a.cache.name, "model": MODEL, "res": a.res, "grid": a.grid, "iou_min": a.iou_min,
           "segments_pct": np.percentile([s["segments"] for s in stats], [10, 50, 90]).tolist(), "frames": stats}
    if cube_hits:
        h = np.array(cube_hits)
        res["privileged_cube_in_small_segment_frac"] = h.mean(0).round(3).tolist()
    (a.out / f"sam2_{a.cache.name}.json").write_text(json.dumps(res, indent=1) + "\n")
    Image.fromarray(np.concatenate(overlays, 0)).resize((a.res, a.res * len(overlays) // 2)).save(a.out / f"sam2_{a.cache.name}.png")
    print(json.dumps({k: v for k, v in res.items() if k != "frames"}), flush=True)


if __name__ == "__main__":
    main()
