#!/usr/bin/env python3
"""Cube direction A, step 1: label-free object discovery -- objects are what events move.

No simulator state is an input. With the fixed camera of the play data:
  1. background = per-pixel median over sampled frames; foreground = pixels whose colour distance
     to it exceeds a 2-means threshold on log distance;
  2. appearance clusters = k-means on foreground pixel colours (overcomplete, C clusters);
  3. per cluster and frame: pixel mass, centroid (u, v) and spatial spread (px);
  4. objects = clusters that are compact (low spread) and slow (their centroid moves in few frames;
     the arm, gripper and shadow move almost always) -- 2-means on each score, as for the puzzle code;
  5. object clusters whose centroids coincide (faces of one object) are merged;
  6. per frame and object: robust centroid (u, v) and visible mass -> tracks for every split.
The privileged cube positions (qpos) are used only for a reported diagnostic: per-object fit of
(u, v) -> table xy on held-out frames, and which cube each discovered object matches.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from sfa_code import two_means_threshold

SLICES = {"single": [14], "double": [14, 21], "triple": [14, 21, 28], "quadruple": [14, 21, 28, 35]}


def kmeans(x, C, iters=60, seed=0):
    import torch

    g = torch.Generator(device=x.device).manual_seed(seed)
    c = x[torch.randint(len(x), (1,), generator=g, device=x.device)]
    for _ in range(1, C):                                                          # k-means++ init
        dmin = torch.cdist(x, c).min(1).values ** 2
        c = torch.cat([c, x[torch.multinomial(dmin / dmin.sum(), 1, generator=g)]])
    for _ in range(iters):
        lab = torch.cdist(x, c).argmin(1)
        c = torch.stack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(len(c))])
    return c


class Perception:
    """Deterministic label-free perception: background + colour clusters + object groups."""

    def __init__(self, bg, tau, centres, groups, device):
        import torch

        self.bg = torch.as_tensor(bg, device=device).float()
        self.tau, self.groups = float(tau), [list(g) for g in groups]
        self.centres = torch.as_tensor(centres, device=device).float()
        v, u = torch.meshgrid(torch.arange(64, device=device).float(), torch.arange(64, device=device).float(), indexing="ij")
        self.u, self.v = u.flatten(), v.flatten()

    @classmethod
    def load(cls, path, device):
        z = np.load(path, allow_pickle=True)
        return cls(z["bg"], z["tau"], z["centres"], z["groups"], device)

    def labels(self, frames):
        """frames uint8 (B, 64, 64, 3) tensor -> cluster label per pixel (B, 4096), -1 = background."""
        x = frames.float()
        fg = (x - self.bg).norm(dim=-1).flatten(1) > self.tau
        lab = ((x.flatten(1, 2)[:, :, None] / 255.0 - self.centres[None, None]) ** 2).sum(-1).argmin(-1)
        return lab.masked_fill(~fg, -1)

    def cluster_stats(self, lab):
        """Per cluster: mass (B, C), centroid (B, C, 2), spread (B, C)."""
        import torch

        oh = torch.nn.functional.one_hot((lab + 1).long(), len(self.centres) + 1)[..., 1:].float()   # (B, 4096, C)
        m = oh.sum(1)
        mu = torch.stack([(oh * self.u[None, :, None]).sum(1), (oh * self.v[None, :, None]).sum(1)], -1) / m.clamp(min=1)[..., None]
        sq = (oh * (self.u ** 2 + self.v ** 2)[None, :, None]).sum(1) / m.clamp(min=1)
        spread = (sq - (mu ** 2).sum(-1)).clamp(min=0).sqrt()
        return m, mu, spread

    def objects(self, frames, radius=3.0):
        """Robust per-object centroid (B, K, 2) in px and visible mass (B, K)."""
        import torch

        lab = self.labels(frames)
        uv, mass = [], []
        for g in self.groups:
            w = torch.zeros_like(lab, dtype=torch.float32)
            for c in g:
                w += (lab == c).float()
            m = w.sum(1)
            mu = torch.stack([(w * self.u).sum(1), (w * self.v).sum(1)], -1) / m.clamp(min=1)[:, None]
            near = ((self.u[None] - mu[:, :1]) ** 2 + (self.v[None] - mu[:, 1:]) ** 2) <= radius ** 2
            w2 = w * near                                                              # trimmed second pass
            m2 = w2.sum(1)
            mu2 = torch.stack([(w2 * self.u).sum(1), (w2 * self.v).sum(1)], -1) / m2.clamp(min=1)[:, None]
            uv.append(torch.where(m2[:, None] > 0, mu2, mu)); mass.append(m)
        return torch.stack(uv, 1), torch.stack(mass, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--kind", default="triple")
    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--bg-frames", type=int, default=5000)
    ap.add_argument("--stat-frames", type=int, default=100_000, help="contiguous train frames for cluster scores")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    obs = {s: np.load(a.cache / f"{s}_observations.npy", mmap_mode="r") for s in ("train", "val")}
    term = {s: np.load(a.cache / f"{s}_terminals.npy") for s in ("train", "val")}

    # 1. background and foreground threshold
    idx = np.sort(rng.choice(len(obs["train"]), a.bg_frames, replace=False))
    S = torch.as_tensor(np.asarray(obs["train"][idx]), device=dev).float()
    bg = S.median(0).values
    d = (S - bg).norm(dim=-1).flatten()
    samp = d[torch.randperm(len(d), device=dev)[:2_000_000]].cpu().numpy()
    thr, sep, frac = two_means_threshold(np.log1p(samp))
    tau = float(np.expm1(thr))
    fgpx = S.flatten(0, 2)[(S - bg).norm(dim=-1).flatten() > tau] / 255.0
    print({"tau": round(tau, 1), "sep": round(sep, 2), "fg_frac": round(frac, 4), "fg_px": len(fgpx)}, flush=True)

    # 2. colour clusters
    fgpx = fgpx[torch.randperm(len(fgpx), device=dev)[:400_000]]
    centres = kmeans(fgpx, a.clusters)
    P = Perception(bg.cpu().numpy(), tau, centres.cpu().numpy(), [], dev)

    # 3. cluster scores on a contiguous window
    n = a.stat_frames
    ep = np.concatenate([[0], np.cumsum(term["train"][:n - 1])])
    M, MU, SP = [], [], []
    for s in range(0, n, a.chunk):
        fr = torch.as_tensor(np.asarray(obs["train"][s:min(n, s + a.chunk)]), device=dev)
        m, mu, sp = P.cluster_stats(P.labels(fr))
        M.append(m.cpu().numpy()); MU.append(mu.cpu().numpy()); SP.append(sp.cpu().numpy())
    M, MU, SP = np.concatenate(M), np.concatenate(MU), np.concatenate(SP)
    present = M >= 3
    both = present[1:] & present[:-1] & (ep[1:] == ep[:-1])[:, None]
    step = np.linalg.norm(MU[1:] - MU[:-1], axis=-1)
    table = []
    for c in range(a.clusters):
        pc = present[:, c]
        table.append({"cluster": c, "rgb": [int(x) for x in (centres[c].cpu().numpy() * 255).round()],
                      "present": float(pc.mean()), "mass_median": float(np.median(M[pc, c])) if pc.any() else 0.0,
                      "spread_median": float(np.median(SP[pc, c])) if pc.any() else 99.0,
                      "moving_frac": float((step[both[:, c], c] > 0.5).mean()) if both[:, c].any() else 1.0,
                      "pos_std": float(MU[pc, c].std(0).mean()) if pc.any() else 0.0})
    # 4. object clusters: compact and slow; ignore clusters that are almost never present
    cand = [t for t in table if t["present"] > 0.05]
    sthr, ssep, _ = two_means_threshold(np.log([t["spread_median"] + 1e-3 for t in cand]))
    mthr, msep, _ = two_means_threshold(np.log([t["moving_frac"] + 1e-3 for t in cand]))
    for t in table:
        t["compact"] = bool(t["present"] > 0.05 and np.log(t["spread_median"] + 1e-3) <= sthr)
        t["slow"] = bool(t["present"] > 0.05 and np.log(t["moving_frac"] + 1e-3) <= mthr)
    obj = [t["cluster"] for t in table if t["compact"] and t["slow"]]

    # 5. merge clusters with coinciding centroids
    parent = {c: c for c in obj}

    def find(c):
        while parent[c] != c:
            c = parent[c]
        return c

    merges = []
    for i, ci in enumerate(obj):
        for cj in obj[i + 1:]:
            co = present[:, ci] & present[:, cj]
            if co.sum() > 100:
                dist = float(np.median(np.linalg.norm(MU[co, ci] - MU[co, cj], axis=-1)))
                merges.append({"pair": [ci, cj], "co_frames": int(co.sum()), "median_dist_px": round(dist, 2)})
                if dist < 2.0:
                    parent[find(cj)] = find(ci)
    groups = {}
    for c in obj:
        groups.setdefault(find(c), []).append(c)
    groups = list(groups.values())
    print(json.dumps({"table": table, "spread_thr": float(np.exp(sthr)), "spread_sep": ssep, "moving_thr": float(np.exp(mthr)),
                      "moving_sep": msep, "object_clusters": obj, "merges": merges, "groups": groups}, indent=None), flush=True)
    np.savez(a.out / "perception.npz", bg=bg.cpu().numpy(), tau=tau, centres=centres.cpu().numpy(),
             groups=np.array(groups, dtype=object))
    P = Perception(bg.cpu().numpy(), tau, centres.cpu().numpy(), groups, dev)

    # 6. tracks for every split
    K = len(groups)
    rep = {}
    for split in ("train", "val"):
        N = len(obs[split])
        uv = np.zeros((N, K, 2), np.float32); mass = np.zeros((N, K), np.int16)
        for s in range(0, N, a.chunk):
            fr = torch.as_tensor(np.asarray(obs[split][s:s + a.chunk]), device=dev)
            u_, m_ = P.objects(fr)
            uv[s:s + len(fr)] = u_.cpu().numpy(); mass[s:s + len(fr)] = m_.cpu().numpy()
        np.savez(a.out / f"tracks_{split}.npz", uv=uv, mass=mass)
        rep[split] = {"frames": N, "visible_frac": (mass >= 3).mean(0).round(4).tolist(),
                      "mass_median": np.median(mass, 0).tolist()}
        print(split, rep[split], round((time.time() - t0) / 60, 1), "min", flush=True)

    # diagnostic only (PRIVILEGED qpos): match objects to cubes, held-out quadratic fit (u, v) -> xy
    q = np.load(a.cache / "val_qpos.npy", mmap_mode="r")
    xyz = np.stack([np.asarray(q[:, s:s + 3]) for s in SLICES[a.kind]], 1)
    z = np.load(a.out / "tracks_val.npz")
    uv, mass = z["uv"], z["mass"]
    half = len(uv) // 2
    diag = []
    for k in range(K):
        vis = mass[:, k] >= 3
        best = None
        for j in range(xyz.shape[1]):
            f = np.c_[np.ones(len(uv)), uv[:, k], uv[:, k] ** 2, uv[:, k, :1] * uv[:, k, 1:]]
            tr, te = vis & (np.arange(len(uv)) < half), vis & (np.arange(len(uv)) >= half)
            if tr.sum() < 100 or te.sum() < 100:
                continue
            W = np.linalg.lstsq(f[tr], xyz[tr, j, :2], rcond=None)[0]
            err = np.linalg.norm(f[te] @ W - xyz[te, j, :2], axis=-1)
            if best is None or np.median(err) < best["median_err_cm"] / 100:
                best = {"object": k, "cube": j, "median_err_cm": float(np.median(err) * 100),
                        "p90_err_cm": float(np.percentile(err, 90) * 100), "within_2cm": float((err < 0.02).mean()),
                        "visible_frac": float(vis.mean())}
        diag.append(best)
    out = {"objects": K, "groups": groups, "group_rgb": [[table[c]["rgb"] for c in g] for g in groups], "table": table,
           "tau": tau, "splits": rep, "privileged_diagnostic": diag, "minutes": round((time.time() - t0) / 60, 1)}
    (a.out / "discover.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({"objects": K, "groups": groups, "privileged_diagnostic": diag}), flush=True)


if __name__ == "__main__":
    main()
