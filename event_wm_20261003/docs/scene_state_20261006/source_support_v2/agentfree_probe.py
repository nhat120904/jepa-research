#!/usr/bin/env python3
"""Unified-method feasibility probe: the agent-free scene image as one state for every task family.

One rule set, no domain switches. With a fixed camera the robot moves almost all the time while the
things it manipulates (lights, cubes, drawers ...) change only at events, so
  1. agent mask: k-means on pixel colours (all pixels, C clusters); a cluster is AGENT if its pixel-mass
     centroid moves in most frames (2-means on log moving fraction, as in cube_discover.py) -- arm,
     gripper and shadow; lights, cubes, panel and floor are slow. Mask dilated by 1 px.
  2. scene image S_t = per-pixel median over the NON-agent frames in [t - W, t + W] (job 57176: a plain
     temporal median kept the slow arm base and its shadow, so S changed every frame);
  3. scene events = runs of frames whose changed-pixel count exceeds the count noise level (2-means on
     log counts), merged within `gap` frames; per-pixel change threshold = 2-means on log |dS| of the
     changed pixels; footprint = pixels differing between S before and after the run.
PRIVILEGED diagnostics only (never inputs): puzzle -> true light toggles (button_states);
cube -> qpos move start (pick) and end (place) times from cube_events.py.
Also saves a PNG per domain: frame, S, agent mask, and a few events (S before / after / footprint).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from sfa_code import two_means_threshold


def scene_images(obs, first, last, ep, s0, s1, W, device, chunk=256):
    """Temporal median S for frames [s0, s1). obs memmap (N, 64, 64, 3) uint8."""
    import torch

    lo = max(0, s0 - W); hi = min(len(obs), s1 + W)
    X = torch.as_tensor(np.asarray(obs[lo:hi]), device=device)
    out = np.zeros((s1 - s0, 64, 64, 3), np.uint8)
    offs = torch.arange(-W, W + 1, device=device)
    for c in range(s0, s1, chunk):
        t = torch.arange(c, min(s1, c + chunk), device=device)
        e = torch.as_tensor(ep[t.cpu().numpy()], device=device)
        f = torch.as_tensor(first, device=device)[e]; l = torch.as_tensor(last, device=device)[e]
        idx = torch.minimum(torch.maximum(t[:, None] + offs[None], f[:, None]), l[:, None]) - lo
        out[c - s0:c - s0 + len(t)] = X[idx].median(dim=1).values.cpu().numpy()
    return out


def colour_agent_mask(obs, ep, n, C, device, rng):
    """k-means colour clusters + agent (fast) clusters -> function frames -> agent mask (dilated)."""
    import torch

    from cube_discover import kmeans

    samp = np.asarray(obs[np.sort(rng.integers(0, n, 3000))]).reshape(-1, 3)
    px = torch.as_tensor(samp[rng.permutation(len(samp))[:400_000]], device=device).float() / 255.0
    cent = kmeans(px, C)
    v, u = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    m = min(n, 20_000)
    X = torch.as_tensor(np.asarray(obs[:m]), device=device).float() / 255.0
    lab = torch.cat([((X[i:i + 1024].reshape(-1, 4096, 1, 3) - cent[None, None]) ** 2).sum(-1).argmin(-1) for i in range(0, m, 1024)])
    uu = torch.as_tensor(u.ravel(), device=device).float(); vv = torch.as_tensor(v.ravel(), device=device).float()
    mass = np.zeros((m, C)); cu = np.zeros((m, C)); cv = np.zeros((m, C))
    for c in range(C):
        mk = (lab == c).float()
        ms = mk.sum(1)
        mass[:, c] = ms.cpu().numpy()
        cu[:, c] = ((mk * uu).sum(1) / ms.clamp(min=1)).cpu().numpy(); cv[:, c] = ((mk * vv).sum(1) / ms.clamp(min=1)).cpu().numpy()
    same = (ep[1:m] == ep[:m - 1])[:, None]
    pres = (mass[1:] >= 3) & (mass[:-1] >= 3) & same
    step = np.hypot(np.diff(cu, axis=0), np.diff(cv, axis=0))
    movf = np.array([(step[pres[:, c], c] > 0.5).mean() if pres[:, c].any() else 0.0 for c in range(C)])
    present = (mass >= 3).mean(0) > 0.05
    thr, sep, _ = two_means_threshold(np.log(movf[present] + 1e-3))
    agent = present & (np.log(movf + 1e-3) > thr)

    def mask(frames_t):
        lab = ((frames_t.float().reshape(len(frames_t), 4096, 1, 3) / 255.0 - cent[None, None]) ** 2).sum(-1).argmin(-1)
        a = torch.as_tensor(agent, device=device)[lab].reshape(-1, 1, 64, 64).float()
        return torch.nn.functional.max_pool2d(a, 3, 1, 1)[:, 0] > 0                   # dilate by 1 px

    info = {"clusters": C, "agent_clusters": np.nonzero(agent)[0].tolist(), "moving_frac": np.round(movf, 3).tolist(),
            "agent_rgb": (cent[torch.as_tensor(agent, device=device)].cpu().numpy() * 255).round().astype(int).tolist(), "sep": round(sep, 2)}
    return mask, info


def scene_images_masked(obs, first, last, ep, s0, s1, W, mask_fn, device, chunk=128):
    """S_t = per-pixel median over non-agent frames in the window (falls back to the plain median)."""
    import torch

    lo = max(0, s0 - W); hi = min(len(obs), s1 + W)
    X = torch.as_tensor(np.asarray(obs[lo:hi]), device=device)
    M = torch.cat([mask_fn(X[i:i + 2048]) for i in range(0, len(X), 2048)])            # (T, 64, 64) agent
    out = np.zeros((s1 - s0, 64, 64, 3), np.uint8)
    offs = torch.arange(-W, W + 1, device=device)
    for c in range(s0, s1, chunk):
        t = torch.arange(c, min(s1, c + chunk), device=device)
        e = torch.as_tensor(ep[t.cpu().numpy()], device=device)
        f = torch.as_tensor(first, device=device)[e]; l = torch.as_tensor(last, device=device)[e]
        idx = torch.minimum(torch.maximum(t[:, None] + offs[None], f[:, None]), l[:, None]) - lo
        x = X[idx].float()                                                              # (B, 2W+1, 64, 64, 3)
        mk = M[idx][..., None].expand_as(x)
        plain = x.median(dim=1).values
        xm = x.masked_fill(mk, float("nan"))
        med = torch.nanmedian(xm, dim=1).values
        out[c - s0:c - s0 + len(t)] = torch.where(torch.isnan(med), plain, med).round().clamp(0, 255).byte().cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=100_000, help="first val frames (whole episodes)")
    ap.add_argument("--W", type=int, default=15)
    ap.add_argument("--gap", type=int, default=5)
    ap.add_argument("--agent", choices=["none", "colour"], default="colour")
    ap.add_argument("--clusters", type=int, default=16)
    ap.add_argument("--ref-events", type=Path, default=None, help="cube_events.py dir (PRIVILEGED, cube only)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    a.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    obs = np.load(a.cache / "val_observations.npy", mmap_mode="r")
    term = np.load(a.cache / "val_terminals.npy")
    n = min(a.frames, len(obs))
    ep = np.concatenate([[0], np.cumsum(term[:-1])]).astype(np.int64)
    first = np.r_[0, np.nonzero(term)[0] + 1][: ep[-1] + 1]
    last = np.r_[np.nonzero(term)[0], len(term) - 1][: ep[-1] + 1]
    n = int(last[ep[n - 1] - 1] + 1) if ep[n - 1] > 0 else n            # whole episodes only
    rng = np.random.default_rng(0)
    agent_info = None
    if a.agent == "colour":
        mask_fn, agent_info = colour_agent_mask(obs, ep, n, a.clusters, dev, rng)
        S = scene_images_masked(obs, first, last, ep, 0, n, a.W, mask_fn, dev)
    else:
        S = scene_images(obs, first, last, ep, 0, n, a.W, dev)
    F = np.asarray(obs[:n])
    # agent-layer threshold (2-means on log colour distance frame vs S)
    samp = rng.integers(0, n, 3000)
    d = np.linalg.norm(F[samp].astype(np.float32) - S[samp].astype(np.float32), axis=-1).ravel()
    thr, sep, frac = two_means_threshold(np.log1p(d))
    tau = float(np.expm1(thr))
    agent_frac = float((np.linalg.norm(F[samp].astype(np.float32) - S[samp].astype(np.float32), axis=-1) > tau).mean())
    # scene changes
    same = ep[1:n] == ep[:n - 1]
    dd = []
    for c in rng.integers(1, n, 400):
        x = np.linalg.norm(S[c].astype(np.float32) - S[c - 1].astype(np.float32), axis=-1).ravel()
        dd.append(x[x > 0])
    dd = np.concatenate(dd)
    sthr, ssep, _ = two_means_threshold(np.log1p(dd))
    tau_s = float(np.expm1(sthr))
    dS = np.zeros(n, np.int64)
    for c in range(1, n, 20000):
        e = min(n, c + 20000)
        dS[c:e] = (np.linalg.norm(S[c:e].astype(np.float32) - S[c - 1:e - 1].astype(np.float32), axis=-1) > tau_s).sum((1, 2))
    dS[1:][~same] = 0
    nz = dS[dS > 0]
    cthr, csep, _ = two_means_threshold(np.log(nz.astype(float))) if len(nz) > 10 else (0.0, 0.0, 0.0)
    on = np.log(np.maximum(dS, 1)) > cthr
    events = []
    t = 1
    while t < n:
        if not on[t]:
            t += 1
            continue
        s0 = t; e0 = t
        while t < n and (on[t] or (t - e0 <= a.gap and ep[t] == ep[s0])):
            if on[t]:
                e0 = t
            t += 1
        b = S[max(s0 - 1, first[ep[s0]])].astype(np.float32); af = S[e0].astype(np.float32)
        fp = np.linalg.norm(af - b, axis=-1) > tau_s
        events.append((s0, e0, int(fp.sum())))
    ev = np.array(events) if events else np.zeros((0, 3), int)
    # footprint-size noise split (tiny flickers vs real changes): 2-means on log size
    fthr, fsep, ffrac = two_means_threshold(np.log1p(ev[:, 2].astype(float))) if len(ev) > 10 else (0.0, 0.0, 1.0)
    real = ev[np.log1p(ev[:, 2]) > fthr] if len(ev) > 10 else ev
    rep = {"env": a.cache.name, "frames": n, "episodes": int(ep[n - 1] + 1), "W": a.W, "agent_mode": a.agent, "agent_info": agent_info,
           "tau": round(tau, 1), "tau_sep": round(sep, 2), "tau_scene_change": round(tau_s, 1), "tau_scene_sep": round(ssep, 2),
           "count_thr_px": round(float(np.exp(cthr)), 1), "count_sep": round(csep, 2), "frames_changing_frac": round(float(on.mean()), 4),
           "agent_pixel_frac": round(agent_frac, 4), "raw_events": int(len(ev)),
           "footprint_split_px": round(float(np.expm1(fthr)), 1), "footprint_sep": round(fsep, 2),
           "events": int(len(real)), "events_per_episode": round(len(real) / (ep[n - 1] + 1), 2),
           "footprint_px_pct": np.percentile(real[:, 2], [10, 50, 90]).tolist() if len(real) else None,
           "duration_pct": np.percentile(real[:, 1] - real[:, 0] + 1, [10, 50, 90]).tolist() if len(real) else None}
    # PRIVILEGED diagnostics
    if (a.cache / "val_button_states.npy").exists():
        bs = np.load(a.cache / "val_button_states.npy", mmap_mode="r")[:n]
        tog = np.nonzero((np.asarray(bs[1:]) != np.asarray(bs[:-1])).any(1) & same)[0] + 1
        # group true toggles of one press (lights may flip over a few frames)
        presses = [tog[0]] if len(tog) else []
        for x in tog[1:]:
            if x - presses[-1] > 10:
                presses.append(x)
        presses = np.array(presses)
        hit_t = np.array([np.abs(real[:, 0] - p).min() <= a.W if len(real) else False for p in presses])
        hit_e = np.array([np.abs(presses - s).min() <= a.W for s in real[:, 0]]) if len(real) else np.array([])
        rep["privileged"] = {"true_presses": int(len(presses)), "recall": float(hit_t.mean()), "precision": float(hit_e.mean()) if len(hit_e) else None}
    if a.ref_events is not None:
        Q = np.load(a.ref_events / "cube_events_val.npz")
        m = Q["t"] < n
        qs, qe = Q["t_start"][m], Q["t"][m]
        near = lambda x: (np.abs(real[:, 0][None] - x[:, None]).min(1) <= a.W) if len(real) else np.zeros(len(x), bool)
        cand = np.r_[qs, qe]
        hit_e = np.array([np.abs(cand - s).min() <= a.W for s in real[:, 0]]) if len(real) else np.array([])
        rep["privileged"] = {"true_moves": int(m.sum()), "pick_recall": float(near(qs).mean()), "place_recall": float(near(qe).mean()),
                             "precision_vs_pick_or_place": float(hit_e.mean()) if len(hit_e) else None}
    # visualisation: 3 random frames (frame | S | agent mask) and 4 events (S before | S after | footprint)
    from PIL import Image
    rows = []
    for i in rng.integers(0, n, 3):
        mask = (np.linalg.norm(F[i].astype(np.float32) - S[i].astype(np.float32), axis=-1) > tau)
        rows.append(np.concatenate([F[i], S[i], np.repeat(mask[..., None] * 255, 3, -1).astype(np.uint8)], 1))
    for k in rng.choice(len(real), min(4, len(real)), replace=False) if len(real) else []:
        s0, e0, _ = real[k]
        b = S[max(s0 - 1, 0)]; af = S[e0]
        fp = np.linalg.norm(af.astype(np.float32) - b.astype(np.float32), axis=-1) > tau_s
        rows.append(np.concatenate([b, af, np.repeat(fp[..., None] * 255, 3, -1).astype(np.uint8)], 1))
    img = np.concatenate(rows, 0)
    tag = f"{a.cache.name}_W{a.W}_{a.agent}"
    Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(a.out / f"probe_{tag}.png")
    (a.out / f"probe_{tag}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
