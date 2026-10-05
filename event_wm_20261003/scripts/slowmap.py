#!/usr/bin/env python3
"""Unified-method probe 2: a dense slow-feature map learned from play video (deep spatial SFA).

One architecture and one set of hyperparameters for every task family. A fully convolutional encoder
maps a frame to a map Phi (32 x 32 x C). Trained on frame pairs (x_t, x_{t+k}), k ~ U{1..K}, same episode:
  invariance  mean |Phi_t - Phi_{t+k}|            (L1: the map changes rarely and abruptly)
  variance    hinge on the per-channel temporal std, i.e. the std over the batch at each location,
              averaged over locations (static background cells may stay constant; a fixed positional
              pattern does not count as variation, which a std pooled over locations would allow)
  covariance  off-diagonal channel covariance of the per-location-centred map (decorrelated channels)
  contrastive (--w-nce > 0) InfoNCE between pooled embeddings of Phi_t and Phi_{t+k}, negatives = the
              other pairs in the batch: Phi must hold everything that tells scenes apart (job 57180/57184:
              without it the cube map held only 1 of 3 cubes -- the variance hinge does not ask for
              completeness). k is drawn from [k_min, K] so that the arm has moved within a positive pair.
              --hard-neg: every anchor also brings a negative from the SAME episode |d| in [d_min, d_max]
              frames away (some objects have moved since): with negatives from other episodes only,
              one cube's position already separates the scenes (job 57185: still 1 of 3 cubes).
--objective layered (deep robust-PCA): a decoder reconstructs the frame from Phi and a mask net marks
  pixels Phi need not explain; loss = mean((1 - m) |D(Phi) - x|^2) + lam mean(m) + w_inv |Phi_t - Phi_t+k|.
  Every non-agent pixel must come from the slow map (completeness); the moving robot is cheaper to mask
  than to encode in a slow map. lam = the expected squared error of leaving a CHANGING pixel unexplained
  (centre of the upper 2-means cluster of per-pixel squared change over K frames); the variance hinge
  anchors the scale of Phi so that the L1 slowness cost cannot be dodged by shrinking the map.
  (Job 57195 used the 2-means split as lam and no scale anchor: masking was cheaper than encoding a cube,
  Phi shrank and absorbed the arm -- arm R^2 .83-.99, no cube.)
What is slow and varies across the data -- lights, cube positions, drawers -- must enter Phi; the arm
and its shadow move fast and the robot base never varies, so neither should. This generalises the
linear SFA of the puzzle code (sfa_code.py) to continuous, spatially placed state.
Evaluation (PRIVILEGED probes, never training inputs): event recall/precision of Phi jumps; linear
probes from Phi to light states (puzzle) / cube xy (cube) vs to arm joints (cube qpos) -- the scene
should be decodable, the arm should not.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from sfa_code import two_means_threshold


def make_encoder(C: int = 16, width: int = 64):
    import torch.nn as nn

    def block(cin, cout, stride=1):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(8, cout), nn.GELU())

    return nn.Sequential(block(3, width), block(width, width), block(width, width, 2), block(width, width),
                         block(width, width), nn.Conv2d(width, C, 1))


def make_decoder(C: int = 16, width: int = 64):
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU())

    return nn.Sequential(block(C, width), block(width, width), nn.Upsample(scale_factor=2, mode="nearest"),
                         block(width, width), block(width, width), nn.Conv2d(width, 3, 1), nn.Tanh())


def make_masknet(width: int = 32):
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, 1, 1), nn.GroupNorm(8, cout), nn.GELU())

    return nn.Sequential(block(3, width), block(width, width), block(width, width), nn.Conv2d(width, 1, 1))


def vicreg_terms(z, gamma=1.0, n_cov=32768):
    """z (B, C, H, W) -> variance hinge on the location-averaged temporal std, covariance penalty."""
    import torch

    zc = z - z.mean(0, keepdim=True)                                       # remove the static per-location layout
    std = torch.sqrt(zc.var(0) + 1e-4).mean((1, 2))                         # (C,)
    var = torch.relu(gamma - std).mean()
    flat = zc.permute(0, 2, 3, 1).reshape(-1, z.shape[1])
    flat = flat[torch.randperm(len(flat), device=z.device)[:n_cov]]
    cov = (flat.T @ flat) / (len(flat) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var, (off ** 2).sum() / z.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--train-frames", type=int, default=500_000)
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--k-min", type=int, default=1)
    ap.add_argument("--w-nce", type=float, default=0.0)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--hard-neg", action="store_true")
    ap.add_argument("--objective", choices=["vicreg", "layered"], default="vicreg")
    ap.add_argument("--d-min", type=int, default=150)
    ap.add_argument("--d-max", type=int, default=600)
    ap.add_argument("--C", type=int, default=16)
    ap.add_argument("--w-inv", type=float, default=25.0)
    ap.add_argument("--w-var", type=float, default=25.0)
    ap.add_argument("--w-cov", type=float, default=1.0)
    ap.add_argument("--eval-frames", type=int, default=60_000)
    ap.add_argument("--ref-events", type=Path, default=None, help="cube_events.py dir (PRIVILEGED, cube)")
    ap.add_argument("--eval-only", action="store_true", help="load out/slowmap.pt instead of training")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    obs = np.load(a.cache / "train_observations.npy", mmap_mode="r")
    term = np.load(a.cache / "train_terminals.npy")
    n = min(a.train_frames, len(obs))
    ep = np.concatenate([[0], np.cumsum(term[:-1])]).astype(np.int64)[:n]
    last = np.r_[np.nonzero(term)[0], len(term) - 1][ep]
    firsts = np.r_[0, np.nonzero(term)[0] + 1][ep]
    enc = make_encoder(a.C).to(dev)
    proj = torch.nn.Sequential(torch.nn.Linear(a.C * 64, 512), torch.nn.GELU(), torch.nn.Linear(512, 256)).to(dev)
    params = list(enc.parameters()) + list(proj.parameters())
    lam = None
    if a.objective == "layered":
        dec, mnet = make_decoder(a.C).to(dev), make_masknet().to(dev)
        params += list(dec.parameters()) + list(mnet.parameters())
        # lam: 2-means split of per-pixel squared change over K frames (static vs changing), images in [-1, 1]
        ii = rng.integers(0, n - a.K - 1, 2000)
        jj = np.minimum(ii + a.K, last[ii])
        dsq = ((np.asarray(obs[ii]).astype(np.float32) - np.asarray(obs[jj]).astype(np.float32)) / 127.5) ** 2
        dsq = dsq.mean(-1).ravel()
        ld = np.log(dsq[dsq > 0] + 1e-8)
        lthr, lsep, lfrac = two_means_threshold(ld)
        lam = float(np.exp(ld[ld > lthr].mean()))
        print({"lam": lam, "lam_sep": lsep, "changing_px_frac": lfrac}, flush=True)
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    log = []
    if a.eval_only:
        enc.load_state_dict(torch.load(a.out / "slowmap.pt", map_location=dev)["enc"])
    for step in range(0 if a.eval_only else a.steps):
        i = rng.integers(0, n - a.K - 1, a.batch)
        j = np.minimum(i + rng.integers(a.k_min, a.K + 1, a.batch), last[i])
        idx = np.r_[i, j]
        if a.hard_neg:
            d = rng.integers(a.d_min, a.d_max + 1, a.batch) * rng.choice([-1, 1], a.batch)
            h = np.clip(i + d, firsts[i], last[i])
            h = np.where(np.abs(h - i) < a.d_min, np.clip(i - d, firsts[i], last[i]), h)
            idx = np.r_[idx, h]
        x = torch.as_tensor(np.asarray(obs[idx]), device=dev).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z = enc(x).float()                                                      # (2B, C, 32, 32)
        z1, z2 = z[:a.batch], z[a.batch:2 * a.batch]
        inv = (z1 - z2).abs().mean()
        var, cov = vicreg_terms(z[:2 * a.batch])
        if a.objective == "layered":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xh = dec(z).float()
                mk = torch.sigmoid(mnet(x).float())
            rec = ((1 - mk) * ((xh - x) ** 2).mean(1, keepdim=True)).mean()
            loss = rec + lam * mk.mean() + a.w_inv * inv + a.w_var * var
            var = rec.detach(); cov = mk.mean().detach()                          # logged as rec / mask fraction
        else:
            loss = a.w_inv * inv + a.w_var * var + a.w_cov * cov
        nce = torch.zeros((), device=dev)
        if a.w_nce > 0:
            e = torch.nn.functional.normalize(proj(torch.nn.functional.adaptive_avg_pool2d(z, 8).flatten(1)), dim=-1)
            ea, ep_, en = e[:a.batch], e[a.batch:2 * a.batch], e[2 * a.batch:]
            lab = torch.arange(a.batch, device=dev)
            logits = ea @ torch.cat([ep_, en]).T / a.tau                         # own positive vs all others
            nce = torch.nn.functional.cross_entropy(logits, lab)
            if not a.hard_neg:
                nce = 0.5 * (nce + torch.nn.functional.cross_entropy((ea @ ep_.T / a.tau).T, lab))
            loss = loss + a.w_nce * nce
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 1000 == 0 or step == a.steps - 1:
            log.append({"step": step, "inv": inv.item(), "var": var.item(), "cov": cov.item(), "nce": nce.item(), "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
    enc.eval()
    if not a.eval_only:
        sv = {"enc": enc.state_dict(), "C": a.C, "objective": a.objective, "lam": lam}
        if a.objective == "layered":
            sv.update(dec=dec.state_dict(), mask=mnet.state_dict())
        torch.save(sv, a.out / "slowmap.pt")

    # ---------------- evaluation on val (PRIVILEGED probes) ----------------
    vo = np.load(a.cache / "val_observations.npy", mmap_mode="r")
    vt = np.load(a.cache / "val_terminals.npy")
    m = min(a.eval_frames, len(vo))
    vep = np.concatenate([[0], np.cumsum(vt[:-1])]).astype(np.int64)[:m]
    Z = np.zeros((m, a.C, 32, 32), np.float16)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for s in range(0, m, 2048):
            x = torch.as_tensor(np.array(vo[s:min(m, s + 2048)]), device=dev).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)
            Z[s:s + len(x)] = enc(x).float().cpu().numpy()
    Zf = Z.astype(np.float32)
    same = vep[1:] == vep[:-1]
    jump = np.zeros(m); jump[1:] = np.abs(Zf[1:] - Zf[:-1]).mean((1, 2, 3)); jump[1:][~same] = 0
    jthr, jsep, jfrac = two_means_threshold(np.log(jump[1:][same] + 1e-6))
    on = np.log(jump + 1e-6) > jthr
    # events = runs of jump frames merged within 5 frames
    ev, t = [], 1
    while t < m:
        if not on[t]:
            t += 1; continue
        s0 = e0 = t
        while t < m and (on[t] or (t - e0 <= 5 and vep[t] == vep[s0])):
            if on[t]:
                e0 = t
            t += 1
        ev.append((s0, e0))
    ev = np.array(ev) if ev else np.zeros((0, 2), int)
    res = {"env": a.cache.name, "frames": m, "jump_thr": float(np.exp(jthr)), "jump_sep": jsep, "jump_frame_frac": float(on.mean()),
           "events": int(len(ev)), "events_per_episode": float(len(ev) / (vep[-1] + 1)),
           "event_duration_pct": np.percentile(ev[:, 1] - ev[:, 0] + 1, [10, 50, 90]).tolist() if len(ev) else None}
    # pooled features for linear probes: 8x8 average pooling of Phi (C x 8 x 8)
    P = Zf.reshape(m, a.C, 8, 4, 8, 4).mean((3, 5)).reshape(m, -1)
    P = (P - P.mean(0)) / (P.std(0) + 1e-6)
    half = m // 2

    def ridge(X, Y, lam=10.0):
        Xb = np.c_[X, np.ones(len(X))]
        W = np.linalg.solve(Xb[:half].T @ Xb[:half] + lam * np.eye(Xb.shape[1]), Xb[:half].T @ Y[:half])
        return Xb[half:] @ W

    if (a.cache / "val_button_states.npy").exists():
        bs = np.asarray(np.load(a.cache / "val_button_states.npy", mmap_mode="r")[:m]).astype(np.float32)
        pr = ridge(P, bs)
        acc = ((pr > 0.5) == (bs[half:] > 0.5)).mean(0)
        res["probe_lights_acc_per_light"] = np.round(acc, 3).tolist()
        res["probe_lights_acc_min_mean"] = [float(acc.min()), float(acc.mean())]
        tog = np.nonzero((bs[1:] != bs[:-1]).any(1) & same)[0] + 1
        pres = [tog[0]] if len(tog) else []
        for x in tog[1:]:
            if x - pres[-1] > 10:
                pres.append(x)
        pres = np.array(pres)
        res["event_recall"] = float(np.mean([np.abs(ev[:, 0] - p).min() <= 15 for p in pres])) if len(ev) else 0.0
        res["event_precision"] = float(np.mean([np.abs(pres - s).min() <= 15 for s in ev[:, 0]])) if len(ev) else None
    if (a.cache / "val_qpos.npy").exists() and (a.cache / "val_button_states.npy").exists():
        # PRIVILEGED reference for scene-like envs: any object joint moving (qpos[14:]) or a button toggling
        qq = np.asarray(np.load(a.cache / "val_qpos.npy", mmap_mode="r")[:m]).astype(np.float32)
        bb = np.asarray(np.load(a.cache / "val_button_states.npy", mmap_mode="r")[:m])
        mv = (np.abs(np.diff(qq[:, 14:], axis=0)).max(1) > 2e-3) | (bb[1:] != bb[:-1]).any(1)
        mv &= same
        d_ = np.diff(np.r_[0, mv.astype(np.int8), 0]); st_, en_ = np.nonzero(d_ == 1)[0] + 1, np.nonzero(d_ == -1)[0]
        keep = [0]
        for i_ in range(1, len(st_)):
            if st_[i_] - en_[keep[-1]] > 10:
                keep.append(i_)
            else:
                en_[keep[-1]] = en_[i_]
        st_, en_ = st_[keep], en_[keep]
        cand = np.r_[st_, en_]
        res["object_events_ref"] = int(len(st_))
        res["object_event_recall"] = float(np.mean([(np.abs(ev[:, 0] - x).min() <= 15) or (np.abs(ev[:, 0] - y).min() <= 15) for x, y in zip(st_, en_)])) if len(ev) else 0.0
        res["object_event_precision"] = float(np.mean([np.abs(cand - s_).min() <= 15 for s_ in ev[:, 0]])) if len(ev) else None
    if (a.cache / "val_qpos.npy").exists():
        q = np.asarray(np.load(a.cache / "val_qpos.npy", mmap_mode="r")[:m]).astype(np.float32)
        slices = [s for s in (14, 21, 28, 35) if s + 3 <= q.shape[1]]
        cubes = np.concatenate([q[:, s:s + 2] for s in slices], 1)
        arm = q[:, :6]
        r2 = lambda pred, Y: 1 - ((pred - Y[half:]) ** 2).sum(0) / ((Y[half:] - Y[half:].mean(0)) ** 2).sum(0)
        res["probe_cube_xy_r2"] = np.round(r2(ridge(P, cubes), cubes), 3).tolist()
        res["probe_arm_joint_r2"] = np.round(r2(ridge(P, arm), arm), 3).tolist()
        # baseline: same probe on 8x8-pooled raw pixels
        R = np.asarray(vo[:m]).reshape(m, 8, 8, 8, 8, 3).mean((2, 4)).reshape(m, -1).astype(np.float32)
        R = (R - R.mean(0)) / (R.std(0) + 1e-6)
        res["pixels_probe_cube_xy_r2"] = np.round(r2(ridge(R, cubes), cubes), 3).tolist()
        res["pixels_probe_arm_joint_r2"] = np.round(r2(ridge(R, arm), arm), 3).tolist()
        if a.ref_events is not None:
            Q = np.load(a.ref_events / "cube_events_val.npz")
            mm = Q["t"] < m
            cand = np.r_[Q["t_start"][mm], Q["t"][mm]]
            near = lambda x: (np.abs(ev[:, 0][None] - x[:, None]).min(1) <= 15) if len(ev) else np.zeros(len(x), bool)
            res["pick_recall"] = float(near(Q["t_start"][mm]).mean()); res["place_recall"] = float(near(Q["t"][mm]).mean())
            res["event_precision_vs_pick_or_place"] = float(np.mean([np.abs(cand - s).min() <= 15 for s in ev[:, 0]])) if len(ev) else None
    res["log"] = log
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (a.out / f"slowmap_{a.cache.name}.json").write_text(json.dumps(res, indent=1) + "\n")
    # visualise: frame and 3 PCA channels of Phi for 6 frames
    from PIL import Image
    idx = np.sort(rng.integers(0, m, 6))
    Zs = Zf[idx].transpose(0, 2, 3, 1).reshape(-1, a.C)
    U, S_, Vt = np.linalg.svd(Zs - Zs.mean(0), full_matrices=False)
    pc = ((Zs - Zs.mean(0)) @ Vt[:3].T).reshape(6, 32, 32, 3)
    pc = ((pc - pc.min()) / (pc.max() - pc.min() + 1e-6) * 255).astype(np.uint8)
    rows = [np.concatenate([np.asarray(vo[i]), np.kron(pc[r], np.ones((2, 2, 1), np.uint8))], 1) for r, i in enumerate(idx)]
    if a.objective == "layered":
        with torch.no_grad():
            xi = torch.as_tensor(np.array(vo[np.sort(idx)]), device=dev).permute(0, 3, 1, 2).float().div(127.5).sub(1)
            xh = ((dec(enc(xi)).clamp(-1, 1) + 1) * 127.5).byte().permute(0, 2, 3, 1).cpu().numpy()
            mk = (torch.sigmoid(mnet(xi))[:, 0] * 255).byte().cpu().numpy()
        rows = [np.concatenate([r_, xh[k], np.repeat(mk[k][..., None], 3, -1)], 1) for k, r_ in enumerate(
            [np.concatenate([np.asarray(vo[i]), np.kron(pc[r], np.ones((2, 2, 1), np.uint8))], 1) for r, i in enumerate(np.sort(idx))])]
    img = np.concatenate(rows, 0)
    Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(a.out / f"slowmap_{a.cache.name}.png")
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)


if __name__ == "__main__":
    main()
