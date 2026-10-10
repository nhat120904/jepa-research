#!/usr/bin/env python3
"""Component 5c v2 (method/V2_PLAN.md, Sec. 3.1): the EFFECTOR TRACK, learned from frames and commanded actions only.

v1 (effector.py: a soft-argmax keypoint whose displacement over 1-4 steps predicts the mean command through a free MLP)
gave a diffuse map whose expectation sat ~16 px from the pressed light (held-out 4x4 / dev 4x5: nearest light at the press
frame .38 / .34). v2 makes the keypoint carry the effector's POSITION, not only its velocity:

  net(frame, arm map) -> heatmap over 64 x 64 restricted to the dilated arm mask (the effector is part of the agent: with
  a learned camera map the keypoint is otherwise any affine image of the effector position, and the first run put it
  ~33 px off the arm, 22% of its mass on the arm) -> keypoint kappa = (u, v) (expectation), height zeta (pooled features)
  q = M [u / 32 - 1, v / 32 - 1, zeta] + c          a learned affine camera inverse, q in units of summed action
  windows (t, h), h in {1, 2, 4, 8, 16, 32}: q_hat = q_t, q_hat <- q_hat + a_i[:3] with the height held at or above the
  FLOOR (running .5% quantile of q_z: a press pushes down while the effector rests on the table); one-sided, so a constant
  q cannot fit the upward motions. Without any bound the windows that push into the table did not fit (held-out 3x3 val
  window loss .7 vs .02 with bounds); learned two-sided bounds collapsed (--clip, see the flag)
  loss = Huber(q_{t+h} - q_hat) + anchor (heatmap mass inside the dilated arm mask) + small entropy

The OGBench command is an effector displacement (unit gain in summed-action units); clipping at learned workspace bounds
lets the integrator stop at the table while the command keeps pushing down (a press). Points of the arm nearer the base
move with gain below one and non-linearly, so the effector body (or its shadow, which meets it at the table) explains the
windows best; long windows make an offset point pay. One setting for every environment.

CONTACT RUNS (V2_PLAN.md Sec. 3.2): a frame is DOWN when zeta lies in the lower cluster of the 2-means split of TRAIN zeta
(Ashman D > 2, else none); a contact run is a maximal down run of >= 2 frames; its start is the contact moment.

Outputs (--out): effector.pt; track_{split}.npy (n, 7) float32 = u, v, zeta, q_x, q_y, q_z, peak mass; contacts.json
(zeta split); contacts_{split}.npy (runs: start, end) int64; effector_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from utils import Frames, save_json, two_means_threshold

GAPS = (1, 2, 4, 8, 16, 32)


def make_effector2(w=48):
    import torch
    import torch.nn as nn
    from frontend import Res, conv_block, up_block

    class Effector2(nn.Module):
        def __init__(self):
            super().__init__()
            self.e1 = conv_block(4, w)                                            # 64
            self.e2 = conv_block(w, 2 * w, 2)                                     # 32
            self.e3 = conv_block(2 * w, 2 * w, 2)                                 # 16
            self.mid = nn.Sequential(Res(2 * w), Res(2 * w))
            self.u2 = up_block(2 * w, 2 * w)                                      # 16 -> 32
            self.u1 = up_block(4 * w, w)                                          # 32 -> 64
            self.head = nn.Conv2d(2 * w, 1, 3, 1, 1)
            self.zeta = nn.Sequential(nn.Linear(2 * w, 64), nn.GELU(), nn.Linear(64, 1))
            self.cam = nn.Linear(3, 3)
            self.lo = nn.Parameter(torch.full((3,), -20.0)); self.hi = nn.Parameter(torch.full((3,), 20.0))
            g = torch.arange(64, dtype=torch.float32)
            self.register_buffer("gu", g.repeat(64)); self.register_buffer("gv", g.repeat_interleave(64))

        def track(self, x, arm):
            """x (B, 3, 64, 64) in [0, 1], arm (B, 1, 64, 64) in [0, 1] -> u, v (B,), zeta (B,), p (B, 4096), q (B, 3)."""
            h1 = self.e1(torch.cat([x * 2 - 1, arm * 2 - 1], 1))
            h2 = self.e2(h1)
            h3 = self.mid(self.e3(h2))
            f = torch.cat([self.u1(torch.cat([self.u2(h3), h2], 1)), h1], 1)    # (B, 2w, 64, 64)
            lg = self.head(f).flatten(1).float()
            p = torch.softmax(lg - 1e4 * (1 - arm.flatten(1).float()), -1)        # the effector is part of the agent
            u, v = (p * self.gu).sum(-1), (p * self.gv).sum(-1)
            z = self.zeta((f.flatten(2).float() * p[:, None]).sum(-1)).squeeze(-1)
            q = self.cam(torch.stack([u / 32 - 1, v / 32 - 1, z], -1))
            return u, v, z, p, q

    return Effector2()


def arm_maps(seg, idx):
    """dilated segmenter agent tokens of frames idx -> (n, 1, 64, 64) float (the anchor channel)."""
    from view import agent_tokens, token_pixels
    return token_pixels(agent_tokens(seg, idx, 1)).astype(np.float32)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="front-end dir (seg_{split}.npy)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--clip", action="store_true",
                    help="clip the integrated command at learned workspace bounds (V2_PLAN Sec 3.1). Off by default: learned "
                         "bounds collapsed (held-out 3x3: x bounds .61 / .62, so q_x became constant and the keypoint stopped "
                         "following the effector vertically in the image; offset to the pressed light -6 / +1 / +9.5 px by row)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch / local/run_stage.ps1")
    import torch
    import torch.nn.functional as F

    t0 = time.time()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    dev = a.device
    a.out.mkdir(parents=True, exist_ok=True)
    S = {}
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        term = np.load(a.cache / f"{split}_terminals.npy")
        ends = np.flatnonzero(term)[:n_ep]; n = int(ends[-1] + 1)
        ep_of = np.concatenate([[0], np.cumsum(term[:n - 1])]).astype(np.int64)
        act = np.asarray(np.load(a.cache / f"{split}_actions.npy", mmap_mode="r")[:n, :3], np.float32)
        S[split] = (Frames(a.cache / f"{split}_observations.npy", n, ram=(split == "val")), act, ends[ep_of], n,
                    np.load(a.run / f"seg_{split}.npy", mmap_mode="r"), np.r_[0, ends[:-1] + 1])
    H = max(GAPS)

    def sample(split, B, r):
        obs, act, last, n, seg, _ = S[split]
        h = np.array(GAPS)[r.integers(0, len(GAPS), B)]
        t = r.integers(0, n, B)
        t = np.maximum(np.minimum(t, last[t] - h), 0)
        h = np.minimum(h, last[t] - t)
        keep = h >= 1
        t, h = t[keep], h[keep]
        A = np.stack([act[np.minimum(t + i, n - 1)] for i in range(H)], 1)        # (B, H, 3)
        m = (np.arange(H)[None] < h[:, None])
        f0, f1 = t, t + h
        x = torch.as_tensor(obs[np.r_[f0, f1]], device=dev).permute(0, 3, 1, 2).float().div_(255)
        arm = torch.as_tensor(arm_maps(seg, np.r_[f0, f1]), device=dev)
        return x, arm, torch.as_tensor(A, device=dev), torch.as_tensor(m, device=dev), len(t)

    net = make_effector2().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))

    def window_loss(x, arm, A, m, B):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            u, v, z, p, q = net.track(x, arm)
        q = q.float(); q0, q1 = q[:B], q[B:]
        qh = q0
        for i in range(H):
            nxt = qh + A[:, i]
            if a.clip:
                nxt = torch.maximum(torch.minimum(nxt, net.hi), net.lo)
            elif floor[0] is not None:                                       # the table: a one-sided bound on height
                nxt = torch.cat([nxt[:, :2], torch.maximum(nxt[:, 2:], floor[0])], 1)
            qh = torch.where(m[:, i, None], nxt, qh)
        if not a.clip and net.training:                                      # floor = running .5% quantile of q_z
            qz = q.detach()[:, 2]
            f_ = torch.quantile(qz, 0.005)
            floor[0] = f_ if floor[0] is None else 0.99 * floor[0] + 0.01 * f_
        lid = F.huber_loss(q1, qh, delta=1.0)
        anchor = -torch.log((p * arm.flatten(1)).sum(-1).clamp_min(1e-8)).mean()   # 0 under the hard arm constraint
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1).mean()
        return lid, anchor, ent, p

    floor = [None]
    vr = np.random.default_rng(123)
    vb = [sample("val", 64, vr) for _ in range(4)]
    log = []
    for step in range(a.steps):
        x, arm, A, m, B = sample("train", a.batch, rng)
        if step == 1000 and a.clip:                                          # workspace bounds from the q now in use
            with torch.no_grad():
                qs = torch.cat([net.track(*sample("train", 64, rng)[:2])[4].float() for _ in range(8)])
                net.lo.copy_(torch.quantile(qs, 0.005, dim=0)); net.hi.copy_(torch.quantile(qs, 0.995, dim=0))
        lid, anchor, ent, p = window_loss(x, arm, A, m, B)
        w_anchor = 0.1 if step < 2000 else 0.01
        loss = lid + w_anchor * anchor + 1e-3 * ent
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % 1000 == 0 or step == a.steps - 1:
            net.eval()
            with torch.no_grad():
                vl = [window_loss(*b)[0].item() for b in vb]
            net.train()
            log.append({"step": step, "id": round(lid.item(), 4), "val_id": round(float(np.mean(vl)), 4), "anchor": round(anchor.item(), 3),
                        "floor": None if floor[0] is None else round(float(floor[0]), 3),
                        "peak": round(float(p.max(-1).values.mean()), 4), "lo": np.round(net.lo.detach().cpu().numpy(), 2).tolist(),
                        "hi": np.round(net.hi.detach().cpu().numpy(), 2).tolist(), "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
    net.eval()
    torch.save({"net": net.state_dict(), "gaps": GAPS}, a.out / "effector.pt")
    rep = {"run": str(a.run), "cache": str(a.cache), "steps": a.steps, "log": log}
    for split in ("train", "val"):
        obs, act, last, n, seg, starts = S[split]
        T = np.lib.format.open_memmap(a.out / f"track_{split}.npy", "w+", np.float32, (n, 7))
        with torch.no_grad():
            for s0 in range(0, n, 1024):
                idx = np.arange(s0, min(s0 + 1024, n))
                x = torch.as_tensor(obs[idx], device=dev).permute(0, 3, 1, 2).float().div_(255)
                arm = torch.as_tensor(arm_maps(seg, idx), device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    u, v, z, p, q = net.track(x, arm)
                T[idx] = torch.cat([torch.stack([u, v, z], -1).float(), q.float(), p.max(-1).values.float()[:, None]], -1).cpu().numpy()
        T.flush()
        S[split] = S[split] + (np.asarray(T),)
    # CONTACT RUNS: zeta in the lower cluster of the TRAIN 2-means split
    ztr = S["train"][-1][:, 2]
    thr, D, frac_hi = two_means_threshold(ztr[:: max(1, len(ztr) // 200000)])
    rep["contacts"] = {"zeta_threshold": thr, "ashman_D": D, "down_share": float((ztr <= thr).mean()) if D > 2 else 0.0}
    save_json(a.out / "contacts.json", rep["contacts"])
    for split in ("train", "val"):
        obs, act, last, n, seg, starts, Tr = S[split]
        down = (Tr[:, 2] <= thr) if D > 2 else np.zeros(n, bool)
        ep_start = np.zeros(n, bool); ep_start[starts] = True
        brk = np.r_[True, (down[1:] != down[:-1]) | ep_start[1:]]
        st = np.flatnonzero(brk); en = np.r_[st[1:], n] - 1
        runs = np.stack([st, en], 1)[down[st] & (en - st + 1 >= 2)]
        np.save(a.out / f"contacts_{split}.npy", runs.astype(np.int64))
        rep[split] = {"frames": n, "contact_runs": int(len(runs)), "runs_per_episode": float(len(runs) / len(starts)),
                      "u_p5_p95": np.percentile(Tr[:, 0], [5, 95]).round(1).tolist(), "v_p5_p95": np.percentile(Tr[:, 1], [5, 95]).round(1).tolist(),
                      "peak_p50": float(np.median(Tr[:, 6])), "step_px_p50": float(np.median(np.linalg.norm(np.diff(Tr[:, :2], axis=0), axis=-1)))}
        print(split, json.dumps(rep[split]), flush=True)
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    save_json(a.out / "effector_report.json", rep)


if __name__ == "__main__":
    main()
