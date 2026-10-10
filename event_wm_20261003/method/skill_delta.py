#!/usr/bin/env python3
"""Low level L0 (method/V2_PLAN.md Sec. 4): an executor conditioned on DELTA MAPS (goal_maps.py), no acted entity.

pi(chunk | current frame, maps of what must change from the event's before-state to its after-state). Data = the event
segments of events_objects.py (frames from the end of the previous event to this event's end + release), every entity known
before and after enters Delta when it changes (the events' change rule), so attribution never reaches the executor. One frame
(two frames copy the ongoing motion and ignore the condition: puzzle condition-swap .095); behaviour cloning of 16-action
chunks (MSE over valid steps), shift augmentation applied to frame and maps together; best validation checkpoint kept.
Events whose Delta is empty (nothing known changed) do not train it.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

from skill_segments import segment_frames
from utils import Frames, save_json


def make_delta_skill(chunk=16, act_dim=5, width=32, hidden=512, in_ch=6):
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(), nn.MaxPool2d(2))

    class DeltaSkill(nn.Module):
        def __init__(self):
            super().__init__()
            self.chunk, self.act_dim = chunk, act_dim
            self.cnn = nn.Sequential(block(in_ch, width), block(width, 2 * width), block(2 * width, 2 * width), block(2 * width, 4 * width))
            self.head = nn.Sequential(nn.Linear(4 * width * 16, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, chunk * act_dim))

        def forward(self, frame, maps):
            """frame (B, 3, 64, 64) in [0, 1], maps (B, 3 or 4, 64, 64) in [0, 1] -> (B, chunk, act_dim) (standardised actions)."""
            import torch
            f = self.cnn(torch.cat([frame * 2 - 1, maps * 2 - 1], 1))
            return self.head(f.flatten(1)).view(len(frame), self.chunk, self.act_dim)

    return DeltaSkill()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--release", type=int, default=10)
    ap.add_argument("--press-point", type=Path, default=None,
                    help="effector3.py output: a 4th map channel, a blob at the effector position at the event's contact moment "
                         "(t_core0; the realised press / grasp point, no attribution label); in the loop the planned entity's position")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch / local/run_stage.ps1")
    import torch
    from goal_maps import delta_mask, render_torch

    torch.manual_seed(0); rng = np.random.default_rng(0)
    dev = a.device
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    D = {}
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        term = np.load(a.cache / f"{split}_terminals.npy")
        ends = np.flatnonzero(term)[:n_ep]; n = int(ends[-1] + 1)
        ep_of = np.concatenate([[0], np.cumsum(term[:n - 1])]).astype(np.int64)
        last = ends[ep_of]
        z = dict(np.load(a.events / f"events_{split}.npz"))
        tol, thr, unit = float(z["tol_pos"]), np.asarray(z["thr_app_id"], np.float64), np.asarray(z["app_unit_id"], np.float64)
        unit = np.where(np.isfinite(unit), unit, 0.1)
        dm = delta_mask(z["before"], z["after"], z["before_known"].astype(bool), z["after_known"].astype(bool), tol, thr)
        rows, evs = [], []
        for i in np.flatnonzero(dm.any(1)):
            te = int(z["t"][i])
            if te >= n:
                continue
            f = np.fromiter(segment_frames(int(z["seg_start"][i]), te, a.release, int(last[te])), dtype=np.int64)
            rows.append(f); evs.append(np.full(len(f), i))
        rows, evs = np.concatenate(rows), np.concatenate(evs)
        act = np.asarray(np.load(a.cache / f"{split}_actions.npy", mmap_mode="r")[:n], np.float32)
        pp = None
        if a.press_point is not None:                                        # effector at the contact moment, per event
            tr_ = np.load(a.press_point / f"track_{split}.npy", mmap_mode="r")
            pp = torch.as_tensor(np.asarray(tr_[np.clip(z["t_core"][:, 0], 0, n - 1), :2], np.float32), device=dev)
        D[split] = dict(obs=Frames(a.cache / f"{split}_observations.npy", n, ram=(split == "val")), act=act, last=last, rows=rows, evs=evs, pp=pp,
                        S=torch.as_tensor(z["before"], device=dev).float(), G=torch.as_tensor(z["after"], device=dev).float(),
                        dm=torch.as_tensor(dm, device=dev), unit=torch.as_tensor(unit, device=dev).float(), tol=tol, thr=thr,
                        n_events=int(dm.any(1).sum()), n_all=int(len(dm)))
    mu, sd = D["train"]["act"].mean(0), D["train"]["act"].std(0) + 1e-6
    H = a.chunk

    def batch(split, idx, augment=False):
        d = D[split]
        t, e = d["rows"][idx], d["evs"][idx]
        x = torch.as_tensor(d["obs"][t], device=dev).permute(0, 3, 1, 2).float().div_(255)
        shift = None
        if augment:
            dx, dy = rng.integers(0, 5, 2)
            x = torch.nn.functional.pad(x, (2, 2, 2, 2), mode="replicate")[:, :, dy:dy + 64, dx:dx + 64]
            shift = torch.tensor([2.0 - float(dx), 2.0 - float(dy)], device=dev, dtype=torch.float32).expand(len(t), 2)
        et = torch.as_tensor(e, device=dev)
        maps = render_torch(d["S"][et], d["G"][et], d["dm"][et], d["unit"], shift)
        if d["pp"] is not None:
            g = torch.arange(64, device=dev, dtype=torch.float32)
            q = d["pp"][et] + (shift if shift is not None else 0.0)
            blob = torch.exp(-((g[None, None, :] - q[:, 0, None, None]) ** 2 + (g[None, :, None] - q[:, 1, None, None]) ** 2) / (2 * 2.0 ** 2))
            maps = torch.cat([maps, blob[:, None]], 1)
        steps = t[:, None] + np.arange(H)[None]
        valid = steps <= d["last"][t][:, None]
        steps = np.minimum(steps, d["last"][t][:, None])
        y = torch.as_tensor((d["act"][steps] - mu) / sd, device=dev)
        return x, maps, y, torch.as_tensor(valid, device=dev).float()

    pi = make_delta_skill(chunk=H, in_ch=7 if a.press_point is not None else 6).to(dev)
    opt = torch.optim.AdamW(pi.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    rep = {split: {"rows": int(len(D[split]["rows"])), "events_with_delta": D[split]["n_events"], "events": D[split]["n_all"]} for split in D}
    print(rep, flush=True)
    meta = {"kind": "delta", "press_point": a.press_point is not None, "chunk": H, "action_mean": mu, "action_std": sd, "tol_pos": D["train"]["tol"], "thr_app_id": D["train"]["thr"],
            "app_unit_id": D["train"]["unit"].cpu().numpy(), "release": a.release}
    log, best = [], float("inf")
    vsel = np.random.default_rng(1).integers(0, len(D["val"]["rows"]), 2048)
    for step in range(a.steps):
        idx = rng.integers(0, len(D["train"]["rows"]), a.batch)
        x, maps, y, m = batch("train", idx, augment=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            pred = pi(x, maps).float()
        loss = (((pred - y) ** 2).mean(-1) * m).sum() / m.sum()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt.step(); sched.step()
        if step % 2000 == 0 or step == a.steps - 1:
            pi.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                vl, vcs = [], []
                for s0 in range(0, 2048, 256):
                    xv, mv, yv, mm = batch("val", vsel[s0:s0 + 256])
                    pv = pi(xv, mv).float()
                    vl.append(((((pv - yv) ** 2).mean(-1) * mm).sum() / mm.sum()).item())
                    # condition sensitivity: the same frames with the maps of other events (rolled), vs other frames, same maps
                    pg = pi(xv, mv.roll(1, 0)).float(); po = pi(xv.roll(1, 0), mv).float()
                    vcs.append(((pv - pg).abs().mean() / (pv - po).abs().mean().clamp_min(1e-6)).item())
            pi.train()
            log.append({"step": step, "loss": round(loss.item(), 4), "val_chunk": round(float(np.mean(vl)), 4), "csr": round(float(np.mean(vcs)), 3),
                        "min": round((time.time() - t0) / 60, 1)})
            print(log[-1], flush=True)
            if log[-1]["val_chunk"] < best:
                best = log[-1]["val_chunk"]
                torch.save({"skill": pi.state_dict(), "step": step, **meta}, a.out / "skill_best.pt")
    torch.save({"skill": pi.state_dict(), **meta}, a.out / "skill.pt")
    rep["log"] = log
    save_json(a.out / "skill_report.json", rep)


if __name__ == "__main__":
    main()
