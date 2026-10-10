#!/usr/bin/env python3
"""Component 8c (method/README.md): the acted entity from a learned CONTACT model. One setting for every environment.

Why. The contact rule of events_objects.py (an identity the dilated arm mask touches at the contact moment, the one
nearest the centre of the known changes) is a hand rule: the arm mask touches most changed identities (held-out
puzzle-3x3 views: in the most frequent effect pattern both other members are touched in every event), and the centre of
an interaction's changes is not its contact point when the effects are asymmetric (a board's edge: pressed lights 5, 7,
8 labelled .33 / .31 / .07, their inner neighbours chosen instead). Relabelling by a world model trained on those labels
(relabel_events.py) made it worse there (.46 -> .37 of presses).

Model. C(frame) = a probability map over a 32 x 32 grid of where an interaction happens in this frame. Trained by
MULTIPLE-INSTANCE learning on the events (events_objects.py): at frames of an event's contact core the map's mass near the
event's known changed identities, summed, is maximised (-log sum_k p_k, p_k = mass within a 2 px Gaussian of identity k).
No identity is labelled as the acted one: the place that is among the changes whenever the agent looks as it does in the
frame (its effector) explains every event, a neighbour only some. Acted entity = among the identities not known to stay as
they were, the one with the most mass at the contact moment; its target is unknown when it was not seen before and after
(the event then trains neither the world model nor the skill), instead of a wrongly attributed neighbour.

Input: an events dir (events_objects.py) + the cache frames. Output: an events dir with e / target / target_known
replaced (other files copied), contact.pt, contact_report.json (PRIVILEGED scoring only where the cache has button states).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from utils import Frames, save_json

G = 32                                                                            # map cells per side (2 px)
SIGMA = 2.0                                                                       # px


def make_contact(w=64):
    import torch.nn as nn
    from frontend import Res, conv_block

    class ContactNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(conv_block(3, w // 2), conv_block(w // 2, w, 2), Res(w), conv_block(w, w, 2), Res(w), Res(w))
            self.up = nn.Sequential(nn.ConvTranspose2d(w, w // 2, 4, 2, 1), nn.GroupNorm(8, w // 2), nn.GELU(), nn.Conv2d(w // 2, 1, 3, 1, 1))

        def forward(self, x):
            return self.up(self.f(x * 2 - 1)).flatten(1)                          # (B, G * G) logits

    return ContactNet()


def kernels(pos):
    """identity positions (..., 2) px -> normalised Gaussian weights over the G x G cells (..., G * G)."""
    import torch
    c = (torch.arange(G, device=pos.device, dtype=torch.float32) * (64 / G) + (64 / G - 1) / 2)
    du = (c[None, :] - pos[..., 0:1]) ** 2                                        # (..., G) columns
    dv = (c[None, :] - pos[..., 1:2]) ** 2                                        # (..., G) rows
    k = torch.exp(-(dv[..., :, None] + du[..., None, :]) / (2 * SIGMA ** 2)).flatten(-2)
    return k / k.sum(-1, keepdim=True).clamp_min(1e-12)


def changed_sets(z, tol):
    bk, ak = z["before_known"].astype(bool), z["after_known"].astype(bool)
    unit = np.asarray(z["app_unit_id"], np.float64)
    unit = np.where(np.isfinite(unit), unit, 0.1)
    b4, af = z["before"], z["after"]
    chg = bk & ak & ((np.linalg.norm(af[..., :2] - b4[..., :2], axis=-1) > tol) | (np.abs(af[..., 2:5] - b4[..., 2:5]).max(-1) > unit))
    stay = bk & ak & ~chg
    return chg, stay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--jitter", type=int, default=2, help="training frames: the contact moment +- this many frames")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch / local/run_stage.ps1")
    import torch

    t0 = time.time()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    dev = a.device
    a.out.mkdir(parents=True, exist_ok=True)
    data = {}
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        z = dict(np.load(a.events / f"events_{split}.npz"))
        term = np.load(a.cache / f"{split}_terminals.npy")
        ends = np.flatnonzero(term)[:n_ep]; n = int(ends[-1] + 1)
        ep_of = np.concatenate([[0], np.cumsum(term[:n - 1])]).astype(np.int64)
        first = np.r_[0, ends[:-1] + 1][ep_of]; last = ends[ep_of]
        data[split] = (z, Frames(a.cache / f"{split}_observations.npy", n, ram=(split == "val")), first, last)
    tol = float(data["train"][0]["tol_pos"])
    z, obs, first, last = data["train"]
    chg, _ = changed_sets(z, tol)
    use = np.flatnonzero(chg.any(1))
    t_c = np.clip(z["t_core"][:, 0].astype(np.int64), 0, len(first) - 1)
    pos = torch.as_tensor(z["before"][..., :2], device=dev).float()               # (N, K, 2)
    chg_t = torch.as_tensor(chg, device=dev)
    net = make_contact().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    log = []
    for step in range(a.steps):
        i = use[rng.integers(0, len(use), a.batch)]
        f = np.clip(t_c[i] + rng.integers(-a.jitter, a.jitter + 1, len(i)), first[t_c[i]], last[t_c[i]])
        x = torch.as_tensor(obs[f], device=dev).permute(0, 3, 1, 2).float().div_(255)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            lg = net(x).float()
        p = torch.softmax(lg, -1)                                                 # (B, G*G)
        it = torch.as_tensor(i, device=dev)
        pk = (kernels(pos[it]) * p[:, None, :]).sum(-1)                           # (B, K) mass near each identity
        loss = -torch.log((pk * chg_t[it]).sum(-1).clamp_min(1e-8)).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % 1000 == 0 or step == a.steps - 1:
            log.append({"step": step, "loss": round(loss.item(), 4), "min": round((time.time() - t0) / 60, 1)}); print(log[-1], flush=True)
    net.eval()
    torch.save({"net": net.state_dict(), "G": G, "sigma": SIGMA}, a.out / "contact.pt")
    for f_ in a.events.iterdir():
        if f_.name not in ("events_train.npz", "events_val.npz"):
            shutil.copy(f_, a.out / f_.name)
    rep = {"events": str(a.events), "steps": a.steps, "log": log}
    for split, (z, obs, first, last) in data.items():
        chg, stay = changed_sets(z, tol)
        n = len(z["e"]); K = z["before"].shape[1]
        P = np.zeros((n, K), np.float32)
        t_c = np.clip(z["t_core"][:, 0].astype(np.int64), 0, len(first) - 1)
        with torch.no_grad():
            for s0 in range(0, n, 512):
                f = t_c[s0:s0 + 512]
                x = torch.as_tensor(obs[f], device=dev).permute(0, 3, 1, 2).float().div_(255)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    p = torch.softmax(net(x).float(), -1)
                P[s0:s0 + 512] = (kernels(torch.as_tensor(z["before"][s0:s0 + 512, :, :2], device=dev).float()) * p[:, None, :]).sum(-1).cpu().numpy()
        cand = ~stay                                                              # not known to stay as it was
        cand[~cand.any(1)] = True
        e_new = np.where(cand, P, -1.0).argmax(1)
        bk, ak = z["before_known"].astype(bool), z["after_known"].astype(bool)
        old = z["e"].copy()
        z["e"] = e_new
        z["target"] = z["after"][np.arange(n), e_new]
        z["target_known"] = bk[np.arange(n), e_new] & ak[np.arange(n), e_new]
        np.savez_compressed(a.out / f"events_{split}.npz", **z)
        rep[split] = {"events": int(n), "changed_from_contact_rule": float((e_new != old).mean()),
                      "target_known_share": float(z["target_known"].mean()),
                      "acted_is_a_known_change": float(chg[np.arange(n), e_new].mean()),
                      "mass_on_candidates_median": float(np.median(np.where(cand, P, 0).sum(1)))}
        print(split, json.dumps(rep[split]), flush=True)
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    save_json(a.out / "contact_report.json", rep)


if __name__ == "__main__":
    main()
