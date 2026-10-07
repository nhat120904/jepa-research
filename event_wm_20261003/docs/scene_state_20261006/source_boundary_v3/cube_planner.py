#!/usr/bin/env python3
"""Cube direction D: event WM over (privileged) cube positions, affordance model, and move search.

Event WM  f(s, k, q_xy) -> s': state after moving cube k to xy q. The resulting height (ground or on
          top of another cube) and any side effects are learned from play moves, so "a cube needs
          support" is not hand-coded.
Affordance a(s) -> distribution over which cube can be moved, learned from which cube the play data
          moves in each state (the data only moves top cubes, so covered cubes get ~0 probability).
Search    breadth-first over moves (k, q) with q from the goal xy of every cube plus free buffer
          positions drawn from the data's free placements; goal test = every cube within `tol`.
          Cube tasks need <= ~5 moves, so no learned cost-to-go is used here (unlike the puzzle).
State     direction D: K x 3 metres (privileged qpos, cube_events.py). Direction A: K x (u, v[, covered])
          from label-free discovery (cube_events_px.py, whose events carry lo/hi/thr_px[/n_bin]).
          Binary state dims (covered) are predicted as logits; the goal test requires them to match,
          and a covered object is never moved (play data never moves one: data-support rule).
Run as a script: train both models on the events and report offline checks.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

LO = np.array([0.2, -0.4, 0.0], np.float32)          # coarse workspace bounds for input scaling
HI = np.array([0.9, 0.4, 0.2], np.float32)


def norm_s(s):
    return (s - LO) / (HI - LO) * 2 - 1


def make_cube_wm(K: int, hidden: int = 512, lo=LO, hi=HI, out_scale: float = 0.1, n_bin: int = 0):
    import torch
    import torch.nn as nn

    D = len(lo)
    P = D - n_bin                                                                  # continuous dims

    class CubeWM(nn.Module):
        def __init__(self):
            super().__init__()
            self.K = K
            self.net = nn.Sequential(nn.Linear(D * K + K + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, D * K))

        def forward(self, s, k, q, logits=False):
            """s (B, K, D) state units, k (B,), q (B, 2) -> s' (B, K, D); binary dims as probabilities
            (or, with logits=True, the pair (continuous part, binary logits))."""
            l, h = torch.as_tensor(lo, device=s.device), torch.as_tensor(hi, device=s.device)
            sn = ((s - l) / (h - l) * 2 - 1).flatten(1)
            qn = (q - l[:2]) / (h - l)[:2] * 2 - 1
            x = torch.cat([sn, nn.functional.one_hot(k.long(), self.K).float(), qn], -1)
            out = self.net(x).view(-1, self.K, D)
            if n_bin == 0:
                return s + out * out_scale
            cont = s[..., :P] + out[..., :P] * out_scale
            return (cont, out[..., P:]) if logits else torch.cat([cont, torch.sigmoid(out[..., P:])], -1)

    return CubeWM()


def make_affordance(K: int, hidden: int = 256, lo=LO, hi=HI):
    import torch
    import torch.nn as nn

    D = len(lo)

    class Affordance(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(D * K, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                     nn.Linear(hidden, K))

        def forward(self, s):
            l, h = torch.as_tensor(lo, device=s.device), torch.as_tensor(hi, device=s.device)
            return self.net(((s - l) / (h - l) * 2 - 1).flatten(1))

    return Affordance()


def load_cube_planner(path, device):
    """-> (wm, aff, checkpoint dict); handles both D (metres) and A (pixels) checkpoints."""
    import torch

    pk = torch.load(path, map_location="cpu", weights_only=False)
    lo, hi = np.asarray(pk.get("lo", LO), np.float32), np.asarray(pk.get("hi", HI), np.float32)
    wm = make_cube_wm(pk["K"], lo=lo, hi=hi, out_scale=pk.get("out_scale", 0.1), n_bin=pk.get("n_bin", 0)).to(device).eval()
    aff = make_affordance(pk["K"], lo=lo, hi=hi).to(device).eval()
    wm.load_state_dict(pk["wm"]); aff.load_state_dict(pk["aff"])
    return wm, aff, pk


def plan_moves(s0, goal, wm, aff, buffers, device, tol=0.04, max_depth=6, aff_min=0.02, max_nodes=200_000,
               chunk=65536, key_res=0.01, occ_min=0.0, n_bin=0):
    """BFS over move sequences. s0, goal (K, D); buffers (M, 2). Returns list of (k, q_xy) or None.
    Among the goal-reaching plans of the shortest depth, the one with the smallest predicted max error wins.
    occ_min > 0: data-support constraint -- a move whose target lies closer than occ_min to another
    object's current position (inside an occupied place, never seen in play data) is not considered.
    n_bin > 0: the last n_bin state dims are binary (covered): rounded, matched exactly in the goal
    test, and an object whose last dim is set (covered) is never moved."""
    import torch

    targets = np.concatenate([goal[:, :2], buffers], 0).astype(np.float32)          # candidate xy
    s0, goal = np.array(s0, np.float32), np.array(goal, np.float32)
    P = s0.shape[1] - n_bin
    if n_bin:
        s0[:, P:] = s0[:, P:] > 0.5; goal[:, P:] = goal[:, P:] > 0.5

    def state_key(x):
        return np.concatenate([np.round(x[..., :P] / key_res), x[..., P:]], -1).astype(np.int64).reshape(len(x), -1)

    frontier = [(s0, [])]
    seen = {tuple(state_key(s0[None])[0])}
    nodes = 0
    for depth in range(max_depth):
        if not frontier:
            break
        S = torch.as_tensor(np.stack([f[0] for f in frontier]), device=device)
        with torch.no_grad():
            p = torch.softmax(aff(S), -1).cpu().numpy()                                # (F, K)
        allowed = p >= aff_min
        if n_bin:
            allowed &= S[..., -1].cpu().numpy() < 0.5                                 # covered: never moved
        fi, ki = np.nonzero(allowed)
        if not len(fi):
            break
        Fi, Ki = np.repeat(fi, len(targets)), np.repeat(ki, len(targets))
        Qi = np.tile(np.arange(len(targets)), len(fi))
        if occ_min > 0:
            Snp = S.cpu().numpy()
            d = np.linalg.norm(Snp[Fi, :, :2] - targets[Qi][:, None], axis=-1)              # (N, K)
            d[np.arange(len(Ki)), Ki] = np.inf
            keep = d.min(1) >= occ_min
            Fi, Ki, Qi = Fi[keep], Ki[keep], Qi[keep]
            if not len(Fi):
                break
        nxt = []
        with torch.no_grad():
            for c in range(0, len(Fi), chunk):
                sl = slice(c, c + chunk)
                nxt.append(wm(S[torch.as_tensor(Fi[sl], device=device)], torch.as_tensor(Ki[sl], device=device),
                              torch.as_tensor(targets[Qi[sl]], device=device)).cpu().numpy())
        nxt = np.concatenate(nxt)
        if n_bin:
            nxt[..., P:] = nxt[..., P:] > 0.5
        nodes += len(nxt)
        err = np.linalg.norm(nxt[..., :P] - goal[None, :, :P], axis=-1).max(1)
        ok = err <= tol
        if n_bin:
            ok &= (nxt[..., P:] == goal[None, :, P:]).all((1, 2))
        hit = np.nonzero(ok)[0]
        if len(hit):
            j = hit[np.argmin(err[hit])]
            return frontier[Fi[j]][1] + [(int(Ki[j]), targets[Qi[j]].tolist())], {"depth": depth + 1, "nodes": nodes}
        if nodes > max_nodes or depth == max_depth - 1:
            break
        keys = state_key(nxt)
        _, first = np.unique(keys, axis=0, return_index=True)
        new_frontier = []
        for j in np.sort(first):
            key = tuple(keys[j])
            if key not in seen:
                seen.add(key)
                new_frontier.append((nxt[j], frontier[Fi[j]][1] + [(int(Ki[j]), targets[Qi[j]].tolist())]))
        frontier = new_frontier
    return None, {"depth": None, "nodes": nodes}


def buffer_positions(ground_xy, avoid, n=6, min_dist=0.08, seed=0):
    """Free buffer xy from the data's ground placements, at least min_dist from every xy in `avoid`."""
    rng = np.random.default_rng(seed)
    cand = ground_xy[rng.permutation(len(ground_xy))[:2000]]
    out = []
    for c in cand:
        if np.linalg.norm(avoid - c, axis=-1).min() >= min_dist and all(np.linalg.norm(c - o) >= min_dist for o in out):
            out.append(c)
        if len(out) == n:
            break
    return np.array(out, np.float32).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    dev = "cuda"
    tr, va = np.load(a.events / "cube_events_train.npz"), np.load(a.events / "cube_events_val.npz")
    K = tr["before"].shape[1]
    px = "lo" in tr.files                                         # direction A: pixel state
    lo, hi = (tr["lo"], tr["hi"]) if px else (LO, HI)
    n_bin = int(tr["n_bin"]) if "n_bin" in tr.files else 0
    P = len(lo) - n_bin
    # keep D's relative weighting of the WM loss vs the affordance loss: 1 unit ~ the metre scale
    unit = float(np.mean(hi - lo) / np.mean(HI[:2] - LO[:2])) if px else 1.0
    out_scale = 0.1 * unit
    tol = float(tr["thr_px"]) if px else 0.04
    wm, aff = make_cube_wm(K, lo=lo, hi=hi, out_scale=out_scale, n_bin=n_bin).to(dev), make_affordance(K, lo=lo, hi=hi).to(dev)
    T = {k: torch.as_tensor(tr[k], device=dev) for k in ("before", "after", "k", "target_xy")}
    opt = torch.optim.AdamW(list(wm.parameters()) + list(aff.parameters()), lr=1e-3, weight_decay=1e-4)
    for step in range(a.steps):
        i = torch.randint(0, len(T["k"]), (1024,), device=dev)
        aff_loss = F.cross_entropy(aff(T["before"][i]), T["k"][i].long())
        if n_bin:
            cont, logit = wm(T["before"][i], T["k"][i], T["target_xy"][i], logits=True)
            # position loss in px^2 (job 57130 used /unit as for D: ~0.03 vs BCE ~0.4, positions neglected)
            loss = (F.mse_loss(cont, T["after"][i][..., :P])
                    + F.binary_cross_entropy_with_logits(logit, T["after"][i][..., P:]) + aff_loss)
        else:
            pred = wm(T["before"][i], T["k"][i], T["target_xy"][i])
            loss = F.mse_loss(pred / unit, T["after"][i] / unit) * 100 + aff_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    wm.eval(); aff.eval()
    V = {k: torch.as_tensor(va[k], device=dev) for k in ("before", "after", "k", "target_xy")}
    with torch.no_grad():
        pred = wm(V["before"], V["k"], V["target_xy"]).cpu().numpy()
        pa = torch.softmax(aff(V["before"]), -1).cpu().numpy()
    after, before, k = va["after"], va["before"], va["k"]
    idx = np.arange(len(k))
    err_moved = np.linalg.norm(pred[idx, k, :P] - after[idx, k, :P], axis=-1)
    err_other = np.linalg.norm(pred[..., :P] - after[..., :P], axis=-1)            # (N, K)
    mask = np.ones((len(k), K), bool); mask[idx, k] = False
    res = {"units": "px" if px else "m", "tol": tol, "val_moves": int(len(k)),
           "moved_cube_err_median": float(np.median(err_moved)), "moved_cube_within_tol": float((err_moved < tol).mean()),
           "other_cubes_err_median": float(np.median(err_other[mask])),
           "affordance_top1_acc": float((pa.argmax(1) == k).mean())}
    if n_bin:
        pb, ab, bb = pred[..., P:] > 0.5, after[..., P:] > 0.5, before[..., P:] > 0.5
        flip = ab != bb
        res["binary_acc"] = float((pb == ab).mean())
        res["binary_acc_where_changed"] = float((pb == ab)[flip].mean()) if flip.any() else None
        res["binary_changes"] = int(flip.sum())
        res["moved_object_covered_before"] = float(bb[idx, k].mean())
    if not px:
        # privileged geometry: which cubes are covered (another cube above within 2 cm in xy)
        covered = np.zeros((len(k), K), bool)
        for i_ in range(K):
            for j_ in range(K):
                if i_ != j_:
                    covered[:, i_] |= (before[:, j_, 2] > before[:, i_, 2] + 0.02) & (np.linalg.norm(before[:, i_, :2] - before[:, j_, :2], axis=-1) < 0.02)
        stacked = after[idx, k, 2] > 0.04
        res.update({"stacked_moves_within_4cm": float((err_moved[stacked] < 0.04).mean()) if stacked.any() else None,
                    "stacked_moves": int(stacked.sum()),
                    "affordance_mass_on_covered_cubes": float(pa[covered].mean()) if covered.any() else None,
                    "affordance_mass_on_free_cubes": float(pa[~covered].mean())})
        ground = tr["target_xy"][tr["after"][np.arange(len(tr["k"])), tr["k"], 2] < 0.03]
    else:
        # free placements (label-free): the moved object ends > 2 thr from every other object
        ia = np.arange(len(tr["k"]))
        oth = np.linalg.norm(tr["after"][..., :2] - tr["after"][ia, tr["k"], :2][:, None], axis=-1)
        oth[ia, tr["k"]] = np.inf
        ground = tr["target_xy"][oth.min(1) > 2 * tol]
        res["free_placements"] = int(len(ground))
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"wm": wm.state_dict(), "aff": aff.state_dict(), "K": K, "ground_xy": ground, "lo": np.asarray(lo, np.float32),
                "hi": np.asarray(hi, np.float32), "out_scale": out_scale, "tol": tol, "n_bin": n_bin}, a.out / "cube_planner.pt")
    (a.out / "cube_planner_eval.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
