#!/usr/bin/env python3
"""Scene memory v2 on VAL: rule-based memory over the learned codes, events, and PRIVILEGED scoring (simulator state
is used here only to score; nothing feeds back into training).

Memory (sm2_model.memory_rule): a token takes its current code after k consecutive trusted frames with that code.
Four variants = code space x trust: learned SceneCodes or raw-pixel tokens (4 x 4 patch mean RGB, 8 levels per channel,
the control that separates the learned codes' contribution), trusted = outside the dilated agent mask (teacher |
segmenter) or = SeeThrough probability > .5 (the code of the scene behind the agent). Event frames = frames where a known
token's memory changes; events = runs merged within --merge frames (the criterion's definition) and, separately,
interactions from sm2_model.assemble_events with the gap fitted label-free on --fit-episodes TRAIN episodes.

Scored against the criteria pre-registered on 2026-10-04 for the unified state component (JOB_LEDGER.md), with the
same reference definitions as slowmap.py / slowmap_probe.py:
  c1 discrete state   linear probe per button / light >= .98
  c2 completeness     median xy error per cube of a probe on the memory <= 1.2 x the same probe on raw pixels.
                      Pre-registered probe: keypoint (spatial softmax). DEVIATION (2026-10-08): the keypoint probe is
                      unreliable on raw pixels (cube-triple VAL, same data: 3.5-15 cm over seeds / lengths), so c2 is
                      judged with an MLP (2 x 512) on 8 x 8-pooled features (raw pixels 4.4-5.6 cm, stable over seeds);
                      the keypoint numbers are still reported.
  c3 agent-free       arm-joint MLP R^2 <= .2
  c4 events           recall >= .9 and precision >= .8, window 15 frames; reference = any object joint (qpos[14:])
                      moving > 2e-3 per frame or a button toggling (scene, cube), light toggles (puzzle); reference
                      segments closer than 10 frames merged
Also (diagnostics, not criteria): recall per object type; button probe on settled frames (object rest and no object
change in the previous 30 frames: the memory waits until the agent has left, so right after a press it still holds the
old state); stability at rest and distinctness of rest periods (as sm_diag.py); agent-mask diagnostics; event lag after
the end of the object change; panels.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sm2_model import (SceneCodes, SeeThrough, adjacent_gaps, agent_masks, assemble_events, backdate, memory_rule,
                       merge_onsets, pixel_codes, see_codes, two_means_threshold)
from sm_diag import episode_bounds, periods, privileged_state, rest_mask, state_differs


def pixel_tokens(obs, n):
    """raw-pixel control codes (n, 256), sm2_model.pixel_codes in chunks."""
    return np.concatenate([pixel_codes(obs[s:min(s + 5000, n)]) for s in range(0, n, 5000)])


TYPE_DIMS = {"scene": {"cube": range(14, 21), "buttons": range(21, 23), "drawer": [23], "window": [24]},
             "cube": {"cube0": range(14, 21), "cube1": range(21, 28), "cube2": range(28, 35)}}


def reference_segments(cache, family, n, starts, T, gap=10):
    """PRIVILEGED object-change segments [(start, end)], never across episodes, merged when closer than gap frames:
    all objects together (the criterion) and per object type. Also the per-frame change indicator."""
    ep = np.repeat(np.arange(len(starts)), T)[:n]
    same = np.r_[False, ep[1:] == ep[:-1]]
    bfile = cache / "val_button_states.npy"
    tog = None
    if bfile.exists():
        bs = np.asarray(np.load(bfile, mmap_mode="r")[:n])
        tog = np.r_[False, (bs[1:] != bs[:-1]).any(1)]
    per = {}
    if family == "puzzle":
        mv = tog.copy()
        per["lights"] = merge_onsets(tog & same, starts, T, merge=gap)
    else:
        q = np.asarray(np.load(cache / "val_qpos.npy", mmap_mode="r")[:n], np.float32)
        mv = np.r_[False, np.abs(np.diff(q[:, 14:], axis=0)).max(1) > 2e-3]
        if tog is not None:
            mv |= tog
        for name, dims in TYPE_DIMS[family].items():
            m = np.r_[False, np.abs(np.diff(q[:, list(dims)], axis=0)).max(1) > 2e-3]
            if name == "buttons" and tog is not None:
                m |= tog
            per[name] = merge_onsets(m & same, starts, T, merge=gap)
    return merge_onsets(mv & same, starts, T, merge=gap), per, mv & same


def near(x, ref, w):
    """bool per x: some value of ref lies within w of it."""
    x, ref = np.asarray(x), np.sort(np.asarray(ref))
    if not len(ref):
        return np.zeros(len(x), bool)
    i = np.searchsorted(ref, x)
    d = np.minimum(np.abs(ref[np.clip(i, 0, len(ref) - 1)] - x), np.abs(ref[np.clip(i - 1, 0, len(ref) - 1)] - x))
    return d <= w


def score_events(ev_start, segs, w):
    """criterion definition (slowmap.py): a reference segment is found when an event start lies within w of its start or
    end; an event is correct when its start lies within w of any reference start or end."""
    cand = np.r_[segs[:, 0], segs[:, 1]]
    rec = (near(segs[:, 0], ev_start, w) | near(segs[:, 1], ev_start, w)).mean() if len(segs) else None
    prec = near(ev_start, cand, w).mean() if len(ev_start) else None
    return {"recall": None if rec is None else float(rec), "precision": None if prec is None else float(prec)}


def score_spans(ev, segs, w):
    """events as spans [start, end] (assembled interactions): found / correct when the start OR the end matches."""
    pts = np.r_[ev[:, 0], ev[:, 1]]
    cand = np.r_[segs[:, 0], segs[:, 1]]
    rec = (near(segs[:, 0], pts, w) | near(segs[:, 1], pts, w)).mean() if len(segs) else None
    prec = (near(ev[:, 0], cand, w) | near(ev[:, 1], cand, w)).mean() if len(ev) else None
    return {"recall": None if rec is None else float(rec), "precision": None if prec is None else float(prec)}


def event_lag(ev_start, segs, w, horizon=200):
    """For each reference segment: first event start >= seg start - w; lag = that start - seg end (if within horizon)."""
    ev_start = np.sort(ev_start)
    if not len(ev_start) or not len(segs):
        return {"segments_with_event_within_horizon": 0.0, "lag_after_end_p10_p50_p90": None}
    i = np.searchsorted(ev_start, segs[:, 0] - w)
    ok = i < len(ev_start)
    lag = ev_start[np.minimum(i, len(ev_start) - 1)] - segs[:, 1]
    lag = lag[ok & (lag <= horizon)]
    return {"segments_with_event_within_horizon": float(len(lag) / max(len(segs), 1)),
            "lag_after_end_p10_p50_p90": np.percentile(lag, [10, 50, 90]).round(1).tolist() if len(lag) else None}


def probe_suite(V, st, pix_kp, dev, steps=3000, sub=None, pix_mlp=None):
    """PRIVILEGED probes on features V (n, C, 16, 16) (torch, on dev). First half of the frames fits, second half scores.
    sub (n,) bool: also score the button and joint probes on this subset of the second half."""
    torch.manual_seed(0)
    n = len(V); half = n // 2
    res = {}

    def fit(Fx, Y, kind):
        mu, sd = Fx[:half].mean(0), Fx[:half].std(0) + 1e-3
        net = (nn.Linear(Fx.shape[1], Y.shape[1]) if kind == "logit" else
               nn.Sequential(nn.Linear(Fx.shape[1], 256), nn.GELU(), nn.Linear(256, Y.shape[1]))).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        ym, ys = Y[:half].mean(0), Y[:half].std(0) + 1e-6
        for _ in range(steps):
            i = torch.randint(0, half, (512,), device=dev)
            p = net((Fx[i] - mu) / sd)
            loss = nn.functional.binary_cross_entropy_with_logits(p, Y[i]) if kind == "logit" else ((p - (Y[i] - ym) / ys) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            p = torch.cat([net((Fx[s:s + 8192] - mu) / sd) for s in range(half, n, 8192)])
        return p if kind == "logit" else p * ys + ym

    def r2(p, y):
        return (1 - ((p - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)).cpu().numpy().round(3).tolist()

    flat = V.flatten(1)
    if st["disc"] is not None:
        Y = torch.as_tensor(st["disc"], device=dev).float()
        hit = ((fit(flat, Y, "logit") > 0).float() == Y[half:]).float()
        acc = hit.mean(0).cpu().numpy()
        res["buttons_acc"] = acc.round(4).tolist(); res["buttons_min"] = float(acc.min())
        if sub is not None:
            sm = torch.as_tensor(sub[half:], device=dev)
            res["buttons_acc_settled"] = hit[sm].mean(0).cpu().numpy().round(4).tolist()
            res["settled_frac"] = float(sm.float().mean())
    pooled = nn.functional.adaptive_avg_pool2d(V, 8).flatten(1)
    if st["pos"] is not None:
        res["memory_keypoint"] = keypoint(lambda i: V[i], V.shape[1], st, n, half, dev, steps)
        res["pixels_keypoint"] = pix_kp
        res["keypoint_ratio_memory_over_pixels"] = (np.array(res["memory_keypoint"]["median_err_cm"]) /
                                                    np.maximum(pix_kp["median_err_cm"], 1e-3)).round(2).tolist()
        if pix_mlp is not None:
            res["memory_mlp_pos"] = mlp_pos(pooled, st, n, half, dev, steps)
            res["pixels_mlp_pos"] = pix_mlp
            res["mlp_pos_ratio_memory_over_pixels"] = (np.array(res["memory_mlp_pos"]["median_err_cm"]) /
                                                       np.maximum(pix_mlp["median_err_cm"], 1e-3)).round(2).tolist()
    if st["arm"] is not None:
        res["arm_mlp_r2"] = r2(fit(pooled, torch.as_tensor(st["arm"], device=dev), "mlp"), torch.as_tensor(st["arm"][half:], device=dev))
    if st["joint"] is not None:
        pj = fit(flat, torch.as_tensor(st["joint"], device=dev), "mlp")
        yj = torch.as_tensor(st["joint"][half:], device=dev)
        res["joint_mlp_r2"] = r2(pj, yj)
        if sub is not None:
            sm = torch.as_tensor(sub[half:], device=dev)
            res["joint_mlp_r2_settled"] = r2(pj[sm], yj[sm])
    return res


def mlp_pos(Fx, st, n, half, dev, steps):
    """c2 probe (see the c2 note): MLP (2 x 512) on 8 x 8-pooled features Fx (n, D) -> object xy, fitted on the first
    half, median error per object (cm) on the second half."""
    torch.manual_seed(0)
    K = st["pos"].shape[1]
    tgt = torch.as_tensor(st["pos"][..., :2].reshape(n, -1), device=dev).float()
    m_, s_ = Fx[:half].mean(0), Fx[:half].std(0) + 1e-3
    net = nn.Sequential(nn.Linear(Fx.shape[1], 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 2 * K)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    ym, ys = tgt[:half].mean(0), tgt[:half].std(0) + 1e-6
    for _ in range(steps):
        i = torch.randint(0, half, (512,), device=dev)
        loss = ((net((Fx[i] - m_) / s_) - (tgt[i] - ym) / ys) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.cat([net((Fx[s:s + 8192] - m_) / s_) for s in range(half, n, 8192)]) * ys + ym
    err = (p - tgt[half:]).view(-1, K, 2).norm(dim=-1)
    return {"median_err_cm": (err.median(0).values * 100).cpu().numpy().round(2).tolist(),
            "within_2cm": (err < 0.02).float().mean(0).cpu().numpy().round(3).tolist()}


def keypoint(feat, C, st, n, half, dev, steps):
    """keypoint probe as slowmap_probe.py: 1x1 conv -> spatial softmax per object -> affine to metres."""
    K = st["pos"].shape[1]
    tgt = torch.as_tensor(st["pos"][..., :2].reshape(n, -1), device=dev)
    H = W = feat(torch.zeros(1, dtype=torch.long, device=dev)).shape[-1]
    conv, aff = nn.Conv2d(C, K, 1).to(dev), nn.Linear(2 * K, 2 * K).to(dev)
    gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
    opt = torch.optim.Adam(list(conv.parameters()) + list(aff.parameters()), lr=3e-3)
    mu, sd = tgt[:half].mean(0), tgt[:half].std(0)

    def fwd(f):
        att = torch.softmax(conv(f).flatten(2), -1).view(len(f), K, H, W)
        return aff(torch.stack([(att * gx).sum((2, 3)), (att * gy).sum((2, 3))], -1).flatten(1)) * sd + mu

    for _ in range(steps):
        i = torch.randint(0, half, (512,), device=dev)
        loss = ((fwd(feat(i)) - tgt[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = torch.cat([fwd(feat(torch.arange(s, min(s + 4096, n), device=dev))) for s in range(half, n, 4096)])
    err = (p - tgt[half:]).view(-1, K, 2).norm(dim=-1)
    return {"median_err_cm": (err.median(0).values * 100).cpu().numpy().round(2).tolist(),
            "within_2cm": (err < 0.02).float().mean(0).cpu().numpy().round(3).tolist()}


def stability(mem, rest, starts, T, st):
    n = len(mem)
    ends = starts + T - 1
    nxt = np.r_[rest[1:] & rest[:-1], False]; nxt[ends] = False
    ch = (mem[1:] != mem[:-1]) & (mem[:-1] >= 0)                                # a known token changing (first sight excluded)
    out = {"rest_frame_fraction": float(rest.mean()),
           "token_change_per_rest_frame": float(ch[nxt[:-1]].mean()),
           "rest_frames_with_any_change": float(ch[nxt[:-1]].any(-1).mean()),
           "known_token_fraction": float((mem >= 0).mean())}
    per = periods(rest, starts, T)
    diff_ok = diff_n = same_ok = same_n = 0
    for (e1, a1, b1), (e2, a2, b2) in zip(per[:-1], per[1:]):
        if e1 != e2:
            continue
        i, j = (a1 + b1) // 2, (a2 + b2) // 2
        neq = mem[i] != mem[j]
        if state_differs(st, i, j):
            diff_n += 1; diff_ok += bool(neq.any())
        else:
            same_n += 1; same_ok += bool(~neq.any())
    out.update(rest_periods=len(per), changed_pairs=diff_n, changed_pairs_codes_differ=diff_ok / max(diff_n, 1),
               unchanged_pairs=same_n, unchanged_pairs_codes_identical=same_ok / max(same_n, 1))
    return out


def criteria(pr, ev):
    c = {}
    if "buttons_min" in pr:
        c["c1_discrete_ge_.98"] = bool(pr["buttons_min"] >= 0.98)
    if "mlp_pos_ratio_memory_over_pixels" in pr:                               # MLP probe (deviation, see c2 note)
        c["c2_complete_ratio_le_1.2"] = bool(max(pr["mlp_pos_ratio_memory_over_pixels"]) <= 1.2)
    elif "keypoint_ratio_memory_over_pixels" in pr:
        c["c2_complete_ratio_le_1.2_keypoint"] = bool(max(pr["keypoint_ratio_memory_over_pixels"]) <= 1.2)
    if "arm_mlp_r2" in pr:
        c["c3_agent_free_r2_le_.2"] = bool(max(pr["arm_mlp_r2"]) <= 0.2)
    c["c4_events_rec_ge_.9_prec_ge_.8"] = bool((ev["recall"] or 0) >= 0.9 and (ev["precision"] or 0) >= 0.8)
    return c


def run_memory(codes, agent, starts, T, k, return_confirmed=False):
    mem = np.zeros_like(codes, dtype=np.int64); changed = np.zeros(codes.shape, bool); conf = np.zeros(codes.shape, bool)
    for s0 in starts:
        mem[s0:s0 + T], changed[s0:s0 + T], conf[s0:s0 + T] = memory_rule(codes[s0:s0 + T], agent[s0:s0 + T], k, return_confirmed=True)
    return (mem, changed, conf) if return_confirmed else (mem, changed)


def panels(model, obs, mem, agent, starts, path, dev, episodes=3, frames=(0, 100, 300, 600, 900)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    with torch.no_grad():
        for e in range(min(episodes, len(starts))):
            for t in frames:
                i = starts[e] + t
                x = torch.as_tensor(np.asarray(obs[i:i + 1]), device=dev).permute(0, 3, 1, 2).float() / 255
                m = mem[i]
                v = model.fsq.values(torch.as_tensor(np.maximum(m, 0), device=dev)).T.reshape(1, -1, 16, 16)
                v = v * torch.as_tensor(m >= 0, device=dev).float().view(1, 1, 16, 16)
                dec = model.decode(v)[0]
                am = torch.as_tensor(agent[i], device=dev).float().view(1, 1, 16, 16)
                am = nn.functional.interpolate(am, scale_factor=4)[0]
                over = x[0] * (1 - 0.6 * am) + 0.6 * am * torch.tensor([1.0, 0.0, 0.0], device=dev).view(3, 1, 1)
                rows.append([x[0], over, dec, (dec - x[0]).abs().mean(0, keepdim=True).expand(3, -1, -1) * 3])
    fig, ax = plt.subplots(len(rows), 4, figsize=(8, 2 * len(rows)))
    for r, row in enumerate(rows):
        for j, (im, name) in enumerate(zip(row, ("frame", "untrusted (agent / low p)", "memory decoded", "|memory - frame| x3"))):
            ax[r, j].imshow(im.permute(1, 2, 0).clamp(0, 1).cpu().numpy()); ax[r, j].axis("off")
            if r == 0:
                ax[r, j].set_title(name, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="sm2_train.py --out directory")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--family", choices=("cube", "puzzle", "scene"), required=True)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--k", type=int, default=3, help="memory debounce: consecutive agent-free frames with the same code")
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--merge", type=int, default=5)
    ap.add_argument("--window", type=int, default=15)
    ap.add_argument("--probe-steps", type=int, default=3000)
    ap.add_argument("--fit-episodes", type=int, default=100, help="TRAIN episodes for the label-free assembly gap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    dev = a.device
    t0 = time.time()
    torch.backends.cudnn.deterministic = True
    a.out.mkdir(parents=True, exist_ok=True)
    starts, ends = episode_bounds(a.cache, "val", a.val_episodes)
    T = int(ends[0] - starts[0] + 1)
    assert np.all(ends - starts + 1 == T), "episodes of equal length expected"
    n = len(starts) * T
    obs = np.array(np.load(a.cache / "val_observations.npy", mmap_mode="r")[:n])
    is_start = np.zeros(n, bool); is_start[starts] = True
    ck = torch.load(a.run / "scene.pt", map_location=dev, weights_only=False)
    model = SceneCodes(ck["width"]).to(dev).eval(); model.load_state_dict(ck["model"])
    codes = np.zeros((n, 256), np.int64)
    with torch.no_grad():
        for s in range(0, n, 2048):
            x = torch.as_tensor(obs[s:s + 2048], device=dev).permute(0, 3, 1, 2).float() / 255
            codes[s:s + len(x)] = model.codes(x).cpu().numpy()
    pcodes = pixel_tokens(obs, n)
    bits, segp = np.load(a.run / "teacher_val.npy")[:n], np.load(a.run / "seg_val.npy")[:n]
    idx = np.arange(n)
    agent = agent_masks(idx, bits, segp, is_start, r=a.dilate)
    agent_seg = agent_masks(idx, bits, segp, is_start, r=a.dilate, teacher=False)
    teach0 = agent_masks(idx, bits, None, is_start, r=0)
    seg0 = segp > 127
    res = {"run": str(a.run), "family": a.family, "frames": n, "episodes": len(starts), "k": a.k, "dilate": a.dilate,
           "merge": a.merge, "window": a.window, "train_cfg": ck["cfg"],
           "codes_distinct_val": int(len(np.unique(codes))),
           "agent_mask": {"token_frac_union_dilated": float(agent.mean()), "token_frac_segmenter_dilated": float(agent_seg.mean()),
                          "token_frac_teacher": float(teach0.mean()), "token_frac_segmenter": float(seg0.mean()),
                          "segmenter_recall_of_teacher": float((seg0 & teach0).sum() / max(teach0.sum(), 1)),
                          "teacher_share_of_segmenter": float((seg0 & teach0).sum() / max(seg0.sum(), 1))}}
    st = privileged_state(a.cache, a.family, n)
    if st["arm"] is None and (a.cache / "val_qpos.npy").exists():                # puzzle: arm joints for c3 too
        st["arm"] = np.asarray(np.load(a.cache / "val_qpos.npy", mmap_mode="r")[:n, :6], np.float32)
    rest = rest_mask(st, starts, T)
    segs, segs_type, mv = reference_segments(a.cache, a.family, n, starts, T)
    res["reference"] = {"segments": int(len(segs)), "per_episode": float(len(segs) / len(starts)),
                        "per_type": {k: int(len(v)) for k, v in segs_type.items()}}
    last = np.full(n, -10 ** 6)                                                  # frame of the last object change
    for s0 in starts:
        lc = -10 ** 6
        for t in range(s0, s0 + T):
            if mv[t]:
                lc = t
            last[t] = lc
    settled = rest & (np.arange(n) - last >= 30)
    ofs = torch.as_tensor(obs, device=dev)                                      # uint8 on the GPU (1.2 GB for 100 episodes)
    pix_kp = pix_mlp = None
    if st["pos"] is not None:
        pix_kp = keypoint(lambda i: ofs[i].permute(0, 3, 1, 2).float().div(127.5).sub(1), 3, st, n, n // 2, dev, a.probe_steps)
        px8 = torch.cat([ofs[s:s + 4096].permute(0, 3, 1, 2).float().reshape(-1, 3, 8, 8, 8, 8).mean((3, 5)).flatten(1) / 255
                         for s in range(0, n, 4096)])                           # 8 x 8-pooled RGB, chunked
        pix_mlp = mlp_pos(px8, st, n, n // 2, dev, a.probe_steps)
        del px8
    # ceiling of the same probes on the CURRENT frame's raw-pixel tokens (mean RGB per 4 x 4 patch; contains the arm)
    cur = np.zeros((n, 3, 16, 16), np.float32)
    for s in range(0, n, 5000):
        cur[s:s + 5000] = np.asarray(obs[s:s + 5000], np.float32).reshape(-1, 16, 4, 16, 4, 3).mean((2, 4)).transpose(0, 3, 1, 2) / 255
    cur = torch.as_tensor(cur, device=dev)
    res["current_frame_pixel_probes"] = probe_suite(cur, {**st, "pos": None}, None, dev, a.probe_steps, sub=settled)
    del cur
    vals = model.fsq.values(torch.arange(5 ** 6, device=dev))                   # (15625, 6)
    feats = {"learned": lambda m: vals[torch.as_tensor(np.maximum(m, 0), device=dev)].permute(0, 2, 1).reshape(len(m), 6, 16, 16),
             "pixel": lambda m: torch.as_tensor(np.stack([(np.maximum(m, 0) // 64) % 8, (np.maximum(m, 0) // 8) % 8, np.maximum(m, 0) % 8], 1)
                                                .astype(np.float32) / 7, device=dev).reshape(len(m), 3, 16, 16)}
    see = {}
    for sp in ("learned", "pixel"):
        f = a.run / f"seethru_{sp}.pt"
        if f.exists():
            sk = torch.load(f, map_location=dev, weights_only=False)
            see[sp] = SeeThrough(sk["nd"], sk["nl"], sk["width"]).to(dev).eval(); see[sp].load_state_dict(sk["model"])

    def codes_of(space, x_u8):
        if space == "pixel":
            return pixel_codes(x_u8)
        with torch.no_grad():
            return model.codes(torch.as_tensor(x_u8, device=dev).permute(0, 3, 1, 2).float() / 255).cpu().numpy()

    def see_of(space, x_u8):
        out = [see_codes(see[space], torch.as_tensor(x_u8[s:s + 1024], device=dev).permute(0, 3, 1, 2).float() / 255)
               for s in range(0, len(x_u8), 1024)]
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])

    variants = [("learned", "learned", "mask"), ("pixel_tokens", "pixel", "mask")]
    variants += [(f"see_{sp}", sp, "see") for sp in ("learned", "pixel") if sp in see]
    variants += [(f"conf_{sp}", sp, "conf") for sp in ("learned", "pixel") if sp in see]
    # TRAIN episodes (label-free) for the assembly gap of each variant
    tterm = np.load(a.cache / "train_terminals.npy"); tends = np.flatnonzero(tterm)[:a.fit_episodes]; tstarts = np.r_[0, tends[:-1] + 1]
    tobs_m = np.load(a.cache / "train_observations.npy", mmap_mode="r")
    tbits, tsegp = np.load(a.run / "teacher_train.npy", mmap_mode="r"), np.load(a.run / "seg_train.npy", mmap_mode="r")
    tis_start = np.zeros(int(tends[-1] + 1), bool); tis_start[tstarts] = True
    res["assembly_gap"] = {}
    for name, space, trust in variants:
        gaps = []
        for s0, e0 in zip(tstarts, tends):
            idx_t = np.arange(s0, e0 + 1)
            x = np.asarray(tobs_m[s0:e0 + 1])
            if trust == "mask":
                c_t, ag_t = codes_of(space, x), agent_masks(idx_t, tbits, tsegp, tis_start, r=a.dilate)
            else:
                c_t, p_t = see_of(space, x); ag_t = p_t <= 0.5
            mf_t, ch_t = memory_rule(c_t, ag_t, a.k)
            if trust == "conf":
                mc_t, chc_t, cf_t = memory_rule(c_t, ag_t | agent_masks(idx_t, tbits, tsegp, tis_start, r=a.dilate), a.k, return_confirmed=True)
                ch_t = backdate(chc_t, mc_t, cf_t, mf_t, np.array([0]), len(idx_t))
            gaps.append(adjacent_gaps(ch_t, np.array([0]), len(idx_t)))
        gaps = np.concatenate(gaps)
        thr, sep, frac = two_means_threshold(np.log10(gaps)) if len(gaps) > 10 else (np.log10(30), 0.0, 0.0)
        gap = float(10 ** thr) if sep > 2.0 else 30.0
        res["assembly_gap"][name] = {"gaps": int(len(gaps)), "log10_split": thr, "ashman_D": sep, "gap_frames": gap}
        if trust == "mask":
            cds, ag = (codes if space == "learned" else pcodes), agent
            conf_frac = None
        else:
            cds, pr = see_of(space, obs); ag = pr <= 0.5
            conf_frac = float((~ag).mean())
        mem, changed = run_memory(cds, ag, starts, T, a.k)
        mem_ev = mem
        if trust == "conf":                                                     # events: confirmed after the agent left, backdated
            mem_ev, ch_c, cf_c = run_memory(cds, ag | agent, starts, T, a.k, return_confirmed=True)
            changed = backdate(ch_c, mem_ev, cf_c, mem, starts, T)
            r_bd = {"confirmed_changes": int(ch_c.sum()), "moved_earlier_frac": float((ch_c & ~changed).sum() / max(ch_c.sum(), 1))}
        on = changed.any(1)
        ev = merge_onsets(on, starts, T, a.merge)
        asm = assemble_events(changed, starts, T, gap)
        r = {"trusted_token_frac": float((~ag).mean()), "seethru_conf_frac": conf_frac,
             "event_frames_per_1000": float(1000 * on.mean()), "events": int(len(ev)), "events_per_episode": float(len(ev) / len(starts)),
             "changed_tokens_per_event_frame_median": float(np.median(changed[on].sum(1))) if on.any() else None,
             "events_w15": score_events(ev[:, 0], segs, a.window), "events_w30": score_events(ev[:, 0], segs, 2 * a.window),
             "recall_per_type_w15_w30": {k: [score_events(ev[:, 0], v, a.window)["recall"], score_events(ev[:, 0], v, 2 * a.window)["recall"]]
                                         for k, v in segs_type.items()},
             "lag": event_lag(ev[:, 0], segs, a.window), "stability": stability(mem_ev, rest, starts, T, st),
             "state_stability": stability(mem, rest, starts, T, st) if trust == "conf" else None,
             "backdate": r_bd if trust == "conf" else None,
             "assembled": {"gap_frames": gap, "events_per_episode": float(len(asm) / len(starts)),
                           "tokens_per_event_median": float(np.median(asm[:, 2])) if len(asm) else None,
                           "w15": score_events(asm[:, 0], segs, a.window), "w30": score_events(asm[:, 0], segs, 2 * a.window),
                           "spans_w15": score_spans(asm, segs, a.window), "spans_w30": score_spans(asm, segs, 2 * a.window),
                           "lag": event_lag(asm[:, 0], segs, a.window)}}
        V = torch.cat([feats[space](mem[s:s + 8192]) for s in range(0, n, 8192)])
        V = V * torch.as_tensor(mem >= 0, device=dev).view(n, 1, 16, 16).float()
        r["probes"] = probe_suite(V.float(), st, pix_kp, dev, a.probe_steps, sub=settled, pix_mlp=pix_mlp)
        r["criteria"] = criteria(r["probes"], r["events_w15"])
        r["criteria_assembled_events"] = criteria(r["probes"], r["assembled"]["w15"])
        res[name] = r
        print(name, json.dumps({k: v for k, v in r.items() if k != "probes"}), "\nPROBES", json.dumps(r["probes"]), flush=True)
        if space == "learned":
            E = a.out / "export" / "val" / name; E.mkdir(parents=True, exist_ok=True)
            np.save(E / "mem.npy", mem.astype(np.int32)); np.save(E / "changed.npy", np.packbits(changed, axis=1))
            np.save(E / "untrusted.npy", np.packbits(ag, axis=1)); np.save(E / "codes.npy", np.asarray(cds, np.int32))
            np.save(E / "starts.npy", starts); np.save(E / "assembled_events.npy", asm)
            try:
                panels(model, obs, mem, ag, starts, a.out / f"panels_{name}.png", dev)
            except Exception as ex:                                             # figure only; never blocks the numbers
                res[f"panels_error_{name}"] = repr(ex)
        del V
    # agent mask itself should track the arm (high R^2 here = the mask follows the agent)
    if st["arm"] is not None:
        Am = torch.as_tensor(agent, device=dev).float().view(n, 1, 16, 16)
        res["agent_mask"]["arm_r2_from_mask"] = probe_suite(Am, {"disc": None, "pos": None, "joint": None, "arm": st["arm"]}, None, dev,
                                                            a.probe_steps)["arm_mlp_r2"]
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (a.out / "diag.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k not in ("train_cfg", "learned", "pixel_tokens", "see_learned", "see_pixel",
                                                              "conf_learned", "conf_pixel")}), flush=True)


if __name__ == "__main__":
    main()
