#!/usr/bin/env python3
"""Method front-end training (stages of components 1-6, method/README.md). Code identical to scripts/sm2_train.py
(sha256 feea3560aa527970...) except the import of frontend.py.

Train scene memory v2 (sm2_model.py) from offline play: observations + actions only. Stages, run in order, all into
--out; the same hyper-parameters for every family:

  agent    AgentNet: next frame from (x_t, a_t, a_{t-1}) with action dropout (flag 0 = action-free prediction).
  label    teacher (sm2_model.contingent): per pair (t, t+1) and token, the pixels change and the action explains most of
           the change (R = clip(1 - e_action / e_free, 0, 1) above its 2-means split over changing tokens of a TRAIN
           sample, 0.5 if not bimodal), grown through changing tokens (--teacher-grow 1; the first scene run used 0);
           bits for every TRAIN and VAL pair -> teacher_{split}.npy (packbits).
  seg      Segmenter: one frame -> per-token agent logit, BCE on frame labels (pair (t-1, t) | pair (t, t+1)).
  segpred  segmenter probabilities for every TRAIN and VAL frame -> seg_{split}.npy (uint8, p x 255).
  scene    SceneCodes: per-patch FSQ codes + decoder; reconstruction weighted by per-pixel temporal std over the clip
           (as sm_train.py) and counted only outside the dilated agent mask (teacher | segmenter).

  stargets see-through targets (sm2_model.see_targets) for every TRAIN and VAL frame and token, in both code spaces
           (learned SceneCodes 5^6 and the raw-pixel control 8^3): memory with the agent mask (dilate --dilate), then
           clean anchors + agent-covered frames of visits after which the same code is confirmed -> see_{space}_{split}.npy
           (uint16, 65535 = no target).
  seethru  SeeThrough for --space learned | pixel: cross-entropy over the code digits on those targets.

Long stages (agent, seg, scene, seethru) save resume.pt every --ckpt-every steps and continue from it with --resume.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from frontend import (AgentNet, SceneCodes, SeeThrough, Segmenter, action_share, agent_masks, contingent, grow_tokens,
                       memory_rule, pixel_codes, see_codes, see_targets, token_errors, two_means_threshold)

SPACES = {"learned": (6, 5), "pixel": (3, 8)}                                   # digits, levels


class Frames:
    """(N, 64, 64, 3) uint8 frames, memory-mapped or in RAM. A memory map is reopened every `remap` reads, so the pages
    read so far leave the working set (on Windows it otherwise grows to the file size)."""

    def __init__(self, path, n, ram=False, remap=200):
        self.path, self.n, self.remap, self.k, self.ram = path, n, remap, 0, ram
        src = np.load(path, mmap_mode="r")
        self.a = np.array(src[:n]) if ram else src[:n]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if not self.ram:
            self.k += 1
            if self.k % self.remap == 0:
                self.a = np.load(self.path, mmap_mode="r")[:self.n]
        return np.array(self.a[idx])


def load_split(cache: Path, split: str, episodes: int, ram=False):
    term = np.load(cache / f"{split}_terminals.npy")
    ends = np.flatnonzero(term)[:episodes]
    n = int(ends[-1] + 1)
    starts = np.r_[0, ends[:-1] + 1]
    obs = Frames(cache / f"{split}_observations.npy", n, ram=ram)
    act = np.asarray(np.load(cache / f"{split}_actions.npy", mmap_mode="r")[:n], np.float32)
    is_start = np.zeros(n, bool); is_start[starts] = True
    is_end = np.zeros(n, bool); is_end[ends] = True
    return obs, act, starts, ends, is_start, is_end


def to_t(x, dev):
    return torch.as_tensor(x, device=dev).permute(0, 3, 1, 2).float().div_(255)


def prev_action(act, t, is_start):
    ap = act[np.maximum(t - 1, 0)].copy()
    ap[is_start[t]] = 0
    return ap


def schedule(opt, steps, warm=500):
    return torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)))


class Resume:
    def __init__(self, path: Path, every: int):
        self.path, self.every = path, every

    def load(self, modules, opt, sched, rng, dev):
        if not self.path.exists():
            return 0, []
        ck = torch.load(self.path, map_location=dev, weights_only=False)
        for k, m in modules.items():
            m.load_state_dict(ck[k])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["np_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        if dev == "cuda":
            torch.cuda.set_rng_state(ck["cuda_rng"].cpu())
        print(f"RESUMED {self.path.name} from step {ck['step']}", flush=True)
        return ck["step"], ck["log"]

    def maybe_save(self, step, steps, modules, opt, sched, rng, log, dev):
        if not self.every or (step + 1) % self.every or step + 1 >= steps:
            return
        d = {k: m.state_dict() for k, m in modules.items()}
        d.update(opt=opt.state_dict(), sched=sched.state_dict(), step=step + 1, log=log, np_rng=rng.bit_generator.state,
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state() if dev == "cuda" else None)
        torch.save(d, self.path.with_suffix(".tmp"))
        self.path.with_suffix(".tmp").replace(self.path)                  # atomic


def amp_ctx(dev):
    return torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else torch.autocast("cpu", enabled=False)


# ---------------------------------------------------------------------------------------------------------------- agent
def stage_agent(a, dev):
    obs, act, st, en, is_start, is_end = load_split(a.cache, "train", a.episodes)
    vobs, vact, vst, ven, vstart, vend = load_split(a.cache, "val", a.val_episodes, ram=True)
    rng = np.random.default_rng(a.seed); torch.manual_seed(a.seed)
    net = AgentNet(act.shape[1], a.agent_width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.agent_lr, weight_decay=1e-4)
    sched = schedule(opt, a.agent_steps)
    res = Resume(a.out / "resume_agent.pt", a.ckpt_every)
    start, log = res.load({"net": net}, opt, sched, rng, dev) if a.resume else (0, [])
    vr = np.random.default_rng(12345)
    ve = vr.integers(len(vst), size=512); vt = vst[ve] + vr.integers(0, ven[ve] - vst[ve], size=512)
    t_last = time.time()
    for step in range(start, a.agent_steps):
        net.train()
        e = rng.integers(len(st), size=a.batch); t = st[e] + rng.integers(0, en[e] - st[e], size=a.batch)   # t+1 in episode
        x, xn = to_t(obs[t], dev), to_t(obs[t + 1], dev)
        at, ap = torch.as_tensor(act[t], device=dev), torch.as_tensor(prev_action(act, t, is_start), device=dev)
        flag = torch.as_tensor(rng.random(a.batch) > a.p_drop, device=dev)
        with amp_ctx(dev):
            pred = net(x, at, ap, flag)
        err = (pred.float() - xn).pow(2).mean((1, 2, 3))
        loss = err.mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0 or step == a.agent_steps - 1:
            row = {"step": step, "loss": float(loss), "loss_action": float(err[flag].mean()) if flag.any() else None,
                   "loss_free": float(err[~flag].mean()) if (~flag).any() else None,
                   "sec_per_step": (time.time() - t_last) / (1 if step == start else 500)}
            if step % 2000 == 0 or step == a.agent_steps - 1:
                net.eval()
                ef, ea, ch = errors_for(net, vobs, vact, vstart, vt, dev)
                top = ch > np.percentile(ch, 95)                                  # where the scene changes most
                row.update(val_err_free=float(ef.mean()), val_err_action=float(ea.mean()),
                           val_R_median_top5pct_change=float(np.median(action_share(ef[top], ea[top]))))
            t_last = time.time(); log.append(row); print(json.dumps(row), flush=True)
        res.maybe_save(step, a.agent_steps, {"net": net}, opt, sched, rng, log, dev)
    torch.save({"net": net.state_dict(), "act_dim": act.shape[1], "width": a.agent_width, "cfg": cfg_of(a)}, a.out / "agent_net.pt")
    (a.out / "agent_log.json").write_text(json.dumps(log, indent=1) + "\n")


def load_agent(a, dev):
    ck = torch.load(a.out / "agent_net.pt", map_location=dev, weights_only=False)
    net = AgentNet(ck["act_dim"], ck["width"]).to(dev).eval(); net.load_state_dict(ck["net"])
    return net


def errors_for(net, obs, act, is_start, t, dev, chunk=512):
    out = []
    for i in range(0, len(t), chunk):
        tt = t[i:i + chunk]
        with amp_ctx(dev):
            ef, ea, ch = token_errors(net, to_t(obs[tt], dev), to_t(obs[tt + 1], dev), torch.as_tensor(act[tt], device=dev),
                                  torch.as_tensor(prev_action(act, tt, is_start), device=dev))
        out.append((ef.cpu().numpy(), ea.cpu().numpy(), ch.cpu().numpy()))
    return tuple(np.concatenate([o[j] for o in out]) for j in range(3))


# ---------------------------------------------------------------------------------------------------------------- label
def stage_label(a, dev):
    net = load_agent(a, dev)
    rep = {}
    obs, act, st, en, is_start, is_end = load_split(a.cache, "train", a.episodes)
    rng = np.random.default_rng(a.seed + 1)
    e = rng.integers(len(st), size=a.label_fit_pairs); t = np.sort(st[e] + rng.integers(0, en[e] - st[e], size=a.label_fit_pairs))
    ef, ea, ch = (v.ravel() for v in errors_for(net, obs, act, is_start, t, dev))
    chg = ch > 1e-9
    R = action_share(ef[chg], ea[chg])
    r_thr, r_sep, r_frac = two_means_threshold(R)
    if r_sep <= 2.0:                                                              # not bimodal: majority rule
        r_thr = 0.5
    rep["fit"] = {"pairs": int(len(t)), "changing_token_frac": float(chg.mean()), "thr_R": r_thr, "R_ashman_D": r_sep,
                  "R_quantiles_changing": np.percentile(R, [10, 25, 50, 75, 90]).round(3).tolist(),
                  "contingent_token_frac": float(contingent(ef, ea, ch, r_thr).mean())}
    print("FIT", json.dumps(rep["fit"]), flush=True)
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        obs, act, st, en, is_start, is_end = load_split(a.cache, split, n_ep)
        N = len(obs)
        bits = np.zeros((N, 32), np.uint8)
        t0 = time.time()
        for s in range(0, N, a.chunk):
            t = np.arange(s, min(s + a.chunk, N))
            t = t[~is_end[t]]                                                     # pair (t, t+1) inside the episode
            if len(t):
                ef, ea, ch = errors_for(net, obs, act, is_start, t, dev)
                c = contingent(ef, ea, ch, r_thr)
                if a.teacher_grow:
                    c = grow_tokens(c, ch > 1e-9)
                bits[t] = np.packbits(c, axis=1)
        np.save(a.out / f"teacher_{split}.npy", bits)
        frac_tok = float(np.unpackbits(bits[:min(N, 200_000)], axis=1).mean())
        rep[split] = {"frames": int(N), "token_frac_contingent": frac_tok, "seconds": round(time.time() - t0, 1)}
        print(split, json.dumps(rep[split]), flush=True)
    rep["teacher_grow"] = a.teacher_grow
    (a.out / "teacher.json").write_text(json.dumps(rep, indent=1) + "\n")


# ------------------------------------------------------------------------------------------------------------------ seg
def stage_seg(a, dev):
    obs, act, st, en, is_start, is_end = load_split(a.cache, "train", a.episodes)
    vobs, vact, vst, ven, vstart, vend = load_split(a.cache, "val", a.val_episodes, ram=True)
    bits, vbits = np.load(a.out / "teacher_train.npy"), np.load(a.out / "teacher_val.npy")
    rng = np.random.default_rng(a.seed + 2); torch.manual_seed(a.seed + 2)
    seg = Segmenter(a.seg_width).to(dev)
    opt = torch.optim.AdamW(seg.parameters(), lr=a.seg_lr, weight_decay=1e-4)
    sched = schedule(opt, a.seg_steps)
    res = Resume(a.out / "resume_seg.pt", a.ckpt_every)
    start, log = res.load({"seg": seg}, opt, sched, rng, dev) if a.resume else (0, [])
    vt = np.random.default_rng(12345).integers(0, len(vobs), size=2048)
    vy = agent_masks(vt, vbits, None, vstart, r=0)
    t_last = time.time()
    for step in range(start, a.seg_steps):
        seg.train()
        t = rng.integers(0, len(obs), size=a.batch)
        y = torch.as_tensor(agent_masks(t, bits, None, is_start, r=0), device=dev).float()
        with amp_ctx(dev):
            logit = seg(to_t(obs[t], dev))
        loss = F.binary_cross_entropy_with_logits(logit.float(), y)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(seg.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0 or step == a.seg_steps - 1:
            seg.eval()
            with torch.no_grad(), amp_ctx(dev):
                p = torch.cat([seg(to_t(vobs[vt[i:i + 256]], dev)).float() for i in range(0, 2048, 256)]).cpu().numpy() > 0
            tp = (p & vy).sum()
            row = {"step": step, "loss": float(loss), "val_pos_frac_label": float(vy.mean()), "val_pos_frac_pred": float(p.mean()),
                   "val_recall": float(tp / max(vy.sum(), 1)), "val_precision": float(tp / max(p.sum(), 1)),
                   "sec_per_step": (time.time() - t_last) / (1 if step == start else 500)}
            t_last = time.time(); log.append(row); print(json.dumps(row), flush=True)
        res.maybe_save(step, a.seg_steps, {"seg": seg}, opt, sched, rng, log, dev)
    torch.save({"seg": seg.state_dict(), "width": a.seg_width, "cfg": cfg_of(a)}, a.out / "segmenter.pt")
    (a.out / "seg_log.json").write_text(json.dumps(log, indent=1) + "\n")


def load_seg(a, dev):
    ck = torch.load(a.out / "segmenter.pt", map_location=dev, weights_only=False)
    seg = Segmenter(ck["width"]).to(dev).eval(); seg.load_state_dict(ck["seg"])
    return seg


def stage_segpred(a, dev):
    seg = load_seg(a, dev)
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        obs = load_split(a.cache, split, n_ep)[0]
        out = np.zeros((len(obs), 256), np.uint8)
        with torch.no_grad(), amp_ctx(dev):
            for s in range(0, len(obs), a.chunk):
                p = torch.sigmoid(seg(to_t(obs[s:s + a.chunk], dev)).float())
                out[s:s + len(p)] = (p * 255).round().byte().cpu().numpy()
        np.save(a.out / f"seg_{split}.npy", out)
        print(split, {"frames": len(obs), "token_frac_p_gt_half": float((out[:200_000] > 127).mean())}, flush=True)


# ---------------------------------------------------------------------------------------------------------------- scene
def clip_batch(obs, bits, segp, is_start, first, T, dilate, dev):
    idx = (first[:, None] + np.arange(T)).ravel()
    x = to_t(obs[idx], dev)                                                       # (B*T, 3, 64, 64)
    m = agent_masks(idx, bits, segp, is_start, r=dilate)                          # (B*T, 256)
    vis = torch.as_tensor(~m, device=dev).float().view(-1, 1, 16, 16)
    vis = F.interpolate(vis, scale_factor=4, mode="nearest")                      # (B*T, 1, 64, 64)
    return x, vis


def scene_loss(model, x, vis, B, T, kappa):
    q = model.encode(x)
    xh = model.decode(q)
    xc = x.view(B, T, 3, 64, 64)
    std = xc.std(1, keepdim=True).mean(2, keepdim=True)                           # (B, 1, 1, 64, 64), as sm_train.py
    w = (1 + kappa * std / std.amax((3, 4), keepdim=True).clamp_min(1e-6)).expand(B, T, 1, 64, 64).reshape(B * T, 1, 64, 64)
    e = (xh.float() - x).pow(2).mean(1, keepdim=True)
    return (w * vis * e).sum() / (w * vis).sum().clamp_min(1), q


def stage_scene(a, dev):
    obs, act, st, en, is_start, is_end = load_split(a.cache, "train", a.episodes)
    vobs, vact, vst, ven, vstart, vend = load_split(a.cache, "val", a.val_episodes, ram=True)
    bits, vbits = np.load(a.out / "teacher_train.npy"), np.load(a.out / "teacher_val.npy")
    segp, vsegp = np.load(a.out / "seg_train.npy"), np.load(a.out / "seg_val.npy")             # 256 MB per 1M frames
    rng = np.random.default_rng(a.seed + 3); torch.manual_seed(a.seed + 3)
    model = SceneCodes(a.scene_width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.scene_lr, weight_decay=1e-4)
    sched = schedule(opt, a.scene_steps)
    res = Resume(a.out / "resume_scene.pt", a.ckpt_every)
    start, log = res.load({"model": model}, opt, sched, rng, dev) if a.resume else (0, [])
    T, B = a.clip, a.clips
    vr = np.random.default_rng(12345); ve = vr.integers(len(vst), size=32)
    vfirst = vst[ve] + vr.integers(0, ven[ve] - vst[ve] + 2 - T, size=32)
    t_last = time.time()
    for step in range(start, a.scene_steps):
        model.train()
        e = rng.integers(len(st), size=B); first = st[e] + rng.integers(0, en[e] - st[e] + 2 - T, size=B)
        x, vis = clip_batch(obs, bits, segp, is_start, first, T, a.dilate, dev)
        with amp_ctx(dev):
            loss, q = scene_loss(model, x, vis, B, T, a.kappa)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0 or step == a.scene_steps - 1:
            row = {"step": step, "loss": float(loss), "visible_frac": float(vis.mean()),
                   "codes_used": int(torch.unique(model.fsq.index(q.detach().permute(0, 2, 3, 1).float())).numel()),
                   "sec_per_step": (time.time() - t_last) / (1 if step == start else 500)}
            if step % 2000 == 0 or step == a.scene_steps - 1:
                model.eval(); vl = []
                with torch.no_grad(), amp_ctx(dev):
                    for i in range(0, 32, 8):
                        xv, visv = clip_batch(vobs, vbits, vsegp, vstart, vfirst[i:i + 8], T, a.dilate, dev)
                        vl.append(float(scene_loss(model, xv, visv, 8, T, a.kappa)[0]))
                row["val_loss"] = float(np.mean(vl))
            t_last = time.time(); log.append(row); print(json.dumps(row), flush=True)
        res.maybe_save(step, a.scene_steps, {"model": model}, opt, sched, rng, log, dev)
    torch.save({"model": model.state_dict(), "width": a.scene_width, "cfg": cfg_of(a)}, a.out / "scene.pt")
    (a.out / "scene_log.json").write_text(json.dumps(log, indent=1) + "\n")


def load_scene(a, dev):
    ck = torch.load(a.out / "scene.pt", map_location=dev, weights_only=False)
    model = SceneCodes(ck["width"]).to(dev).eval(); model.load_state_dict(ck["model"])
    return model


def frame_codes(space, scene, x_u8, dev):
    if space == "pixel":
        return pixel_codes(x_u8)
    with torch.no_grad():
        return scene.codes(to_t(x_u8, dev)).cpu().numpy()


def stage_stargets(a, dev):
    scene = load_scene(a, dev)
    rep = {}
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        obs, act, st, en, is_start, is_end = load_split(a.cache, split, n_ep)
        bits, segp = np.load(a.out / f"teacher_{split}.npy"), np.load(a.out / f"seg_{split}.npy")
        outs = {sp: np.lib.format.open_memmap(a.out / f"see_{sp}_{split}.npy", "w+", np.uint16, (len(obs), 256)) for sp in SPACES}
        outs2 = {sp: np.lib.format.open_memmap(a.out / f"see2_{sp}_{split}.npy", "w+", np.uint16, (len(obs), 256)) for sp in SPACES} if a.partial else {}
        cnt = {sp: np.zeros(4, np.int64) for sp in SPACES}                       # clean, behind, agent-covered, during (set) tokens
        t0 = time.time()
        for s0, e0 in zip(st, en):
            idx = np.arange(s0, e0 + 1)
            x = obs[idx]
            agent = agent_masks(idx, bits, segp, is_start, r=a.dilate)
            for sp in SPACES:
                codes = frame_codes(sp, scene, x, dev)
                mem, _, conf = memory_rule(codes, agent, a.k, return_confirmed=True)
                y = see_targets(codes, agent, mem, conf, partial=a.partial)
                y2 = None
                if a.partial:
                    y, y2 = y
                    outs2[sp][idx] = np.where(y2 >= 0, y2, 65535).astype(np.uint16)
                outs[sp][idx] = np.where(y >= 0, y, 65535).astype(np.uint16)
                single = (y >= 0) & (y2 < 0) if y2 is not None else (y >= 0)
                cnt[sp] += [int((single & ~agent).sum()), int((single & agent).sum()), int(agent.sum()),
                            int((y2 >= 0).sum()) if y2 is not None else 0]
        for sp in SPACES:
            outs[sp].flush()
            if a.partial:
                outs2[sp].flush()
            c, b, g, d = cnt[sp]
            rep[f"{sp}_{split}"] = {"token_frames": int(len(obs) * 256), "clean_targets": int(c), "behind_targets": int(b),
                                    "agent_covered": int(g), "behind_share_of_covered": float(b / max(g, 1)),
                                    "set_targets": int(d), "set_share_of_covered": float(d / max(g, 1))}
        print(split, json.dumps({k: v for k, v in rep.items() if k.endswith(split)}), f"{time.time() - t0:.0f}s", flush=True)
    (a.out / "see_targets.json").write_text(json.dumps(rep, indent=1) + "\n")


def stage_seethru(a, dev):
    nd, nl = SPACES[a.space]
    obs, act, st, en, is_start, is_end = load_split(a.cache, "train", a.episodes)
    vobs, vact, vst, ven, vstart, vend = load_split(a.cache, "val", a.val_episodes, ram=True)
    Y = np.load(a.out / f"see_{a.space}_train.npy", mmap_mode="r")
    vY = np.load(a.out / f"see_{a.space}_val.npy")
    Y2 = np.load(a.out / f"see2_{a.space}_train.npy", mmap_mode="r") if a.partial else None
    vY2 = np.load(a.out / f"see2_{a.space}_val.npy") if a.partial else None
    vbits, vsegp = np.load(a.out / "teacher_val.npy"), np.load(a.out / "seg_val.npy")
    rng = np.random.default_rng(a.seed + 4); torch.manual_seed(a.seed + 4)
    model = SeeThrough(nd, nl, a.see_width).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.see_lr, weight_decay=1e-4)
    sched = schedule(opt, a.see_steps)
    res = Resume(a.out / f"resume_seethru_{a.space}.pt", a.ckpt_every)
    start, log = res.load({"model": model}, opt, sched, rng, dev) if a.resume else (0, [])
    pw = torch.tensor([nl ** j for j in range(nd)], device=dev)
    vt = np.random.default_rng(12345).integers(0, len(vobs), size=2048)
    vy, vag = vY[vt].astype(np.int64), agent_masks(vt, vbits, vsegp, vstart, r=a.dilate)
    t_last = time.time()
    for step in range(start, a.see_steps):
        model.train()
        t = np.sort(rng.integers(0, len(obs), size=a.batch))
        y = torch.as_tensor(np.asarray(Y[t]).astype(np.int64), device=dev)              # (B, 256), 65535 = none
        with amp_ctx(dev):
            logit = model(to_t(obs[t], dev))
        ok = y != 65535
        dig = (y[..., None] // pw) % nl                                                  # (B, 256, nd)
        if Y2 is None:
            loss = F.cross_entropy(logit.float()[ok].reshape(-1, nl), dig[ok].reshape(-1))
        else:                                                                            # set targets: -log(p(y) + p(y2))
            y2 = torch.as_tensor(np.asarray(Y2[t]).astype(np.int64), device=dev)
            two = ok & (y2 != 65535)
            lp = torch.log_softmax(logit.float(), -1)                                    # (B, 256, nd, nl)
            lp1 = lp.gather(-1, (dig % nl)[..., None]).squeeze(-1).sum(-1)               # log p(y) (B, 256)
            lp2 = lp.gather(-1, ((y2[..., None] // pw) % nl)[..., None]).squeeze(-1).sum(-1)
            nll = torch.where(two, -torch.logaddexp(lp1, lp2), -lp1)
            loss = nll[ok].sum() / (ok.sum().clamp_min(1) * nd)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0 or step == a.see_steps - 1:
            row = {"step": step, "loss": float(loss), "sec_per_step": (time.time() - t_last) / (1 if step == start else 500)}
            if step % 2000 == 0 or step == a.see_steps - 1:
                model.eval()
                pc, pp = [], []
                for i in range(0, 2048, 256):
                    c, p_ = see_codes(model, to_t(vobs[vt[i:i + 256]], dev)); pc.append(c); pp.append(p_)
                pc, pp = np.concatenate(pc), np.concatenate(pp)
                has = vy != 65535
                if vY2 is not None:
                    vy2 = vY2[vt].astype(np.int64); pair = has & (vy2 != 65535)
                    row["val_set_acc"] = float(((pc[pair] == vy[pair]) | (pc[pair] == vy2[pair])).mean()) if pair.any() else None
                    row["val_set_conf_gt_half"] = float((pp[pair] > 0.5).mean()) if pair.any() else None
                    has = has & ~pair
                for name, m in (("clean", has & ~vag), ("behind", has & vag), ("covered_no_target", (vy == 65535) & vag)):
                    row[f"val_{name}_acc"] = float((pc[m] == vy[m]).mean()) if m.any() else None
                    row[f"val_{name}_conf_gt_half"] = float((pp[m] > 0.5).mean()) if m.any() else None
                    row[f"val_{name}_acc_if_conf"] = float((pc[m & (pp > 0.5)] == vy[m & (pp > 0.5)]).mean()) if (m & (pp > 0.5)).any() else None
            t_last = time.time(); log.append(row); print(json.dumps(row), flush=True)
        res.maybe_save(step, a.see_steps, {"model": model}, opt, sched, rng, log, dev)
    torch.save({"model": model.state_dict(), "space": a.space, "nd": nd, "nl": nl, "width": a.see_width, "cfg": cfg_of(a)},
               a.out / f"seethru_{a.space}.pt")
    (a.out / f"seethru_{a.space}_log.json").write_text(json.dumps(log, indent=1) + "\n")


def cfg_of(a):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("agent", "label", "seg", "segpred", "scene", "stargets", "seethru"), required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64, help="frames (pairs) per step for agent and seg")
    ap.add_argument("--agent-width", type=int, default=32)
    ap.add_argument("--agent-steps", type=int, default=10000)
    ap.add_argument("--agent-lr", type=float, default=5e-4)
    ap.add_argument("--p-drop", type=float, default=0.5, help="action dropout: share of pairs trained action-free")
    ap.add_argument("--label-fit-pairs", type=int, default=20000)
    ap.add_argument("--teacher-grow", type=int, default=1, help="grow contingent tokens through changing tokens (1) or not (0)")
    ap.add_argument("--chunk", type=int, default=1024, help="frames per pass when labelling / predicting")
    ap.add_argument("--seg-width", type=int, default=64)
    ap.add_argument("--seg-steps", type=int, default=5000)
    ap.add_argument("--seg-lr", type=float, default=3e-4)
    ap.add_argument("--scene-width", type=int, default=128)
    ap.add_argument("--scene-steps", type=int, default=15000)
    ap.add_argument("--scene-lr", type=float, default=3e-4)
    ap.add_argument("--clips", type=int, default=16)
    ap.add_argument("--clip", type=int, default=8)
    ap.add_argument("--kappa", type=float, default=4.0)
    ap.add_argument("--dilate", type=int, default=1, help="agent mask dilation in tokens")
    ap.add_argument("--k", type=int, default=3, help="memory debounce (stargets)")
    ap.add_argument("--space", choices=tuple(SPACES), default="learned", help="code space of the seethru stage")
    ap.add_argument("--partial", action="store_true",
                    help="stargets / seethru: agent-covered frames of a visit that changed the token get the set target {before, after}")
    ap.add_argument("--see-width", type=int, default=128)
    ap.add_argument("--see-steps", type=int, default=10000)
    ap.add_argument("--see-lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("GPU training runs under sbatch")
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    {"agent": stage_agent, "label": stage_label, "seg": stage_seg, "segpred": stage_segpred, "scene": stage_scene,
     "stargets": stage_stargets, "seethru": stage_seethru}[a.stage](a, a.device)
    print(f"DONE {a.stage} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
