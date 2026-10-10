#!/usr/bin/env python3
"""Component 16 (method/README.md): one event-conditioned pixel skill for every family. Port of
docs/generic_state_20261007/base_source/u_skill.py (pixel path; see SOURCES in method/README.md), same recipe:
pi(a | two frames, event) with event = (entity e, its current state, its target state): FiLM conditioning on [entity
embedding, MLP(current pos, target pos, current app, target app)]. Segments from events.py: from the end of the previous
event to the end of this one plus `release` frames (clipped at the episode end), hindsight labels (e, before[e], after[e]);
knocks (several entities moved) excluded unless --include-side-effects. CNN on two frames (gap 2) with history dropout .5 (two frames alone let behaviour cloning copy the ongoing motion),
8-action chunks, shift augmentation; best validation checkpoint kept (skill_best.pt) and the final one (skill.pt).
Differences from u_skill: appearance has A values (u_skill: 3, RGB), so the conditioning MLP takes 4 + 2A inputs; frames are
memory-mapped (utils.Frames) instead of copied to RAM; the state-vector skill of the STATE track is not part of the method.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np

from skill_segments import eligible_segment, segment_frames
from utils import Frames, save_json


def make_skill(K: int, A: int, chunk: int = 8, act_dim: int = 5, width: int = 32, hidden: int = 512, cond: str = "full",
               arch: str = "cnn"):
    import torch
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                             nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(), nn.MaxPool2d(2))

    def maps(px, cur, tgt, shift):
        g = torch.arange(64, device=px.device, dtype=torch.float32)
        out = []
        for st in (cur, tgt):
            uv = st[:, :2].float() + (shift.float() if shift is not None else 0.0)
            out.append(torch.exp(-((g[None, None, :] - uv[:, 0, None, None]) ** 2 + (g[None, :, None] - uv[:, 1, None, None]) ** 2) / 8.0))
        return torch.cat([px, torch.stack(out, 1).to(px.dtype)], 1)

    class SkillKP(nn.Module):
        """--arch keypoints: two conv blocks to 16 x 16, FiLM, two more conv layers, then a spatial softmax of 32 keypoint
        maps (expected image coordinates, the visuomotor-policy layer for precise positions) + pooled features + condition."""

        def __init__(self, nkp=32):
            super().__init__()
            self.chunk, self.act_dim = chunk, act_dim
            self.enc = nn.Sequential(block(8 if cond == "spatial" else 6, width), block(width, 2 * width))
            self.mid = nn.Sequential(nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.GroupNorm(8, 2 * width), nn.GELU(),
                                     nn.Conv2d(2 * width, 2 * width, 3, padding=1), nn.GroupNorm(8, 2 * width), nn.GELU())
            self.kp = nn.Conv2d(2 * width, nkp, 1)
            self.emb = nn.Embedding(K, 64)
            self.cond = nn.Sequential(nn.Linear(4 + 2 * A if cond in ("full", "spatial") else 2 + A, 64), nn.GELU(), nn.Linear(64, 64))
            self.film = nn.Linear(128, 2 * width * 2)
            self.head = nn.Sequential(nn.Linear(2 * nkp + 2 * width + 128, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                      nn.Linear(hidden, chunk * act_dim))

        def forward(self, px, e, cur, tgt, shift=None):
            if cond == "spatial":
                px = maps(px, cur, tgt, shift)
            if cond in ("full", "spatial"):
                z = torch.cat([cur[:, :2] / 32 - 1, tgt[:, :2] / 32 - 1, cur[:, 2:2 + A] * 2 - 1, tgt[:, 2:2 + A] * 2 - 1], -1)
            else:
                z = torch.cat([tgt[:, :2] / 32 - 1, tgt[:, 2:2 + A] * 2 - 1], -1)
            c = torch.cat([self.emb(e.long()), self.cond(z)], -1)
            f = self.enc(px * 2 - 1)
            g, b = self.film(c).chunk(2, -1)
            f = self.mid(f * (1 + g[:, :, None, None]) + b[:, :, None, None])
            k = self.kp(f).float()
            B, N, H, W = k.shape
            att = torch.softmax(k.flatten(2), -1).view(B, N, H, W)
            lin = torch.linspace(-1, 1, H, device=k.device)
            xs = (att * lin[None, None, None, :]).sum((-1, -2)); ys = (att * lin[None, None, :, None]).sum((-1, -2))
            out = self.head(torch.cat([xs, ys, f.float().mean((-1, -2)), c.float()], -1).to(f.dtype))
            return out.view(len(px), self.chunk, self.act_dim)

    if arch == "keypoints":
        return SkillKP()

    class Skill(nn.Module):
        def __init__(self):
            super().__init__()
            self.chunk, self.act_dim = chunk, act_dim
            self.cnn = nn.Sequential(block(8 if cond == "spatial" else 6, width), block(width, 2 * width), block(2 * width, 2 * width), block(2 * width, 4 * width))
            self.emb = nn.Embedding(K, 64)
            self.cond = nn.Sequential(nn.Linear(4 + 2 * A if cond in ("full", "spatial") else 2 + A, 64), nn.GELU(), nn.Linear(64, 64))
            self.film = nn.Linear(128, 4 * width * 2)
            self.head = nn.Sequential(nn.Linear(4 * width * 16 + 128, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                      nn.Linear(hidden, chunk * act_dim))

        def forward(self, px, e, cur, tgt, shift=None):
            """px (B, 6, 64, 64) [0, 1]; e (B,); cur, tgt (B, D) raw states (px, app in [0, 1], covered) -> (B, chunk, act_dim).
            cond "spatial": two more input channels, Gaussian maps (2 px) at the entity's current and target position (moved
            by `shift` (B, 2) px when the frames were shifted for augmentation)."""
            if cond == "spatial":
                g = torch.arange(64, device=px.device, dtype=torch.float32)
                maps = []
                for st in (cur, tgt):
                    uv = st[:, :2].float() + (shift.float() if shift is not None else 0.0)
                    maps.append(torch.exp(-((g[None, None, :] - uv[:, 0, None, None]) ** 2 + (g[None, :, None] - uv[:, 1, None, None]) ** 2) / 8.0))
                px = torch.cat([px, torch.stack(maps, 1).to(px.dtype)], 1)
            if cond in ("full", "spatial"):
                z = torch.cat([cur[:, :2] / 32 - 1, tgt[:, :2] / 32 - 1, cur[:, 2:2 + A] * 2 - 1, tgt[:, 2:2 + A] * 2 - 1], -1)
            else:
                z = torch.cat([tgt[:, :2] / 32 - 1, tgt[:, 2:2 + A] * 2 - 1], -1)
            c = torch.cat([self.emb(e.long()), self.cond(z)], -1)
            f = self.cnn(px * 2 - 1)
            g, b = self.film(c).chunk(2, -1)
            f = f * (1 + g[:, :, None, None]) + b[:, :, None, None]
            return self.head(torch.cat([f.flatten(1), c], -1)).view(len(px), self.chunk, self.act_dim)

    return Skill()


class MSplit:
    """frames memory-mapped (val in RAM), actions / terminals in RAM; the first `episodes` episodes."""

    def __init__(self, cache: Path, split: str, episodes: int):
        term = np.load(cache / f"{split}_terminals.npy")
        ends = np.flatnonzero(term)[:episodes]
        self.n = int(ends[-1] + 1)
        self.obs = Frames(cache / f"{split}_observations.npy", self.n, ram=(split == "val"))
        self.actions = np.asarray(np.load(cache / f"{split}_actions.npy", mmap_mode="r")[:self.n], np.float32)
        self.terminals = np.asarray(term[:self.n])
        self.ep = np.concatenate([[0], np.cumsum(self.terminals[:-1])]).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--hist-gap", type=int, default=2,
                    help="frame gap of the second input frame (0 = the current frame twice). Two frames alone let behaviour "
                         "cloning copy the arm's ongoing motion and ignore the event condition (puzzle VAL: swapping the "
                         "condition changed predicted actions by .095 vs .312 with one frame); one frame alone loses the "
                         "grasp phase (cube-triple task 1 0/6). Default: two frames with history dropout .5, and the closed "
                         "loop feeds the current frame twice at an event's first query (seed 0, 30 episodes: puzzle 3/30, "
                         "cube 4/30; one frame: puzzle 7/30, cube ~0/30)")
    ap.add_argument("--hist-dropout", type=float, default=0.5,
                    help="with --hist-gap > 0: probability that a training sample sees the current frame twice (keeps the "
                         "phase cue of two frames while the motion alone no longer predicts the actions)")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--release", type=int, default=10)
    ap.add_argument("--include-side-effects", action="store_true")
    ap.add_argument("--init-model", type=Path, default=None, help="continue compatible skill weights; fresh optimizer")
    ap.add_argument("--arch", choices=("cnn", "keypoints"), default="cnn")
    ap.add_argument("--cond", choices=("full", "target", "spatial"), default="spatial",
                    help="spatial (default): + Gaussian maps at the entity's current and target position (puzzle-4x5 dev loop, "
                         "PRIVILEGED scoring: exact presses 29% vs 15% with full)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = a.device
    t0 = time.time()
    train, val = MSplit(a.cache, "train", a.episodes), MSplit(a.cache, "val", a.val_episodes)
    tr, va = dict(np.load(a.events / "events_train.npz")), dict(np.load(a.events / "events_val.npz"))
    K, D = tr["before"].shape[1], tr["before"].shape[2]; A = D - 3

    def samples(ev, split):
        n = split.n
        last_frames = np.r_[np.flatnonzero(split.terminals), n - 1]
        idx, ee, cc, tt = [], [], [], []
        tk = ev["target_known"] if "target_known" in ev else np.ones(len(ev["e"]), bool)   # object events: known targets only
        for s0, te, e, b, t_, kn, ok in zip(ev["seg_start"], ev["t"], ev["e"], ev["before"], ev["target"], ev["knock"], tk):
            if not ok or not eligible_segment(kn, a.include_side_effects, te, 0, n):
                continue
            episode_last = last_frames[np.searchsorted(last_frames, te)]
            f = np.fromiter(segment_frames(s0, te, a.release, episode_last), dtype=np.int64)
            idx.append(f); ee.append(np.full(len(f), e)); cc.append(np.repeat(b[e][None], len(f), 0)); tt.append(np.repeat(t_[None], len(f), 0))
        return np.concatenate(idx), np.concatenate(ee), np.concatenate(cc).astype(np.float32), np.concatenate(tt).astype(np.float32)

    xi, xe, xc, xt = samples(tr, train)
    vi, ve, vc, vt = samples(va, val)
    mu, sd = train.actions.mean(0), train.actions.std(0) + 1e-6
    act = torch.as_tensor((train.actions - mu) / sd, device=dev)
    vact = torch.as_tensor((val.actions - mu) / sd, device=dev)
    H = a.chunk
    bounds = {}
    for name, sp in (("train", train), ("val", val)):
        bounds[name] = (np.r_[0, np.nonzero(sp.terminals)[0] + 1][sp.ep], np.r_[np.nonzero(sp.terminals)[0], sp.n - 1][sp.ep])

    def batch(split, idx, actions, augment=False):
        first_s, last_s = bounds["train" if split is train else "val"]
        prev = np.maximum(idx - a.hist_gap, first_s[idx])
        if augment and a.hist_dropout > 0:                                       # history dropout: the current frame twice
            prev = np.where(rng.random(len(idx)) < a.hist_dropout, idx, prev)
        px = torch.as_tensor(np.concatenate([split.obs[prev], split.obs[idx]], -1), device=dev).permute(0, 3, 1, 2).float().div_(255.0)
        shift = torch.zeros(len(idx), 2, device=dev)
        if augment:
            pad = torch.nn.functional.pad(px, (2, 2, 2, 2), mode="replicate")
            dx, dy = rng.integers(0, 5, 2)
            px = pad[:, :, dy:dy + 64, dx:dx + 64]
            shift += torch.tensor([2.0 - dx, 2.0 - dy], device=dev)                 # where a pixel (u, v) moved to
        steps = idx[:, None] + np.arange(H)[None]
        valid = steps <= last_s[idx][:, None]
        steps = np.minimum(steps, last_s[idx][:, None])
        return px, actions[torch.as_tensor(steps, device=dev)], torch.as_tensor(valid, device=dev).float(), shift

    pi = make_skill(K, A, chunk=H, cond=a.cond, arch=a.arch).to(dev)
    extra = {}
    if a.init_model is not None:
        parent = torch.load(a.init_model, map_location="cpu", weights_only=False)
        assert parent["K"] == K and parent["A"] == A and parent["chunk"] == H and parent.get("cond", "full") == a.cond and parent.get("arch", "cnn") == a.arch
        np.testing.assert_allclose(parent["action_mean"], mu)
        np.testing.assert_allclose(parent["action_std"], sd)
        pi.load_state_dict(parent["skill"])
        import hashlib
        extra.update(parent_model=str(a.init_model), parent_sha256=hashlib.sha256(a.init_model.read_bytes()).hexdigest())
    extra.update(include_side_effects=a.include_side_effects, training_steps=a.steps, learning_rate=a.lr, release_clipped_at_episode_boundary=True)
    save_json(a.out / "skill_data_report.json", {
        "train_events": int(len(tr["e"])), "train_multi_object_events": int(np.count_nonzero(tr["knock"])),
        "val_events": int(len(va["e"])), "train_frame_event_pairs": int(len(xi)), "val_frame_event_pairs": int(len(vi)),
        "include_side_effects": a.include_side_effects, "parent_model": str(a.init_model), "steps": a.steps, "lr": a.lr})
    opt = torch.optim.AdamW(pi.parameters(), lr=a.lr, weight_decay=1e-4)
    warm = 1000
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
    print({"train_samples": len(xi), "val_samples": len(vi), "entities": K}, flush=True)
    log, best = [], float("inf")
    meta = {"K": int(K), "A": int(A), "chunk": H, "hist_gap": a.hist_gap, "release": a.release, "action_mean": mu, "action_std": sd, "cond": a.cond, "arch": a.arch, "hist_dropout": a.hist_dropout}
    for step in range(a.steps):
        s = rng.integers(0, len(xi), a.batch)
        px, tgt, m, sh = batch(train, xi[s], act, augment=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            pred = pi(px, torch.as_tensor(xe[s], device=dev), torch.as_tensor(xc[s], device=dev), torch.as_tensor(xt[s], device=dev), sh).float()
        loss = (((pred - tgt) ** 2).mean(-1) * m).sum() / m.sum()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(pi.parameters(), 1.0); opt.step(); sched.step()
        if step % 2000 == 0 or step == a.steps - 1:
            pi.eval()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                sv = np.random.default_rng(1).integers(0, len(vi), 2048)
                vpx, vtg, vm, _ = batch(val, vi[sv], vact)
                vp = pi(vpx, torch.as_tensor(ve[sv], device=dev), torch.as_tensor(vc[sv], device=dev), torch.as_tensor(vt[sv], device=dev)).float()
                vl = ((((vp - vtg) ** 2).mean(-1) * vm).sum() / vm.sum()).item()
            pi.train()
            log.append({"step": step, "loss": loss.item(), "val_chunk": vl, "min": round((time.time() - t0) / 60, 1)}); print(log[-1], flush=True)
            if vl < best:
                best = vl
                a.out.mkdir(parents=True, exist_ok=True)
                torch.save({"skill": pi.state_dict(), "step": step, **meta, **extra}, a.out / "skill_best.pt")
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"skill": pi.state_dict(), **meta, **extra}, a.out / "skill.pt")
    save_json(a.out / "skill_log.json", log)


if __name__ == "__main__":
    main()
