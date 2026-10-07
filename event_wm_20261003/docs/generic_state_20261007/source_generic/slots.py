#!/usr/bin/env python3
"""Unified front end, candidate 3: temporally consistent object slots on frozen DINOv2 features.

Compact SlotContrast (Manasyan et al., CVPR 2025): frozen DINOv2-with-registers ViT-S/14 patch features
(64x64 frames upsampled to 224 -> 16x16 tokens), recurrent Slot Attention (learned fixed initial slots;
slots of frame t+1 initialised from frame t through a one-layer transformer predictor), an MLP broadcast
decoder that reconstructs the patch features with per-slot alpha masks, and a slot-slot contrastive loss
across consecutive frames. One configuration (K slots, all hyperparameters) for every task family.
Evaluation (PRIVILEGED probes only, the pre-registered criteria of the unified state component):
  agent slots = slots whose mask centroid moves in most frames (2-means on log moving fraction);
  (1) lights / buttons from the non-agent slot vectors (ridge, held-out half);
  (2) every cube tracked by some non-agent slot centroid (best slot per cube, affine fit, median error);
  (3) arm joints from non-agent slots (MLP probe R^2), vs from all slots;
  (4) events = jumps of non-agent slot states (vector + centroid), recall / precision vs privileged events.
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

DINO = "facebook/dinov2-with-registers-small"


def make_model(K=8, D=128, F=384, N=256):
    import torch
    import torch.nn as nn

    class SlotAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln_in, self.ln_s, self.ln_m = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
            self.q, self.k, self.v = nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False), nn.Linear(D, D, bias=False)
            self.gru = nn.GRUCell(D, D)
            self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

        def forward(self, x, s, iters):
            x = self.ln_in(x)
            k, v = self.k(x), self.v(x)
            for _ in range(iters):
                q = self.q(self.ln_s(s))
                att = torch.softmax(torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(D), dim=1) + 1e-8
                attn = att / att.sum(-1, keepdim=True)
                upd = torch.einsum("bkn,bnd->bkd", attn, v)
                s = self.gru(upd.reshape(-1, D), s.reshape(-1, D)).view_as(s)
                s = s + self.mlp(self.ln_m(s))
            return s, att

    class Slots(nn.Module):
        def __init__(self):
            super().__init__()
            self.K = K
            self.inp = nn.Sequential(nn.LayerNorm(F), nn.Linear(F, D), nn.GELU(), nn.Linear(D, D))
            self.pos_in = nn.Parameter(torch.randn(1, N, D) * 0.02)
            self.init = nn.Parameter(torch.randn(1, K, D) * 0.1)
            self.sa = SlotAttention()
            self.pred = nn.TransformerEncoderLayer(D, 4, 4 * D, batch_first=True, norm_first=True)
            self.pos_dec = nn.Parameter(torch.randn(1, 1, N, D) * 0.02)
            self.dec = nn.Sequential(nn.Linear(D, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, F + 1))

        def encode(self, feats, prev=None):
            """feats (B, N, F) -> slots (B, K, D), attention (B, K, N)."""
            x = self.inp(feats) + self.pos_in
            if prev is None:
                return self.sa(x, self.init.expand(len(x), -1, -1), 3)
            return self.sa(x, self.pred(prev), 2)

        def decode(self, s):
            out = self.dec(s[:, :, None] + self.pos_dec)                         # (B, K, N, F + 1)
            alpha = torch.softmax(out[..., -1], dim=1)
            return (alpha[..., None] * out[..., :-1]).sum(1), alpha

    return Slots()


def dino_features(dino, x):
    """x uint8 (B, 64, 64, 3) tensor -> (B, 256, 384) patch tokens (registers and CLS dropped)."""
    import torch

    x = x.permute(0, 3, 1, 2).float().div(255.0)
    x = torch.nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device)[None, :, None, None]
    h = dino(pixel_values=(x - mean) / std).last_hidden_state
    return h[:, -256:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--train-frames", type=int, default=1_000_000)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=32, help="clips per step")
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--w-con", type=float, default=0.5)
    ap.add_argument("--eval-frames", type=int, default=30_000)
    ap.add_argument("--ref-events", type=Path, default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch
    from transformers import AutoModel

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda"
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    dino = AutoModel.from_pretrained(DINO, torch_dtype=torch.bfloat16).to(dev).eval()
    for p_ in dino.parameters():
        p_.requires_grad_(False)
    model = make_model(a.K).to(dev)
    log = []
    if a.eval_only:
        model.load_state_dict(torch.load(a.out / "slots.pt", map_location=dev)["model"])
    else:
        obs = np.load(a.cache / "train_observations.npy", mmap_mode="r")
        term = np.load(a.cache / "train_terminals.npy")
        n = min(a.train_frames, len(obs))
        ep = np.concatenate([[0], np.cumsum(term[:-1])]).astype(np.int64)[:n]
        last = np.r_[np.nonzero(term)[0], len(term) - 1][ep]
        span = (a.T - 1) * a.stride
        opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.0)
        warm = 1000
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, a.steps - warm))))
        for step in range(a.steps):
            st = rng.integers(0, n - span - 1, a.batch * 3)
            st = st[st + span <= last[st]][: a.batch]
            idx = (st[:, None] + a.stride * np.arange(a.T)[None]).ravel()
            order = np.argsort(idx)
            x = torch.as_tensor(np.asarray(obs[idx[order]]), device=dev)[torch.as_tensor(np.argsort(order), device=dev)]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                f = dino_features(dino, x).float().view(len(st), a.T, 256, -1)
            f = torch.nn.functional.layer_norm(f, f.shape[-1:])
            rec, con, s_prev, S = 0.0, 0.0, None, []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for t in range(a.T):
                    s, _ = model.encode(f[:, t], s_prev)
                    xh, _ = model.decode(s)
                    rec = rec + ((xh.float() - f[:, t]) ** 2).mean()
                    S.append(s.float()); s_prev = s
            for t in range(a.T - 1):                                               # slot-slot contrast across time
                z1 = torch.nn.functional.normalize(S[t].reshape(-1, S[t].shape[-1]), dim=-1)
                z2 = torch.nn.functional.normalize(S[t + 1].reshape(-1, S[t].shape[-1]), dim=-1)
                lab = torch.arange(len(z1), device=dev)
                con = con + torch.nn.functional.cross_entropy(z1 @ z2.T / a.tau, lab)
            loss = rec / a.T + a.w_con * con / (a.T - 1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.05 * 20)
            opt.step(); sched.step()
            if step % 1000 == 0 or step == a.steps - 1:
                log.append({"step": step, "rec": float(rec) / a.T, "con": float(con) / (a.T - 1), "min": round((time.time() - t0) / 60, 1)})
                print(log[-1], flush=True)
        torch.save({"model": model.state_dict(), "K": a.K}, a.out / "slots.pt")
    model.eval()

    # ---------------- evaluation: run recurrently over whole val episodes ----------------
    vo = np.load(a.cache / "val_observations.npy", mmap_mode="r")
    vt = np.load(a.cache / "val_terminals.npy")
    vep = np.concatenate([[0], np.cumsum(vt[:-1])]).astype(np.int64)
    m = min(a.eval_frames, len(vo))
    m = int(np.nonzero(vt[:m])[0][-1] + 1) if vt[:m].any() else m
    vep = vep[:m]
    starts = np.r_[0, np.nonzero(vt[:m - 1])[0] + 1]
    K = a.K
    SV = np.zeros((m, K, 128), np.float32); CEN = np.zeros((m, K, 2), np.float32); AREA = np.zeros((m, K), np.float32)
    gy, gx = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    gx_t = torch.as_tensor(gx.ravel() * 4 + 2.0, device=dev).float(); gy_t = torch.as_tensor(gy.ravel() * 4 + 2.0, device=dev).float()
    with torch.no_grad():
        # process all episodes in parallel, frame index by frame index
        lens = np.diff(np.r_[starts, m])
        L = lens.max()
        s_prev = None
        for t in range(L):
            alive = np.nonzero(lens > t)[0]
            fr = starts[alive] + t
            x = torch.as_tensor(np.asarray(vo[fr]), device=dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = dino_features(dino, x).float()
                f = torch.nn.functional.layer_norm(f, f.shape[-1:])
                s, att = model.encode(f, None if t == 0 else s_prev[alive])
                _, alpha = model.decode(s)
            alpha = alpha.float()                                                 # (B, K, 256)
            area = alpha.sum(-1)
            cen = torch.stack([(alpha * gx_t).sum(-1), (alpha * gy_t).sum(-1)], -1) / area[..., None].clamp(min=1e-6)
            SV[fr] = s.float().cpu().numpy(); CEN[fr] = cen.cpu().numpy(); AREA[fr] = area.cpu().numpy()
            full = torch.zeros(len(starts), K, s.shape[-1], device=dev) if s_prev is None else s_prev.clone()
            full[torch.as_tensor(alive, device=dev)] = s.float()
            s_prev = full
    same = vep[1:] == vep[:-1]
    step_c = np.linalg.norm(CEN[1:] - CEN[:-1], axis=-1)                          # (m-1, K)
    present = (AREA[1:] > 2) & (AREA[:-1] > 2) & same[:, None]
    movf = np.array([(step_c[present[:, k], k] > 0.5).mean() if present[:, k].any() else 0.0 for k in range(K)])
    thr, sep, _ = two_means_threshold(np.log(movf + 1e-3))
    agent = np.log(movf + 1e-3) > thr
    res = {"env": a.cache.name, "frames": m, "K": K, "slot_moving_frac": np.round(movf, 3).tolist(), "agent_slots": np.nonzero(agent)[0].tolist(),
           "slot_mean_area": np.round(AREA.mean(0), 1).tolist(), "log": log}
    obj = ~agent
    half = m // 2
    Xo = np.concatenate([SV[:, obj].reshape(m, -1), CEN[:, obj].reshape(m, -1) / 64.0], 1)
    Xo = (Xo - Xo[:half].mean(0)) / (Xo[:half].std(0) + 1e-3)

    def ridge(X, Y, lam=10.0):
        Xb = np.c_[X, np.ones(len(X))]
        W = np.linalg.solve(Xb[:half].T @ Xb[:half] + lam * np.eye(Xb.shape[1]), Xb[:half].T @ Y[:half])
        return Xb[half:] @ W

    if (a.cache / "val_button_states.npy").exists():
        bs = np.asarray(np.load(a.cache / "val_button_states.npy", mmap_mode="r")[:m]).astype(np.float32)
        acc = ((ridge(Xo, bs) > 0.5) == (bs[half:] > 0.5)).mean(0)
        res["criterion1_lights_acc_min_mean"] = [float(acc.min()), float(acc.mean())]
    # state jumps of object slots -> events
    sv_n = SV / (np.linalg.norm(SV, axis=-1, keepdims=True) + 1e-6)
    jump = np.zeros(m)
    jump[1:] = (np.linalg.norm(sv_n[1:, obj] - sv_n[:-1, obj], axis=-1).max(1)
                + np.linalg.norm(CEN[1:, obj] - CEN[:-1, obj], axis=-1).max(1) / 64.0)
    jump[1:][~same] = 0
    jthr, jsep, _ = two_means_threshold(np.log(jump[1:][same] + 1e-6))
    on = np.log(jump + 1e-6) > jthr
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
    res["events_per_episode"] = float(len(ev) / (vep[-1] + 1)); res["jump_frame_frac"] = float(on.mean())
    refs = []
    if (a.cache / "val_qpos.npy").exists():
        q = np.asarray(np.load(a.cache / "val_qpos.npy", mmap_mode="r")[:m]).astype(np.float32)
        slices = [s_ for s_ in (14, 21, 28, 35) if s_ + 3 <= q.shape[1]]
        if "cube" in a.cache.name:
            cubes = [q[:, s_:s_ + 2] for s_ in slices]
        else:
            cubes = [q[:, 14:16]]                                                   # scene: one cube
        errs = []
        for c in cubes:                                                             # best object slot per cube
            best = None
            for k in np.nonzero(obj)[0]:
                Xc = np.c_[CEN[:, k], np.ones(m)]
                W = np.linalg.lstsq(Xc[:half], c[:half], rcond=None)[0]
                e = np.median(np.linalg.norm(Xc[half:] @ W - c[half:], axis=-1))
                best = e if best is None else min(best, e)
            errs.append(round(float(best) * 100, 2) if best is not None else None)
        res["criterion2_cube_median_err_cm"] = errs
        arm = q[:, :6]
        import torch.nn as nn

        def mlp_r2(X, Y):
            Xt = torch.as_tensor(X, device=dev).float(); Yt = torch.as_tensor(Y, device=dev).float()
            ym, ys = Yt[:half].mean(0), Yt[:half].std(0) + 1e-6
            net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.GELU(), nn.Linear(256, Y.shape[1])).to(dev)
            o = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
            for _ in range(3000):
                i = torch.randint(0, half, (512,), device=dev)
                l_ = ((net(Xt[i]) - (Yt[i] - ym) / ys) ** 2).mean()
                o.zero_grad(); l_.backward(); o.step()
            with torch.no_grad():
                p = net(Xt[half:]) * ys + ym
            return (1 - ((p - Yt[half:]) ** 2).sum(0) / ((Yt[half:] - Yt[half:].mean(0)) ** 2).sum(0)).cpu().numpy().round(3).tolist()

        Xa = np.concatenate([SV.reshape(m, -1), CEN.reshape(m, -1) / 64.0], 1)
        Xa = (Xa - Xa[:half].mean(0)) / (Xa[:half].std(0) + 1e-3)
        res["criterion3_arm_r2_object_slots"] = mlp_r2(Xo, arm)
        res["arm_r2_all_slots"] = mlp_r2(Xa, arm)
        mv = (np.abs(np.diff(q[:, 14:], axis=0)).max(1) > 2e-3) & same
        if (a.cache / "val_button_states.npy").exists():
            bb = np.asarray(np.load(a.cache / "val_button_states.npy", mmap_mode="r")[:m])
            mv |= (bb[1:] != bb[:-1]).any(1) & same
        refs.append(mv)
    elif (a.cache / "val_button_states.npy").exists():
        bb = np.asarray(np.load(a.cache / "val_button_states.npy", mmap_mode="r")[:m])
        refs.append((bb[1:] != bb[:-1]).any(1) & same)
    if refs:
        mv = refs[0]
        d_ = np.diff(np.r_[0, mv.astype(np.int8), 0]); st_, en_ = np.nonzero(d_ == 1)[0] + 1, np.nonzero(d_ == -1)[0]
        keep = [0] if len(st_) else []
        for i_ in range(1, len(st_)):
            if st_[i_] - en_[keep[-1]] > 10:
                keep.append(i_)
            else:
                en_[keep[-1]] = en_[i_]
        st_, en_ = st_[keep], en_[keep]
        cand = np.r_[st_, en_]
        res["criterion4_event_recall"] = float(np.mean([(np.abs(ev[:, 0] - x).min() <= 15) or (np.abs(ev[:, 0] - y).min() <= 15)
                                                        for x, y in zip(st_, en_)])) if len(ev) and len(st_) else 0.0
        res["criterion4_event_precision"] = float(np.mean([np.abs(cand - s_).min() <= 15 for s_ in ev[:, 0]])) if len(ev) and len(cand) else None
        res["reference_events"] = int(len(st_))
    res["minutes"] = round((time.time() - t0) / 60, 1)
    (a.out / f"slots_{a.cache.name}.json").write_text(json.dumps(res, indent=1) + "\n")
    # visualisation: frames with argmax slot map
    from PIL import Image
    pal = (np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48], [145, 30, 180],
                     [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190], [0, 128, 128], [170, 110, 40]]))
    idx = np.sort(rng.integers(0, m, 6))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        x = torch.as_tensor(np.asarray(vo[idx]), device=dev)
        f = torch.nn.functional.layer_norm(dino_features(dino, x).float(), (384,))
        s, _ = model.encode(f, torch.as_tensor(SV[np.maximum(idx - 1, 0)], device=dev))
        _, alpha = model.decode(s)
    am = alpha.float().argmax(1).view(-1, 16, 16).cpu().numpy()
    rows = [np.concatenate([np.asarray(vo[i]), np.kron(pal[am[r] % len(pal)], np.ones((4, 4, 1))).astype(np.uint8)], 1) for r, i in enumerate(idx)]
    img = np.concatenate(rows, 0)
    Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(a.out / f"slots_{a.cache.name}.png")
    print(json.dumps({k: v for k, v in res.items() if k != "log"}), flush=True)


if __name__ == "__main__":
    main()
