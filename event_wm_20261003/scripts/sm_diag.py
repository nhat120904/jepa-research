#!/usr/bin/env python3
"""Export the scene memory (codes, gate openings, fast-layer coverage, visibility) for train/val episodes and
score it on val with PRIVILEGED simulator state (scoring only; nothing here feeds back into training).

Export (per split, memory-mapped .npy under OUT/export/<split>/): codes uint16 (N, 256), opened bool, gate /
alpha_tok / vis uint8 (x255), plus episode starts. Episodes run causally from their first frame.

Diagnostics on val (diag.json):
  probes      memory values -> buttons (linear, per button), cube xy (keypoint probe as in slowmap_probe.py,
              compared with the same probe on pixels), arm joints qpos[:6] (MLP R^2, agent-freeness: low is
              good), scene drawer / window joints (MLP R^2). First half of the val frames fits, second half scores.
  stability   at PRIVILEGED rest frames (object state unchanged over +-5 frames): token code changes per frame
              and gate-open rate; per-token open-rate map at rest (where false openings happen).
  distinct    rest periods (>= 10 rest frames): consecutive periods whose true state differs must have different
              code grids; consecutive periods with the same true state should keep identical grids (and
              same-configuration periods across the split, puzzle: token agreement).
  occlusion   synthetic box occluder (12 x 12 px, 10 frames) inside long rest periods: are codes unchanged after
              the box leaves, and does the visibility head mark the hidden tokens.
  timing      export frames / s.
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

from sm_model import SceneMemory, run_batch

CUBE_SLICES = (14, 21, 28)


def episode_bounds(cache: Path, split: str, n_ep: int):
    term = np.load(cache / f"{split}_terminals.npy")
    ends = np.flatnonzero(term)[:n_ep]
    starts = np.r_[0, ends[:-1] + 1]
    return starts, ends


def export(model, cache, split, n_ep, out: Path, dev, eb=50):
    starts, ends = episode_bounds(cache, split, n_ep)
    T = int(ends[0] - starts[0] + 1)
    assert np.all(ends - starts + 1 == T), "episodes of equal length expected"
    N = len(starts) * T
    obs = np.load(cache / f"{split}_observations.npy", mmap_mode="r")
    act = np.load(cache / f"{split}_actions.npy", mmap_mode="r")
    out.mkdir(parents=True, exist_ok=True)
    mm = {"codes": np.lib.format.open_memmap(out / "codes.npy", "w+", np.uint16, (N, 256)),
          "opened": np.lib.format.open_memmap(out / "opened.npy", "w+", bool, (N, 256))}
    for k in ("gate", "alpha_tok", "vis"):
        mm[k] = np.lib.format.open_memmap(out / f"{k}.npy", "w+", np.uint8, (N, 256))
    t0 = time.time()
    for b in range(0, len(starts), eb):
        e = np.arange(b, min(b + eb, len(starts)))
        o = np.asarray(obs[starts[e[0]]:starts[e[-1]] + T]).reshape(len(e), T, 64, 64, 3)
        a = np.asarray(act[starts[e[0]]:starts[e[-1]] + T], np.float32).reshape(len(e), T, -1)
        r = run_batch(model, o, a, dev)
        sl = slice(starts[e[0]], starts[e[-1]] + T)
        mm["codes"][sl] = r["codes"].reshape(-1, 256).astype(np.uint16)
        mm["opened"][sl] = r["opened"].reshape(-1, 256)
        for k in ("gate", "alpha_tok", "vis"):
            mm[k][sl] = np.round(np.clip(r[k], 0, 1) * 255).reshape(-1, 256).astype(np.uint8)
    for v in mm.values():
        v.flush()
    np.save(out / "starts.npy", starts)
    return {"frames": N, "episodes": len(starts), "frames_per_s": N / (time.time() - t0)}


def privileged_state(cache: Path, family: str, n: int):
    """Object state per val frame for scoring: dict with 'disc' (n, nb) buttons, 'pos' (n, k, 3) positions,
    'joint' (n, j) slide joints, 'arm' (n, 6) or None."""
    s = {"disc": None, "pos": None, "joint": None, "arm": None}
    if family in ("puzzle", "scene"):
        s["disc"] = np.asarray(np.load(cache / "val_button_states.npy", mmap_mode="r")[:n]).astype(np.int64)
    if family in ("cube", "scene"):
        q = np.asarray(np.load(cache / "val_qpos.npy", mmap_mode="r")[:n], np.float32)
        s["arm"] = q[:, :6]
        sl = tuple(c for c in (14, 21, 28, 35) if c + 3 <= q.shape[1]) if family == "cube" else (14,)   # 1-4 cubes
        s["pos"] = np.stack([q[:, i:i + 3] for i in sl], 1)
        if family == "scene":
            s["joint"] = q[:, 23:25]
    return s


def rest_mask(st, starts, T, w=5, tol_pos=0.002, tol_joint=0.002):
    n = len(next(v for v in st.values() if v is not None))
    still = np.ones(n, bool)
    if st["pos"] is not None:
        d = np.r_[0, np.linalg.norm(np.diff(st["pos"], axis=0), axis=-1).max(-1)]
        still &= d < tol_pos
    if st["joint"] is not None:
        still &= np.r_[0, np.abs(np.diff(st["joint"], axis=0)).max(-1)] < tol_joint
    if st["disc"] is not None:
        still &= np.r_[True, (np.diff(st["disc"], axis=0) == 0).all(-1)]
    still[starts] = True                                                        # episode boundary is not motion
    rest = still.copy()
    for k in range(1, w + 1):
        rest[k:] &= still[:-k]; rest[:-k] &= still[k:]
    ep = np.repeat(np.arange(len(starts)), T)[:n]
    loc = np.arange(n) - starts[ep]
    rest &= (loc >= w) & (loc < T - w)
    return rest


def periods(mask, starts, T, min_len=10):
    out = []
    for e, s0 in enumerate(starts):
        m = mask[s0:s0 + T]
        d = np.diff(np.r_[0, m.astype(int), 0])
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            if b - a >= min_len:
                out.append((e, s0 + a, s0 + b))
    return out


def state_differs(st, i, j, pos_tol=0.02, joint_tol=0.01):
    if st["disc"] is not None and (st["disc"][i] != st["disc"][j]).any():
        return True
    if st["pos"] is not None and np.linalg.norm(st["pos"][i] - st["pos"][j], axis=-1).max() > pos_tol:
        return True
    if st["joint"] is not None and np.abs(st["joint"][i] - st["joint"][j]).max() > joint_tol:
        return True
    return False


def probes(model, codes, st, obs, dev, steps=3000):
    torch.manual_seed(0)
    n = len(codes); half = n // 2
    V = model.fsq.values(torch.as_tensor(codes.astype(np.int64), device=dev)).view(n, 16, 16, -1).permute(0, 3, 1, 2).contiguous()
    res = {}

    def fit(Fx, Y, kind):
        mu, sd = Fx[:half].mean(0), Fx[:half].std(0) + 1e-3
        if kind == "logit":
            net = nn.Linear(Fx.shape[1], Y.shape[1]).to(dev)
        else:
            net = nn.Sequential(nn.Linear(Fx.shape[1], 256), nn.GELU(), nn.Linear(256, Y.shape[1])).to(dev)
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
        acc = ((fit(flat, Y, "logit") > 0).float() == Y[half:]).float().mean(0).cpu().numpy()
        res["buttons_acc"] = acc.round(4).tolist(); res["buttons_min"] = float(acc.min())
    if st["pos"] is not None:
        K = st["pos"].shape[1]
        tgt = torch.as_tensor(st["pos"][..., :2].reshape(n, -1), device=dev)

        def keypoint(Fm):
            C, H, W = Fm.shape[1:]
            conv, aff = nn.Conv2d(C, K, 1).to(dev), nn.Linear(2 * K, 2 * K).to(dev)
            gy, gx = torch.meshgrid(torch.linspace(-1, 1, H, device=dev), torch.linspace(-1, 1, W, device=dev), indexing="ij")
            opt = torch.optim.Adam(list(conv.parameters()) + list(aff.parameters()), lr=3e-3)
            mu, sd = tgt[:half].mean(0), tgt[:half].std(0)

            def fwd(f):
                att = torch.softmax(conv(f).flatten(2), -1).view(len(f), K, H, W)
                return aff(torch.stack([(att * gx).sum((2, 3)), (att * gy).sum((2, 3))], -1).flatten(1)) * sd + mu

            for _ in range(steps):
                i = torch.randint(0, half, (512,), device=dev)
                loss = ((fwd(Fm[i]) - tgt[i]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                p = torch.cat([fwd(Fm[s:s + 4096]) for s in range(half, n, 4096)])
            err = (p - tgt[half:]).view(-1, K, 2).norm(dim=-1)
            return {"median_err_cm": (err.median(0).values * 100).cpu().numpy().round(2).tolist(),
                    "within_2cm": (err < 0.02).float().mean(0).cpu().numpy().round(3).tolist()}

        res["memory_keypoint"] = keypoint(V)
        X = torch.as_tensor(np.asarray(obs[:n]), device=dev).permute(0, 3, 1, 2).float().div(127.5).sub(1)
        res["pixels_keypoint"] = keypoint(X)
        res["keypoint_ratio_memory_over_pixels"] = (np.array(res["memory_keypoint"]["median_err_cm"]) /
                                                    np.maximum(res["pixels_keypoint"]["median_err_cm"], 1e-3)).round(2).tolist()
    pooled = nn.functional.adaptive_avg_pool2d(V, 8).flatten(1)
    if st["arm"] is not None:
        res["arm_mlp_r2"] = r2(fit(pooled, torch.as_tensor(st["arm"], device=dev), "mlp"), torch.as_tensor(st["arm"][half:], device=dev))
    if st["joint"] is not None:
        res["joint_mlp_r2"] = r2(fit(flat, torch.as_tensor(st["joint"], device=dev), "mlp"), torch.as_tensor(st["joint"][half:], device=dev))
    return res


@torch.no_grad()
def occlusion_test(model, obs, act, per, dev, n_tests=200, L=40, seed=0):
    rng = np.random.default_rng(seed)
    long = [p for p in per if p[2] - p[1] >= L]
    if not long:
        return {"tests": 0}
    pick = [long[i] for i in rng.choice(len(long), min(n_tests, len(long)), replace=False)]
    o = np.stack([np.asarray(obs[s:s + L]) for _, s, _ in pick]); a = np.stack([np.asarray(act[s:s + L], np.float32) for _, s, _ in pick])
    oc = o.copy(); masks = np.zeros((len(pick), 16, 16), bool)
    for i in range(len(pick)):
        r, c = rng.integers(0, 53, size=2)
        oc[i, 10:20, r:r + 12, c:c + 12] = rng.integers(0, 256, size=3)
        m = np.zeros((64, 64)); m[r:r + 12, c:c + 12] = 1
        masks[i] = m.reshape(16, 4, 16, 4).mean((1, 3)) >= 0.5
    def chunked(o_, a_, eb=50):   # same episodes-per-pass as export(); 200 x 64 frames at once needs > 12 GB
        parts = [run_batch(model, o_[i:i + eb], a_[i:i + eb], dev) for i in range(0, len(o_), eb)]
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    clean = chunked(o, a); occ = chunked(oc, a)
    same_after = (clean["codes"][:, -1] == occ["codes"][:, -1]).mean()
    hidden_after = np.mean([(clean["codes"][i, -1][masks[i].ravel()] == occ["codes"][i, -1][masks[i].ravel()]).mean() for i in range(len(pick))])
    vis_in = np.mean([occ["vis"][i, 12:18][:, masks[i].ravel()].mean() for i in range(len(pick))])
    vis_out = np.mean([occ["vis"][i, 12:18][:, ~masks[i].ravel()].mean() for i in range(len(pick))])
    return {"tests": len(pick), "codes_equal_after_reveal_all_tokens": float(same_after),
            "codes_equal_after_reveal_hidden_tokens": float(hidden_after),
            "vis_mean_hidden_tokens": float(vis_in), "vis_mean_other_tokens": float(vis_out)}


def panels(model, obs, act, starts, T, path: Path, dev, episodes=3, frames=(0, 100, 300, 600, 900)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    with torch.no_grad():
        for e in range(min(episodes, len(starts))):
            x = torch.as_tensor(np.asarray(obs[starts[e]:starts[e] + T]), device=dev).permute(0, 3, 1, 2).float()[None] / 255
            ap = np.zeros((1, T, act.shape[1]), np.float32); ap[0, 1:] = act[starts[e]:starts[e] + T - 1]
            o = model(x[:, :max(frames) + 1], torch.as_tensor(ap[:, :max(frames) + 1], device=dev))
            for t in frames:
                rows.append([x[0, t], o["scene"][0, t], o["alpha"][0, t].expand(3, -1, -1), o["x_hat"][0, t]])
    fig, ax = plt.subplots(len(rows), 4, figsize=(8, 2 * len(rows)))
    for i, r in enumerate(rows):
        for j, (im, name) in enumerate(zip(r, ("frame", "memory scene", "fast alpha", "composite"))):
            ax[i, j].imshow(im.permute(1, 2, 0).clamp(0, 1).cpu().numpy()); ax[i, j].axis("off")
            if i == 0:
                ax[i, j].set_title(name, fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--family", choices=("cube", "puzzle", "scene"), required=True)
    ap.add_argument("--episodes", type=int, default=1000, help="train episodes to export (0: none)")
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    dev = a.device
    ck = torch.load(a.model, map_location=dev, weights_only=False)
    model = SceneMemory(ck["cfg"]["width"]).to(dev).eval(); model.load_state_dict(ck["model"])
    a.out.mkdir(parents=True, exist_ok=True)
    res = {"model": str(a.model), "family": a.family, "train_cfg": ck["cfg"], "train_steps": ck["step"]}
    res["export_val"] = export(model, a.cache, "val", a.val_episodes, a.out / "export" / "val", dev)
    print("EXPORT val", res["export_val"], flush=True)

    starts, ends = episode_bounds(a.cache, "val", a.val_episodes)
    T = int(ends[0] - starts[0] + 1); n = len(starts) * T
    E = a.out / "export" / "val"
    codes = np.load(E / "codes.npy"); opened = np.load(E / "opened.npy")
    obs = np.load(a.cache / "val_observations.npy", mmap_mode="r"); act = np.load(a.cache / "val_actions.npy", mmap_mode="r")
    st = privileged_state(a.cache, a.family, n)
    rest = rest_mask(st, starts, T)
    nxt = np.r_[rest[1:] & rest[:-1], False]
    nxt[ends] = False
    ch = codes[1:] != codes[:-1]
    res["stability"] = {"rest_frame_fraction": float(rest.mean()),
                        "token_code_change_per_rest_frame": float(ch[nxt[:-1]].mean()),
                        "rest_frames_with_any_code_change": float(ch[nxt[:-1]].any(-1).mean()),
                        "open_rate_rest": float(opened[rest].mean()), "open_rate_nonrest": float(opened[~rest].mean())}
    gate_map = opened[rest].mean(0).reshape(16, 16)
    np.save(a.out / "gate_map_rest.npy", gate_map)
    per = periods(rest, starts, T)
    diff_ok = diff_n = same_ok = same_n = 0; ham = []
    for (e1, a1, b1), (e2, a2, b2) in zip(per[:-1], per[1:]):
        if e1 != e2:
            continue
        i, j = (a1 + b1) // 2, (a2 + b2) // 2
        dif = state_differs(st, i, j); neq = (codes[i] != codes[j])
        if dif:
            diff_n += 1; diff_ok += bool(neq.any())
        else:
            same_n += 1; same_ok += bool(~neq.any()); ham.append(float(neq.mean()))
    res["distinct"] = {"rest_periods": len(per), "changed_pairs": diff_n, "changed_pairs_codes_differ": diff_ok / max(diff_n, 1),
                       "unchanged_pairs": same_n, "unchanged_pairs_codes_identical": same_ok / max(same_n, 1),
                       "unchanged_pairs_token_hamming_mean": float(np.mean(ham)) if ham else None}
    if a.family == "puzzle":
        mids = [((a1 + b1) // 2) for _, a1, b1 in per]
        keys = [st["disc"][m].tobytes() for m in mids]
        agree = []
        by = {}
        for m, k in zip(mids, keys):
            by.setdefault(k, []).append(m)
        for ms in by.values():
            for u in range(len(ms)):
                for v in range(u + 1, min(len(ms), u + 5)):
                    agree.append(float((codes[ms[u]] == codes[ms[v]]).mean()))
        res["distinct"]["same_config_pairs"] = len(agree)
        res["distinct"]["same_config_token_agreement"] = float(np.mean(agree)) if agree else None
    res["probes"] = probes(model, codes[:n], st, obs, dev)
    print("PROBES", json.dumps(res["probes"]), flush=True)
    res["occlusion"] = occlusion_test(model, obs, act, per, dev)
    try:
        panels(model, obs, act, starts, T, a.out / "panels.png", dev)
    except Exception as ex:                                       # figure only; never blocks the numbers
        res["panels_error"] = repr(ex)
    if a.episodes > 0:
        res["export_train"] = export(model, a.cache, "train", a.episodes, a.out / "export" / "train", dev)
    (a.out / "diag.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "train_cfg"}), flush=True)


if __name__ == "__main__":
    main()
