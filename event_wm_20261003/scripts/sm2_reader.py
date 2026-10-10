#!/usr/bin/env python3
"""State reader for scene memory v2 and events from its bits, in the build_events.py format (no state labels).

1. Pseudo-labels (--labels):
   segment  per-bit majority of the sm2_events.py state bits (fresh see-through memory on dynamic tokens) over each
            inter-interaction segment, frames more than --margin from an interaction (as refine_code.py); depends on the
            interaction segmentation (puzzle: 23 detected vs 29.6 true presses per episode, labels .91-.93 per light).
   smooth   per-bit majority of the PER-FRAME SeeThrough bits (no memory) over a +-W frame window inside the episode,
            kept where the majority is >= --agree (per-frame bits .961 per light on puzzle; the memory's p-gating and
            debounce only add lag). --label-mask frame: a frame is labelled only if every bit agrees (puzzle, 47 bits:
            15% of frames; scene, one-hot bits of 89 tokens: 4%); bit: each bit labelled where it agrees, BCE masked.
2. Reader: CNN on one raw frame (train_reader.make_reader), BCE on those labels, small shift augmentation. It reads the
   current frame and the goal image (one frame, arm anywhere) in the closed loop.
3. Events from the reader bits on every frame: per-bit debounce + changes within --window frames merged
   (common.detect_events); type = the pattern of flipped bits. As in build_events.py, a pattern type is a valid event
   identity only where effects do not depend on the state (Lights Out). Vocabulary (--vocab):
   patterns  patterns whose TRAIN count is above the 2-means split of log counts (rare -> -1), as build_events.py;
             with an imperfect reader most events carry a variant of a frequent pattern (puzzle VAL, reader_smooth:
             384 types, 38% of single-press events off their button's dominant pattern, 68% of those by one bit)
   modes     pattern_modes: every pattern climbs to its most frequent Hamming-1 neighbour until none is more frequent;
             the modes are the types, kept above the 2-means split of log mode counts over TRAIN events
   tmaps (8 x 8 per type) = tokens of the flipped bits, for skill v3.
PRIVILEGED (scoring only, reader_report.json): puzzle light accuracy through a linear map fitted on VAL halves; pressed
button (deepest button joint) per event -> type purity; reference segments.
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

from common import detect_events
from sm2_diag import reference_segments, score_spans
from sm2_events import Encoder
from sm2_model import two_means_threshold
from sm_model import FSQ
from sm2_train import Frames, to_t
from train_reader import make_reader


def pseudo_labels(ev, starts, ends, margin):
    """per-bit majority of ev['codes'] over each inter-interaction segment of every episode (frames more than `margin`
    from an interaction); -> labels (n, K) uint8 and labelled (n,) bool."""
    codes = ev["codes"]
    lab = np.zeros_like(codes)
    ok = np.zeros(len(codes), bool)
    for e, (s0, e0) in enumerate(zip(starts, ends)):
        sel = np.flatnonzero(ev["episode"] == e)
        cuts, lo = [], int(s0)
        for ts, te in zip(ev["t_start"][sel], ev["t"][sel]):
            cuts.append((lo, int(ts) - 1 - margin)); lo = int(te) + 1 + margin
        cuts.append((lo, int(e0)))
        for a_, b_ in cuts:
            if b_ < a_:
                continue
            lab[a_:b_ + 1] = (codes[a_:b_ + 1].mean(0) >= 0.5).astype(np.uint8)
            ok[a_:b_ + 1] = True
    return lab, ok


def smooth_labels(bits, starts, ends, w, agree, per_bit=False):
    """per-bit majority of bits (n, K) over +-w frames within each episode; labelled where every bit's majority share
    is >= agree (ok (n,)), or per bit where its share is >= agree (per_bit: ok (n, K))."""
    lab = np.zeros_like(bits); ok = np.zeros(bits.shape if per_bit else len(bits), bool)
    for s0, e0 in zip(starts, ends):
        x = bits[s0:e0 + 1].astype(np.float64)
        cs = np.r_[np.zeros((1, x.shape[1])), np.cumsum(x, 0)]
        T = len(x); lo = np.maximum(np.arange(T) - w, 0); hi = np.minimum(np.arange(T) + w + 1, T)
        m = (cs[hi] - cs[lo]) / (hi - lo)[:, None]
        lab[s0:e0 + 1] = (m >= 0.5).astype(np.uint8)
        conf = np.maximum(m, 1 - m) >= agree
        ok[s0:e0 + 1] = conf if per_bit else conf.all(1)
    return lab, ok


def pattern_modes(keys):
    """Label-free event types from noisy flip patterns. keys: TRAIN patterns as sorted flipped-bit index strings ("" =
    no flip). Every pattern climbs to its most frequent neighbour at Hamming distance 1 (TRAIN counts) until no
    neighbour is more frequent; the local maxima (modes) are the types, kept above the 2-means split of log10(mode count)
    over TRAIN events (each event votes with the count of its mode). -> (kept mode keys, type_of(key) -> index or -1,
    info). Unseen patterns climb with the TRAIN counts."""
    cnt = {}
    for k in keys:
        if k:
            fs = frozenset(map(int, k.split(",")))
            cnt[fs] = cnt.get(fs, 0) + 1
    flip_bits = frozenset().union(*cnt) if cnt else frozenset()
    memo = {}

    def climb(fs):
        path = []
        while fs not in memo:
            path.append(fs)
            best, bc = fs, cnt.get(fs, 0)
            for b in flip_bits | fs:                                          # unseen bits can only be removed
                q = fs ^ {b}
                if cnt.get(q, 0) > bc:
                    best, bc = q, cnt[q]
            if best == fs:
                memo[fs] = fs
                break
            fs = best
        for x in path:
            memo[x] = memo[fs]
        return memo[fs]

    mc = {}
    for fs, c_ in cnt.items():
        m = climb(fs)
        mc[m] = mc.get(m, 0) + c_
    votes = np.log10(np.repeat([mc[climb(fs)] for fs in cnt], [cnt[fs] for fs in cnt])) if cnt else np.zeros(0)
    thr, sep, _ = two_means_threshold(votes) if len(mc) > 10 else (np.log10(10), 0.0, 0.0)
    modes = sorted((m for m in mc if np.log10(mc[m]) > thr), key=lambda m: -mc[m])
    to_e = {m: i for i, m in enumerate(modes)}

    def type_of(k):
        return to_e.get(climb(frozenset(map(int, k.split(",")))), -1) if k else -1

    info = {"rule": "modes", "distinct_patterns": len(cnt), "modes": len(mc), "size": len(modes), "count_split": float(10 ** thr),
            "ashman_D": float(sep), "counts": [mc[m] for m in modes],
            "covered_train_events": float(sum(mc[m] for m in modes) / max(1, len(keys))), "train_events": len(keys)}
    return [",".join(map(str, sorted(m))) for m in modes], type_of, info


def episode_index(cache, split, n_ep):
    term = np.load(cache / f"{split}_terminals.npy")
    ends = np.flatnonzero(term)[:n_ep]
    starts = np.r_[0, ends[:-1] + 1]
    ep = np.repeat(np.arange(len(starts)), ends - starts + 1)
    return starts, ends, ep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True, help="sm2_events.py output dir")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--family", choices=("cube", "puzzle", "scene"), required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--margin", type=int, default=3)
    ap.add_argument("--labels", choices=("segment", "smooth"), default="smooth")
    ap.add_argument("--smooth-w", type=int, default=7)
    ap.add_argument("--agree", type=float, default=0.8)
    ap.add_argument("--label-mask", choices=("frame", "bit"), default="frame")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--window", type=int, default=10, help="detect_events merge window (build_events.py uses 10)")
    ap.add_argument("--stable", type=int, default=3, help="per-bit debounce frames")
    ap.add_argument("--vocab", choices=("patterns", "modes"), default="patterns")
    ap.add_argument("--reuse", action="store_true", help="load reader.pt and reader_bits_*.npy from --out (events only)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.device == "cuda" and "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    dev = a.device
    torch.manual_seed(0); rng = np.random.default_rng(0)
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    enc_j = json.loads((a.events / "encoder.json").read_text())
    dyn = np.array(enc_j["dyn"])
    width = [1 if len(v) == 2 else len(v) for v in enc_j["values"]]
    bit_token = np.repeat(dyn, width)                                             # token of every state bit
    rep = {"events": str(a.events), "family": a.family}

    data = {}
    for split, n_ep in (("train", a.episodes), ("val", a.val_episodes)):
        ev = dict(np.load(a.events / f"events_{split}.npz"))
        starts, ends, ep = episode_index(a.cache, split, n_ep)
        n = int(ends[-1] + 1)
        if a.labels == "segment":
            lab, ok = pseudo_labels(ev, starts, ends, a.margin)
        else:
            enc = Encoder(dyn, enc_j["values"], FSQ().values(torch.arange(5 ** 6)).numpy())
            sc = np.load(a.events / f"see_codes_{split}.npy", mmap_mode="r")
            pf = np.concatenate([enc.encode(np.asarray(sc[s:min(s + 50000, n)], np.int64)[:, dyn]) for s in range(0, n, 50000)])
            lab, ok = smooth_labels(pf, starts, ends, a.smooth_w, a.agree, per_bit=(a.label_mask == "bit"))
        obs = Frames(a.cache / f"{split}_observations.npy", n, ram=(split == "val"))
        data[split] = (obs, lab, ok, starts, ends, ep, n)
        rep[f"labelled_{split}"] = float(ok.mean())
    K = data["train"][1].shape[1]
    vobs, vlab, vok, vst, ven, vep, vn = data["val"]
    if a.reuse:
        bits = {sp: np.load(a.out / f"reader_bits_{sp}.npy") for sp in ("train", "val")}
        rep["train_log"] = json.loads((a.out / "reader_report.json").read_text()).get("train_log") if (a.out / "reader_report.json").exists() else None
    else:
        bits = train_reader(a, data, K, bit_token, rep, dev, rng, t0)
    vb = bits["val"]
    rep["val_vs_pseudo"] = vs_pseudo(vb, vlab, vok)
    events_and_report(a, data, bits, K, bit_token, rep, t0)


def vs_pseudo(b, lab, ok):
    """agreement with the pseudo-labels on labelled bits; frame_exact over frames with a label (every labelled bit right)."""
    m = ok if ok.ndim == 2 else np.repeat(ok[:, None], lab.shape[1], 1)
    rows = m.any(1)
    return {"bit": float((b == lab)[m].mean()), "frame_exact": float(((b == lab) | ~m).all(1)[rows].mean())}


def train_reader(a, data, K, bit_token, rep, dev, rng, t0):
    reader = make_reader(K).to(dev)
    opt = torch.optim.AdamW(reader.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) * 0.5 * (1 + np.cos(np.pi * min(s, a.steps) / a.steps)))
    obs, lab, ok, *_ = data["train"]
    tr_idx = np.flatnonzero(ok.any(1) if ok.ndim == 2 else ok)
    vobs, vlab, vok, vst, ven, vep, vn = data["val"]
    vrows = np.flatnonzero(vok.any(1) if vok.ndim == 2 else vok)
    v_idx = vrows[np.random.default_rng(1).permutation(len(vrows))[:5000]]
    log = []
    for step in range(a.steps):
        reader.train()
        i = np.sort(rng.choice(tr_idx, a.batch))
        px = to_t(obs[i], dev)
        pad = F.pad(px, (2, 2, 2, 2), mode="replicate")
        dx, dy = rng.integers(0, 5, 2)
        px = pad[:, :, dy:dy + 64, dx:dx + 64]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
            logit = reader(px)
        if ok.ndim == 2:                                                       # --label-mask bit: masked BCE
            w_ = torch.as_tensor(ok[i], device=dev).float()
            loss = (F.binary_cross_entropy_with_logits(logit.float(), torch.as_tensor(lab[i], device=dev).float(), reduction="none")
                    * w_).sum() / w_.sum().clamp(min=1)
        else:
            loss = F.binary_cross_entropy_with_logits(logit.float(), torch.as_tensor(lab[i], device=dev).float())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if step % 2000 == 0 or step == a.steps - 1:
            reader.eval()
            with torch.no_grad():
                p = torch.cat([reader(to_t(vobs[v_idx[s:s + 512]], dev)).float() for s in range(0, len(v_idx), 512)]).cpu().numpy() > 0
            vp = vs_pseudo(p, vlab[v_idx], vok[v_idx])
            row = {"step": step, "loss": float(loss), "val_bit": vp["bit"], "val_frame_exact": vp["frame_exact"], "min": round((time.time() - t0) / 60, 1)}
            log.append(row); print(json.dumps(row), flush=True)
    torch.save({"reader": reader.state_dict(), "bits": K, "bit_token": bit_token, "events_dir": str(a.events)}, a.out / "reader.pt")
    rep["train_log"] = log
    reader.eval()
    bits = {}
    for split, (o, _, _, st, en, ep, n) in data.items():
        out = np.zeros((n, K), np.uint8)
        with torch.no_grad():
            for s in range(0, n, 1024):
                out[s:s + 1024] = (reader(to_t(o[s:min(s + 1024, n)], dev)).float() > 0).cpu().numpy()
        bits[split] = out
        np.save(a.out / f"reader_bits_{split}.npy", out)
    return bits


def events_and_report(a, data, bits, K, bit_token, rep, t0):
    vb = bits["val"]
    evs = {}
    for split, (o, _, _, st, en, ep, n) in data.items():
        t_start, t_end, c = detect_events(bits[split], ep, a.stable, a.window)
        pat = (c[t_end + 1] != c[t_start]).astype(np.uint8)
        evs[split] = (t_start, t_end, c, pat, ep)
    key = lambda p_: ",".join(map(str, np.flatnonzero(p_)))                     # flipped bits (np bytes drop trailing zeros)
    keys_tr = [key(p) for p in evs["train"][3]]
    if a.vocab == "modes":
        keep, type_of, vinfo = pattern_modes(keys_tr)
    else:
        uniq, cnt = np.unique(np.array(keys_tr), return_counts=True)
        lc = np.log10(cnt)
        thr, sep, _ = two_means_threshold(lc) if len(lc) > 10 else (np.log10(10), 0.0, 0.0)
        keep = [u for u, c_ in zip(uniq, lc) if c_ > thr and u != ""]
        to_e = {u: i for i, u in enumerate(keep)}
        type_of = lambda k: to_e.get(k, -1)
        vinfo = {"rule": "patterns", "size": int(len(keep)), "count_split": float(10 ** thr), "ashman_D": float(sep),
                 "counts": sorted(cnt[lc > thr].tolist(), reverse=True), "dropped_events": int(cnt[lc <= thr].sum()), "train_events": int(cnt.sum())}
    vocab = np.zeros((len(keep), K), np.uint8)
    for i_, u in enumerate(keep):
        vocab[i_, list(map(int, u.split(",")))] = 1
    tmaps = np.zeros((len(keep), 64), np.float32)
    for e, v in enumerate(vocab):
        m = np.zeros(256, bool); m[bit_token[v.astype(bool)]] = True
        tmaps[e] = m.reshape(8, 2, 8, 2).any((1, 3)).ravel()
    rep["vocab"] = vinfo
    np.save(a.out / "vocab.npy", vocab); np.save(a.out / "tmaps.npy", tmaps); np.save(a.out / "bit_mask.npy", np.ones(K, bool))
    for split, (t_start, t_end, c, pat, ep) in evs.items():
        e = np.array([type_of(key(p)) for p in pat], np.int64)
        seg = np.r_[0, t_end[:-1] + 1]
        new_ep = np.r_[True, ep[t_start[1:]] != ep[t_start[:-1]]]
        st = data[split][3]
        seg[new_ep] = st[ep[t_start[new_ep]]]
        np.savez_compressed(a.out / f"events_{split}.npz", codes=c, t=t_end, t_start=t_start, e=e, seg_start=seg,
                            before=c[t_start], after=c[t_end + 1], episode=ep[t_end])
        rep[f"events_{split}"] = {"events": int(len(e)), "per_episode": float(len(e) / (ep[-1] + 1)), "in_vocab": float((e >= 0).mean()) if len(e) else None}
    # PRIVILEGED scoring on VAL
    vst, ven, vep, vn = data["val"][3], data["val"][4], data["val"][5], data["val"][6]
    segs, _, _ = reference_segments(a.cache, a.family, vn, vst, int(ven[0] - vst[0] + 1))
    t_start, t_end, c, pat, _ = evs["val"]
    rep["val_events_vs_reference"] = {"reference_per_episode": float(len(segs) / len(vst)),
                                      "span_w15": score_spans(np.c_[t_start, t_end], segs, 15), "span_w30": score_spans(np.c_[t_start, t_end], segs, 30)}
    if a.family == "puzzle":
        bs = np.asarray(np.load(a.cache / "val_button_states.npy", mmap_mode="r")[:vn]).astype(np.float64)
        h = vn // 2
        Xb = np.c_[vb.astype(np.float64), np.ones(vn)]
        W = np.linalg.lstsq(Xb[:h], bs[:h], rcond=None)[0]
        acc = ((Xb[h:] @ W > 0.5) == (bs[h:] > 0.5)).mean(0)
        rep["val_light_acc_linear"] = {"per_light": np.round(acc, 4).tolist(), "min": float(acc.min()), "mean": float(acc.mean()),
                                       "frame_all_lights": float(((Xb[h:] @ W > 0.5) == (bs[h:] > 0.5)).all(1).mean())}
        q = np.asarray(np.load(a.cache / "val_qpos.npy", mmap_mode="r")[:vn, 14:34])
        pressed = np.where(q.min(1) < -0.015, q.argmin(1), -1)
        e_val = np.array([type_of(key(p)) for p in pat], np.int64)
        pb = []
        for ts, te in zip(t_start, t_end):
            w = pressed[max(ts - 12, 0):te + 3]
            u, cc = np.unique(w[w >= 0], return_counts=True)
            pb.append(int(u[cc.argmax()]) if len(u) else -1)
        pb = np.array(pb)
        m = (e_val >= 0) & (pb >= 0)
        pur = []
        for e in np.unique(e_val[m]):
            u, cc = np.unique(pb[m & (e_val == e)], return_counts=True)
            pur.append(cc.max() / cc.sum())
        rep["val_type_to_pressed_button_purity"] = {"types_seen": int(len(pur)), "mean": float(np.mean(pur)) if pur else None,
                                                    "min": float(np.min(pur)) if pur else None,
                                                    "distinct_buttons_covered": int(len(np.unique(pb[m])))}
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    (a.out / "reader_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "train_log"}), flush=True)


if __name__ == "__main__":
    main()
