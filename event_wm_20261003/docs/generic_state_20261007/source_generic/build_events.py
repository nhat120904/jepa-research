#!/usr/bin/env python3
"""Encode every cached frame with the event code and extract events (no state labels).

An event is a change of the slow code bits between frames t and t + 1 that is stable for m
frames on both sides. Its type is the XOR pattern of the change; the vocabulary keeps the most
frequent patterns covering --coverage of the detected events. Output per split (events_<split>.npz):
codes of every frame, event indices, types, codes before/after, and the start of the segment
that leads to each event (the frame after the previous event or the episode start).

Privileged button states are used only to report how event types map to true buttons.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from common import detect_events, load_code, pack, save_json, size_of, tokens
from lightsout import solver


def encode_split(enc, logits, bits, obs, dev, bs=4096):
    import torch

    out = np.empty((len(obs), bits), np.float16)
    with torch.no_grad():
        for i in range(0, len(obs), bs):
            frames = np.asarray(obs[i:i + bs])
            out[i:i + len(frames)] = torch.sigmoid(logits(tokens(enc, frames, dev))).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--code", type=Path, required=True)
    ap.add_argument("--stable", type=int, default=3, help="per-bit debounce (frames)")
    ap.add_argument("--window", type=int, default=10, help="merge code changes closer than this into one event")
    ap.add_argument("--coverage", type=float, default=0.999)
    ap.add_argument("--all-bits", action="store_true", help="use all code bits instead of the slow subset")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    dev = "cuda"
    rows, cols = size_of(a.env)
    enc, logits, ck = load_code(a.code, dev)
    sel = np.ones(ck["bits"], bool) if a.all_bits else ck["slow_mask"].numpy().astype(bool)
    A = solver(rows, cols)[0]
    col = {pack(A[:, i][None])[0]: i for i in range(A.shape[1])}
    a.out.mkdir(parents=True, exist_ok=True)
    report, vocab = {"bits_used": int(sel.sum())}, None
    for split in ("train", "val"):
        obs = np.load(a.cache / f"{split}_observations.npy", mmap_mode="r")
        term = np.load(a.cache / f"{split}_terminals.npy")
        bs_true = np.load(a.cache / f"{split}_button_states.npy").astype(np.uint8)
        ep = np.concatenate([[0], np.cumsum(term[:-1])]).astype(np.int64)
        p = encode_split(enc, logits, ck["bits"], obs, dev)
        raw = (p[:, sel] > 0.5).astype(np.uint8)
        ts, t, b = detect_events(raw, ep, a.stable, a.window)      # b = debounced code
        pat = b[t + 1] ^ b[ts]
        if vocab is None:                                   # vocabulary from the train split only
            u, inv, cnt = np.unique(pack(pat), return_inverse=True, return_counts=True)
            order = np.argsort(-cnt)
            cov = np.cumsum(cnt[order]) / cnt.sum()
            cand = order[: int(np.searchsorted(cov, a.coverage) + 1)]
            # Real event types recur at similar rates; code-noise patterns are rare. 2-means on log count.
            lc = np.log(cnt[cand].astype(float))
            c2 = np.array([lc.min(), lc.max()])
            for _ in range(50):
                lab = np.abs(lc[:, None] - c2[None]).argmin(1)
                c2 = np.array([lc[lab == j].mean() if (lab == j).any() else c2[j] for j in range(2)])
            keep = cand[lab == 1] if (lab == 0).any() and c2[1] - c2[0] > np.log(4) else cand
            report["vocab_dropped_counts"] = sorted(cnt[np.setdiff1d(cand, keep)].tolist(), reverse=True)[:20]
            first = {k: i for i, k in enumerate(pack(pat))}
            vocab = np.stack([pat[first[u[k]]] for k in keep])
            report["vocab_size"] = int(len(vocab))
            report["vocab_counts"] = cnt[keep].tolist()
        vkeys = {k: i for i, k in enumerate(pack(vocab))}
        e = np.array([vkeys.get(k, -1) for k in pack(pat)], np.int32)
        ep_first = np.r_[0, np.nonzero(term)[0] + 1]           # first frame of each episode
        seg = ep_first[ep[t]].copy()                           # segment leading to each event:
        same_ep = np.r_[False, ep[t][1:] == ep[t][:-1]]        # previous event's end + 1 .. this start
        seg[same_ep] = t[:-1][same_ep[1:]] + 1
        seg = np.minimum(seg, ts)
        # privileged: true button of each event (nearest true toggle within 3 frames) for reporting
        true_t = np.nonzero((bs_true[1:] != bs_true[:-1]).any(1) & (ep[1:] == ep[:-1]))[0]
        jr = np.searchsorted(true_t, ts - 5)
        j = np.clip(jr, 0, len(true_t) - 1)
        near = (jr < len(true_t)) & (true_t[j] <= t + 5)
        btn = np.array([col.get(k, -1) for k in pack(bs_true[true_t[j] + 1] ^ bs_true[true_t[j]])])
        btn[~near] = -1
        np.savez_compressed(a.out / f"events_{split}.npz", codes=b, t=t, t_start=ts, e=e, seg_start=seg,
                            before=b[ts], after=b[t + 1], true_button=btn, episode=ep[t])
        r = {"frames": int(len(b)), "events": int(len(t)), "in_vocab": float((e >= 0).mean()),
             "events_per_episode": float(len(t) / (ep[-1] + 1)), "true_events": int(len(true_t)),
             "matched_true": float(near.mean())}
        ok = (e >= 0) & (btn >= 0)
        if ok.any():
            m = np.zeros((len(vocab), rows * cols), np.int64)
            np.add.at(m, (e[ok], btn[ok]), 1)
            r["type_to_button_purity"] = float(m.max(1).sum() / m.sum())
            r["type_to_button"] = m.argmax(1).tolist()
            r["buttons_covered"] = int(len(np.unique(m.argmax(1)[m.sum(1) > 0])))
        report[split] = r
        print(split, r, flush=True)
    np.save(a.out / "vocab.npy", vocab)
    np.save(a.out / "bit_mask.npy", sel)
    save_json(a.out / "events_report.json", report)


if __name__ == "__main__":
    main()
