#!/usr/bin/env python3
"""Component 5b (method/README.md): the VIEW, a frame with the agent removed. One rule for every environment.

Why. OGBench draws the arm ~90% transparent: the current frame shows every button's state (PRIVILEGED probe on the raw
4 x 4 patch colours of held-out puzzle-3x3 VAL frames: accuracy >= .999 for all 9 buttons), yet the object rules
(objects.py) observed nothing under the dilated agent mask (42% of the tokens on puzzle-3x3), so places the arm rests
over were rarely or never seen agent-free (held-out puzzle-3x3: buttons 0 and 1 never discovered, button 4 visible in
1.7% of VAL frames; puzzle-4x4: button 2 missing).

Rule. A token in the dilated agent mask (segmenter p > .5 dilated by one token, the agent-free test of the memory rule,
component 6) shows the EXEMPLAR of its SeeThrough code (component 5): the 4 x 4 patch seen most often with that code at
that token in agent-free TRAIN frames. It shows it when the SeeThrough probability is > .5 and the code was seen
agent-free at that token; otherwise it is HIDDEN (unobserved). Tokens outside the mask keep their pixels.
Exemplars keep what the object rules rely on: deterministic rendering repeats a fixed thing's state exactly, so one state
is one exact patch, and a state read through the arm equals the state read in full view. Decoding the codes with the
SceneCodes decoder would mix in neighbouring codes (receptive field ~6 tokens, the arm's among them) and break exact
recurrence.
The agent itself is still located with the segmenter mask (contact, acted entity, a mover at rest = clear of the agent):
the view only decides what is observed.

Outputs (--out): view_table.npz (keys = token * 5^6 + code, sorted; patches (n, 48) uint8 = 4 x 4 x 3; counts; purity =
share of the key's agent-free observations equal to its exemplar), view_report.json, view_panels.png (VAL frames: frame,
agent mask, view, hidden tokens).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from utils import episode_bounds, save_json

NCODE = 5 ** 6                                                                    # SeeThrough learned space: 6 digits x 5 levels
_MIX = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93, 0xFF51AFD7ED558CCD,
                 0xC4CEB9FE1A85EC53], np.uint64)


def agent_tokens(seg, idx, dilate=1):
    """dilated segmenter agent mask (n, 256) bool of frames idx (the agent-free test of component 6)."""
    from frontend import dilate_tokens
    return dilate_tokens(np.asarray(seg[idx]) > 127, dilate)


def patches(X):
    """frames (n, 64, 64, 3) uint8 -> token patches (n, 256, 48) (token = row * 16 + column, as the 16 x 16 grid)."""
    n = len(X)
    return np.ascontiguousarray(np.asarray(X).reshape(n, 16, 4, 16, 4, 3).transpose(0, 1, 3, 2, 4, 5)).reshape(n, 256, 48)


def unpatch(V):
    n = len(V)
    return np.ascontiguousarray(V.reshape(n, 16, 16, 4, 4, 3).transpose(0, 1, 3, 2, 4, 5)).reshape(n, 64, 64, 3)


def hash_patches(pt):
    """(..., 48) uint8 -> (...) uint64 hash of the exact patch."""
    v = np.ascontiguousarray(pt).view(np.uint64)                                 # (..., 6)
    with np.errstate(over="ignore"):
        h = (v * _MIX).sum(-1, dtype=np.uint64)
        h ^= h >> np.uint64(31)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(29)
    return h


def build_table(obs, C, seg, frames, dilate=1, chunk=2000):
    """exemplar per (token, SeeThrough code) from the agent-free tokens of `frames` -> (keys, patches, counts, purity)."""
    keys, hs, src = [], [], []
    tok = np.arange(256, dtype=np.int64)
    for s in range(0, len(frames), chunk):
        f = frames[s:s + chunk]
        free = ~agent_tokens(seg, f, dilate)
        k = tok[None] * NCODE + np.asarray(C[f]).astype(np.int64)
        keys.append(k[free]); hs.append(hash_patches(patches(np.asarray(obs[f])))[free])
        src.append((f[:, None].astype(np.int64) * 256 + tok[None])[free])
    key, h, src = np.concatenate(keys), np.concatenate(hs), np.concatenate(src)
    o = np.lexsort((h, key)); key, h, src = key[o], h[o], src[o]
    new = np.r_[True, (key[1:] != key[:-1]) | (h[1:] != h[:-1])]
    ps = np.flatnonzero(new); pc = np.diff(np.r_[ps, len(key)]); pk = key[ps]
    o2 = np.lexsort((-pc, pk))                                                    # per key the most frequent patch first
    pk2, pc2, ps2 = pk[o2], pc[o2], ps[o2]
    first = np.r_[True, pk2[1:] != pk2[:-1]]
    ukeys, best, at = pk2[first], pc2[first], src[ps2[first]]
    tot = np.add.reduceat(pc2, np.flatnonzero(first))
    pts = np.zeros((len(ukeys), 48), np.uint8)
    fr, tk = np.divmod(at, 256)
    o3 = np.argsort(fr, kind="stable"); frs = fr[o3]
    uf_all = np.unique(frs)
    for s in range(0, len(uf_all), chunk):                                        # read the chosen patches, frames in order
        uf = uf_all[s:s + chunk]
        sel = o3[np.searchsorted(frs, uf[0], "left"):np.searchsorted(frs, uf[-1], "right")]
        pts[sel] = patches(np.asarray(obs[uf]))[np.searchsorted(uf, fr[sel]), tk[sel]]
    return ukeys, pts, tot, best / tot


def make_view(X, codes, prob, agt, keys, pts, force=False):
    """frames X (n, 64, 64, 3) uint8, SeeThrough codes (n, 256) and probabilities (n, 256) (uint8 p x 255), dilated agent
    tokens agt (n, 256) -> view (n, 64, 64, 3) uint8, hidden tokens (n, 256) bool. force: every agent token with a known
    code shows its exemplar whatever its probability (a static goal image: see closed_loop_objects.py)."""
    k = np.arange(256, dtype=np.int64)[None] * NCODE + np.asarray(codes).astype(np.int64)
    i = np.minimum(np.searchsorted(keys, k), len(keys) - 1)
    use = agt & (keys[i] == k)
    if not force:
        use &= np.asarray(prob) > 127
    V = patches(X).copy()
    V[use] = pts[i[use]]
    return unpatch(V), agt & ~use


def token_pixels(m):
    """(n, 256) bool -> (n, 64, 64) bool."""
    m = np.asarray(m).reshape(-1, 16, 16)
    return np.repeat(np.repeat(m, 4, 1), 4, 2)


class Source:
    """frames as the object rules see them. Without a table: the raw frames, observed outside the dilated agent mask (the
    original rule). With a table: views, observed outside their hidden tokens. agent(idx): the dilated agent mask in
    pixels, always from the segmenter (where the agent is)."""

    def __init__(self, obs, seg, table=None, codes=None, prob=None, dilate=1):
        self.obs, self.seg, self.dilate = obs, seg, dilate
        self.keys = self.pts = None
        if table is not None:
            z = np.load(table)
            self.keys, self.pts = z["keys"], z["patches"]
            self.codes, self.prob = codes, prob

    def get(self, idx):
        """-> frames as seen (n, 64, 64, 3) uint8, unobserved pixels (n, 64, 64) bool."""
        idx = np.atleast_1d(np.asarray(idx))
        X = np.asarray(self.obs[idx])
        agt = agent_tokens(self.seg, idx, self.dilate)
        if self.keys is None:
            return X, token_pixels(agt)
        V, hid = make_view(X, np.asarray(self.codes[idx]), np.asarray(self.prob[idx]), agt, self.keys, self.pts)
        return V, token_pixels(hid)

    def raw(self, idx):
        """-> raw frames, dilated agent mask in pixels."""
        idx = np.atleast_1d(np.asarray(idx))
        return np.asarray(self.obs[idx]), token_pixels(agent_tokens(self.seg, idx, self.dilate))


def panels(src, idx, path):
    from PIL import Image
    rows = []
    for t in idx:
        x, ag = src.raw([t]); v, hid = src.get([t])
        m = x[0].copy(); m[ag[0]] = (0.5 * m[ag[0]] + [127, 0, 0]).astype(np.uint8)
        h = v[0].copy(); h[hid[0]] = (0.5 * h[hid[0]] + [127, 0, 0]).astype(np.uint8)
        rows.append(np.concatenate([x[0], np.full((64, 2, 3), 255, np.uint8), m, np.full((64, 2, 3), 255, np.uint8), v[0],
                                    np.full((64, 2, 3), 255, np.uint8), h], 1))
        rows.append(np.full((2, rows[-1].shape[1], 3), 255, np.uint8))
    img = np.concatenate(rows, 0)
    Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="front-end dir (seg_{split}.npy)")
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--see", type=Path, required=True, help="dir with see_codes_{split}.npy / see_prob_{split}.npy (memory_entities.py)")
    ap.add_argument("--episodes", type=int, default=1000, help="TRAIN episodes for the exemplars")
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--dilate", type=int, default=1)
    ap.add_argument("--min-purity", type=float, default=0.5,
                    help="an exemplar is used only when it is MORE THAN this share of its key's agent-free patches (exact recurrence holds); other keys are not in the table, so their tokens stay hidden (0 keeps every key)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch / local/run_stage.ps1")
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    st, en = episode_bounds(a.cache, "train", a.episodes)
    frames = np.concatenate([np.arange(s0, e0 + 1, a.stride) for s0, e0 in zip(st, en)])
    obs = np.load(a.cache / "train_observations.npy", mmap_mode="r")
    seg = np.load(a.run / "seg_train.npy", mmap_mode="r")
    C = np.load(a.see / "see_codes_train.npy", mmap_mode="r")
    keys, pts, cnt, pur = build_table(obs, C, seg, frames, a.dilate)
    # EXACT RECURRENCE GATE: a key whose exemplar is not the majority of its agent-free patches does not show one state
    # (a sliding window / drawer: scene exemplar purity .845 weighted, 13.5% of the key mass below 1/2, vs puzzles and cube
    # .974-.978); its exemplar mixed states and split scene objects into 10 identities (5 exist)
    keep = pur > a.min_purity                                                   # more than half, as the other rules
    dropped = {"keys": int((~keep).sum()), "key_mass": float(cnt[~keep].sum() / max(cnt.sum(), 1))}
    keys, pts, cnt, pur = keys[keep], pts[keep], cnt[keep], pur[keep]
    np.savez(a.out / "view_table.npz", keys=keys, patches=pts, counts=cnt, purity=pur)
    rep = {"run": str(a.run), "cache": str(a.cache), "see": str(a.see), "train_frames": int(len(frames)), "keys": int(len(keys)),
           "tokens_with_keys": int(len(np.unique(keys // NCODE))), "keys_per_token_median": float(np.median(np.bincount(keys // NCODE))),
           "min_purity": a.min_purity, "dropped": dropped, "purity_weighted": float((pur * cnt).sum() / cnt.sum()), "purity_p10_p50": np.percentile(pur, [10, 50]).round(4).tolist(),
           "minutes_table": round((time.time() - t0) / 60, 1)}
    print(json.dumps(rep), flush=True)
    vst, ven = episode_bounds(a.cache, "val", a.val_episodes)
    n = int(ven[-1] + 1)
    vsrc = Source(np.load(a.cache / "val_observations.npy", mmap_mode="r"), np.load(a.run / "seg_val.npy", mmap_mode="r"),
                  a.out / "view_table.npz", np.load(a.see / "see_codes_val.npy", mmap_mode="r"),
                  np.load(a.see / "see_prob_val.npy", mmap_mode="r"), a.dilate)
    smp = np.sort(np.random.default_rng(0).choice(n, min(n, 20000), replace=False))
    ag_frac, hid_frac, shown = [], [], []
    for s in range(0, len(smp), 2000):
        f = smp[s:s + 2000]
        agt = agent_tokens(vsrc.seg, f, a.dilate)
        _, hid = make_view(np.asarray(vsrc.obs[f]), np.asarray(vsrc.codes[f]), np.asarray(vsrc.prob[f]), agt, vsrc.keys, vsrc.pts)
        ag_frac.append(agt.mean(1)); hid_frac.append(hid.mean(1)); shown.append((agt & ~hid).sum(1) / np.maximum(agt.sum(1), 1))
    rep["val"] = {"frames": int(len(smp)), "agent_token_frac": float(np.concatenate(ag_frac).mean()),
                  "hidden_token_frac": float(np.concatenate(hid_frac).mean()),
                  "agent_tokens_shown": float(np.concatenate(shown).mean())}
    panels(vsrc, np.sort(np.random.default_rng(1).choice(n, 12, replace=False)), a.out / "view_panels.png")
    rep["minutes"] = round((time.time() - t0) / 60, 1)
    save_json(a.out / "view_report.json", rep)
    print(json.dumps(rep["val"]), rep["minutes"], flush=True)


if __name__ == "__main__":
    main()
