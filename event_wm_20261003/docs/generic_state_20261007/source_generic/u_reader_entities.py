#!/usr/bin/env python3
"""Self-training: the reader (trained on the SAM 2 pseudo-labels of a subset) labels many more episodes.

Writes the u_events.py entity table for the first N train / M val episodes from the reader alone:
  pos, app   reader state of every identity;
  area       on sampled frames: 1 where the reader's at-rest head is positive (identity observed at rest),
             else 0 -- in transit, covered or occluded frames are unobserved, as in the front end;
  agent      packed reader agent mask (agent head, sigmoid > 1/2);
  processed  every `stride`-th frame of those episodes (the same temporal sampling as the front end).
discover.json is copied from the SAM 2 front end (identities, spread for thr_pos).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from u_reader import make_reader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--front", type=Path, required=True, help="SAM 2 front-end dir (discover.json)")
    ap.add_argument("--reader", type=Path, required=True)
    ap.add_argument("--train-episodes", type=int, default=1000)
    ap.add_argument("--val-episodes", type=int, default=100)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("runs under sbatch")
    import torch

    rk = torch.load(a.reader, map_location="cpu", weights_only=False)
    if not rk.get("agent", False):
        raise SystemExit("reader has no agent head (train u_reader.py with --agent-head)")
    K = rk["K"]
    reader = make_reader(K, agent=True).to(a.device).eval(); reader.load_state_dict(rk["reader"])
    a.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(a.front / "discover.json", a.out / "discover.json")
    rep = {}
    for split, n_ep in (("train", a.train_episodes), ("val", a.val_episodes)):
        obs = np.load(a.cache / f"{split}_observations.npy", mmap_mode="r")
        term = np.load(a.cache / f"{split}_terminals.npy")
        n = len(term)
        starts = np.r_[0, np.nonzero(term)[0] + 1]; ends = np.r_[np.nonzero(term)[0] + 1, n]
        starts, ends = starts[starts < ends], ends[starts < ends]
        pos = np.zeros((n, K, 2), np.float32); app = np.zeros((n, K, 3), np.float32); area = np.zeros((n, K), np.int16)
        agent = np.zeros((n, 64, 8), np.uint8); processed = np.zeros(n, bool)
        with torch.no_grad(), torch.autocast(a.device, dtype=torch.bfloat16):
            for e in range(min(n_ep, len(starts))):
                s0, s1 = starts[e], ends[e]
                x = torch.as_tensor(np.array(obs[s0:s1]), device=a.device).permute(0, 3, 1, 2).float().div_(255.0)
                st, rest, ag = reader(x, return_agent=True)
                st, rest, ag = st.float().cpu().numpy(), rest.float().cpu().numpy(), (ag.float() > 0).cpu().numpy()
                pos[s0:s1], app[s0:s1] = st[..., :2], st[..., 2:5]
                agent[s0:s1] = np.packbits(ag, axis=-1)
                fs = np.arange(s0, s1, a.stride)                                  # observations on sampled frames only
                processed[fs] = True
                area[fs] = (rest[fs - s0] > 0).astype(np.int16)
        np.savez(a.out / f"entities_{split}.npz", pos=pos, app=app, area=area, agent=agent, processed=processed)
        rep[split] = {"episodes": int(min(n_ep, len(starts))), "processed_frames": int(processed.sum()),
                      "observed_at_rest_frac": float((area[processed] > 0).mean())}
        print(split, json.dumps(rep[split]), flush=True)
    (a.out / "reader_entities.json").write_text(json.dumps(rep, indent=1) + "\n")


if __name__ == "__main__":
    main()
