#!/usr/bin/env python3
"""PRIVILEGED diagnostic (puzzle): presses per button in the first N train episodes, and the acted-identity counts
of a u_events run mapped to buttons (identity -> button by u_button_probe-style correlation on val)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--episodes", type=int, default=150)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--cols", type=int, default=5)
    a = ap.parse_args()
    term = np.load(a.cache / "train_terminals.npy")
    n = int(np.nonzero(term)[0][a.episodes - 1]) + 1
    B = np.asarray(np.load(a.cache / "train_button_states.npy", mmap_mode="r")[:n]).astype(int)
    R, C = a.rows, a.cols

    def cross(i):
        r, c = divmod(i, C)
        return {i} | {rr * C + cc for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if 0 <= rr < R and 0 <= cc < C}

    cnt = np.zeros(R * C, int); toggled = np.zeros(R * C, int)
    for t in np.nonzero((B[1:] != B[:-1]).any(1))[0] + 1:
        S = set(np.nonzero(B[t] != B[t - 1])[0].tolist())
        cnt[max(S, key=lambda i: len(cross(i) & S))] += 1
        for i in S:
            toggled[i] += 1
    ev = np.load(a.run / "events" / "events_train.npz")
    print(json.dumps({"presses_per_button": cnt.reshape(R, C).tolist(), "toggles_per_button": toggled.reshape(R, C).tolist(),
                      "acted_identity_counts": np.bincount(ev["e"], minlength=ev["before"].shape[1]).tolist()}))


if __name__ == "__main__":
    main()
