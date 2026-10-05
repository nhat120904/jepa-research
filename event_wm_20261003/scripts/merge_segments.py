#!/usr/bin/env python3
"""Concatenate sharded segments_{split}.npz files (sam2_frames.py --train-ep-start) into one directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shards", type=Path, nargs="+", required=True, help="dirs holding segments_<split>.npz, in episode order")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    zs = [np.load(d / f"segments_{a.split}.npz") for d in a.shards]
    out = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files}
    assert np.all(np.diff(out["frames"]) > 0), "shards overlap or are out of order"
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez(a.out / f"segments_{a.split}.npz", **out)
    print({k: v.shape for k, v in out.items()})


if __name__ == "__main__":
    main()
