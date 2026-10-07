#!/usr/bin/env python3
"""Stream the first N episodes of an OGBench .npz dataset into uncompressed .npy files.

np.load on a compressed .npz decompresses a whole array (37-61 GB of frames for puzzle
4x5/4x6). Streaming the zip member keeps memory low, and later jobs memory-map the .npy
files instead of holding the frames in RAM. `button_states` is kept for evaluation only.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as F

KEYS = ("observations", "actions", "terminals", "button_states", "qpos")   # last two optional (privileged)


def stream_head(npz: Path, key: str, n_rows: int | None, out: Path) -> int:
    with zipfile.ZipFile(npz) as z, z.open(f"{key}.npy") as fh:
        version = F.read_magic(fh)
        read = F.read_array_header_1_0 if version == (1, 0) else F.read_array_header_2_0
        shape, fortran, dtype = read(fh)
        assert not fortran
        n = shape[0] if n_rows is None else min(n_rows, shape[0])
        mm = F.open_memmap(out, mode="w+", dtype=dtype, shape=(n,) + tuple(shape[1:]))
        row = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        step = max(1, (256 << 20) // max(row, 1))
        for i in range(0, n, step):
            k = min(step, n - i)
            mm[i:i + k] = np.frombuffer(fh.read(k * row), dtype).reshape((k,) + tuple(shape[1:]))
        mm.flush()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--env", required=True)
    ap.add_argument("--train-episodes", type=int, default=1500)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = a.out / a.env
    out.mkdir(parents=True, exist_ok=True)
    info = {}
    for split, src in (("train", a.data / f"{a.env}.npz"), ("val", a.data / f"{a.env}-val.npz")):
        term = np.load(src)["terminals"]               # small member; decompressed alone
        ends = np.nonzero(term)[0]
        n = int(ends[a.train_episodes - 1] + 1) if split == "train" and len(ends) >= a.train_episodes else len(term)
        members = {m[:-4] for m in zipfile.ZipFile(src).namelist()}
        for key in KEYS:
            if key in members:
                stream_head(src, key, n, out / f"{split}_{key}.npy")
        info[split] = {"frames": n, "episodes": int(term[:n].sum())}
        print(split, info[split], flush=True)
    (out / "cache_info.json").write_text(json.dumps(info, indent=1) + "\n")


if __name__ == "__main__":
    main()
