#!/usr/bin/env python3
"""Run TD-JEPA's train.py while hashing the actual batches the loader yields.

Batch-stream equality is a more direct criterion than weight equality: if two
configurations feed the model different data, nothing downstream can be
compared. Weights are then only consulted when the streams already match.

What is hashed: the tensors a ``DataLoader`` yields, i.e. after ``__getitem__``
and after ``collate``, so any randomness inside the dataset or its transforms is
captured -- not just sample indices. Model-side GPU transforms
(``on_after_batch_transfer``) are identical across the configurations compared
here and are outside the loader, so they are not hashed.

The upstream repository is not modified: ``DataLoader.__iter__`` is patched at
runtime and ``train.py`` is then executed as ``__main__``.

Env: ``BATCH_HASH_OUT`` (path), ``BATCH_HASH_LIMIT`` (default 200).
Remaining argv is forwarded to ``train.py`` as Hydra overrides.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

OUT = Path(os.environ["BATCH_HASH_OUT"])
LIMIT = int(os.environ.get("BATCH_HASH_LIMIT", "200"))
TRAIN = Path(os.environ.get("TDJEPA_TRAIN", "train.py"))

_hashes: list[str] = []


def _feed(h: hashlib._Hash, obj) -> None:
    """Fold an arbitrary batch structure into the digest, order-sensitively."""
    if torch.is_tensor(obj):
        t = obj.detach()
        h.update(str(tuple(t.shape)).encode())
        h.update(str(t.dtype).encode())
        h.update(np.ascontiguousarray(t.cpu().numpy()).tobytes())
    elif isinstance(obj, np.ndarray):
        h.update(str(obj.shape).encode())
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, dict):
        for k in sorted(obj, key=str):
            h.update(str(k).encode())
            _feed(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _feed(h, v)
    elif isinstance(obj, (str, bytes, int, float, bool)) or obj is None:
        h.update(repr(obj).encode())
    else:
        h.update(f"<unhashable:{type(obj).__name__}>".encode())


def _flush() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"limit": LIMIT, "n": len(_hashes),
                               "hashes": _hashes}, indent=1))


_orig_iter = DataLoader.__iter__


def _hashing_iter(self):
    for batch in _orig_iter(self):
        if len(_hashes) < LIMIT:
            h = hashlib.sha256()
            _feed(h, batch)
            _hashes.append(h.hexdigest())
            if len(_hashes) == LIMIT:
                _flush()
        yield batch


DataLoader.__iter__ = _hashing_iter

try:
    sys.argv = [str(TRAIN)] + sys.argv[1:]
    runpy.run_path(str(TRAIN), run_name="__main__")
finally:
    _flush()
