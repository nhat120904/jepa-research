"""Encode CTA4-visited policy banks with the original PushT PCA basis."""
import argparse
import json
from pathlib import Path

import torch

from cta_encode import Tokens, encode_split, shards
from ti_wm.contract import require_compute
from ti_wm.pusht_runtime import VisualScorer


def main(a):
    require_compute()
    train = shards(a.collect, a.train_first, a.train_last)
    dev = shards(a.collect, a.dev_first, a.dev_last)
    if not train or not dev or set(train) & set(dev):
        raise ValueError("Need disjoint, nonempty train and dev collection shards")
    pca = torch.load(a.old_features / "pca.pt", map_location="cpu")
    tok = Tokens(VisualScorer("cuda"))
    tok.mean, tok.basis = pca["pca_mean"].cuda(), pca["pca_basis"].cuda()
    a.out.mkdir(parents=True, exist_ok=True)
    counts = {"train": encode_split(tok, train, a.out / "train"),
              "dev": encode_split(tok, dev, a.out / "dev")}
    for name in ("pca.pt", "goals.npy"):
        (a.out / name).symlink_to((a.old_features / name).resolve())
    (a.out / "features.json").write_text(json.dumps({"decisions": counts,
        "train_roots": [a.train_first, a.train_last], "dev_roots": [a.dev_first, a.dev_last],
        "shards": {"train": [p.name for p in train], "dev": [p.name for p in dev]},
        "pca_source": str(a.old_features.resolve())}, indent=2))
    print(f"ENCODE_OK {counts}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("collect", "old-features", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    for key in ("train-first", "train-last", "dev-first", "dev-last"):
        p.add_argument(f"--{key}", type=int, required=True)
    main(p.parse_args())
