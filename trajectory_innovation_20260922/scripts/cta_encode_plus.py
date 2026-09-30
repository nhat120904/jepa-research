"""Compact round-4 training set: context tokens + frozen source codes for old and new banks.

docs/CTA_ONPOLICY_DATA_PROTOCOL_20260927.md. Reuses the ORIGINAL PCA basis (cta_feat_54490) so the frozen source
encoder / reader / FULL reader of checkpoint 55018 stay valid. Future tokens are encoded only to compute the source
code S with the frozen encoder, then discarded (raw shards keep the images for any later re-encoding).

Output: cur.npy (D, 256, 128) fp16, prev.npy (D, 64, 128) fp16 per decision; banks.npz per bank with ctx_index,
root, decision, t, source (0 old P0 data, 1 new standard bank, 2 new perturbed bank), mixed, chunk, ctx_pos, end_pos,
end_prev_pos, geom (registration label), cov (native coverage), src (K, M) flat FSQ indices.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

from cta_encode import Tokens
from ti_wm.contract import require_compute
from ti_wm.cta import SourceEncoder, proprio
from ti_wm.cta_geometry import registration_score
from ti_wm.pusht_runtime import VisualScorer
from ti_wm.sibling import PCA_DIM, TOKENS

K, SMALL, BATCH = 8, 64, 64


def load_encoder(parent, device):
    blob = torch.load(parent, map_location="cpu")
    cfg = blob["config"]
    enc = SourceEncoder(cfg["m"], conditional=cfg["conditional"], path=cfg["path"])
    enc.load_state_dict(blob["state"]["enc"], strict=True)
    return enc.to(device).eval().requires_grad_(False), cfg


@torch.inference_mode()
def source_codes(enc, cur, prev, ctx_pos, end, seg, end_pos, end_prev_pos, device):
    """Same inputs/precision as cta_train.Split.context/future under bf16 autocast. Returns (B, K, M) int16."""
    n = len(cur)
    ctx = {"cur": torch.as_tensor(cur).to(device).repeat_interleave(K, 0),
           "prev": torch.as_tensor(prev).to(device).repeat_interleave(K, 0),
           "prop": proprio(torch.as_tensor(ctx_pos[:, 0]), torch.as_tensor(ctx_pos[:, 1])).to(device).repeat_interleave(K, 0)}
    fut = {"end": torch.as_tensor(end).flatten(0, 1).to(device), "seg": torch.as_tensor(seg).flatten(0, 1).to(device),
           "prop": proprio(torch.as_tensor(end_pos), torch.as_tensor(end_prev_pos)).flatten(0, 1).to(device)}
    with torch.autocast(device.type, dtype=torch.bfloat16):
        code = enc(ctx, fut)
    return enc.fsq.codes_to_indices(code).view(n, K, -1).cpu().numpy().astype(np.int16)


def main(a):
    require_compute()
    t0 = time.perf_counter()
    device = torch.device("cuda")
    enc, cfg = load_encoder(a.parent / "cta.pt", device)
    pca = torch.load(a.old_features / "pca.pt", map_location="cpu")
    tok = Tokens(VisualScorer("cuda"))
    tok.mean, tok.basis = pca["pca_mean"].to(device), pca["pca_basis"].to(device)
    old = a.old_features / "train"
    meta = dict(np.load(old / "meta.npz"))
    n_old = len(meta["root"]) if a.old_limit is None else min(a.old_limit, len(meta["root"]))
    shards = sorted(Path(a.collect).glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(a.collect)
    sizes = []
    for f in shards:
        with np.load(f) as z:
            sizes.append(len(z["root"]))
    n_dec = n_old + sum(sizes)
    a.out.mkdir(parents=True, exist_ok=True)
    cur = open_memmap(a.out / "cur.npy", mode="w+", dtype=np.float16, shape=(n_dec, TOKENS, PCA_DIM))
    prev = open_memmap(a.out / "prev.npy", mode="w+", dtype=np.float16, shape=(n_dec, SMALL, PCA_DIM))
    banks = {k: [] for k in ("ctx_index", "root", "decision", "t", "source", "mixed", "chunk", "ctx_pos", "end_pos",
                              "end_prev_pos", "geom", "cov", "src")}

    def add(rows, ctx_index, source, src, mixed):
        banks["ctx_index"].append(ctx_index)
        banks["source"].append(np.full(len(ctx_index), source, np.int8))
        banks["mixed"].append(np.asarray(mixed, bool))
        banks["src"].append(src)
        for k in ("root", "decision", "t", "chunk", "ctx_pos", "end_pos", "end_prev_pos", "geom", "cov"):
            banks[k].append(np.asarray(rows[k]))

    # --- old P0 training data (roots 30250-31049): tokens exist; recompute S with the same frozen encoder
    mm = {k: np.load(old / f"{k}.npy", mmap_mode="r") for k in ("cur", "prev", "end", "seg")}
    for s in range(0, n_old, BATCH):
        e = min(s + BATCH, n_old)
        c, p = np.asarray(mm["cur"][s:e]), np.asarray(mm["prev"][s:e])
        cur[s:e], prev[s:e] = c, p
        src = source_codes(enc, c, p, meta["ctx_pos"][s:e], np.asarray(mm["end"][s:e]), np.asarray(mm["seg"][s:e]),
                           meta["end_pos"][s:e], meta["end_prev_pos"][s:e], device)
        rows = {"root": meta["root"][s:e], "decision": meta["decision"][s:e], "t": meta["t"][s:e],
                "chunk": meta["chunk"][s:e], "ctx_pos": meta["ctx_pos"][s:e], "end_pos": meta["end_pos"][s:e],
                "end_prev_pos": meta["end_prev_pos"][s:e], "geom": meta["cov8"][s:e], "cov": meta["native_cov8"][s:e]}
        add(rows, np.arange(s, e), 0, src, np.zeros(e - s, bool))
    print(f"old: {n_old} decisions", flush=True)
    # --- new shards: two banks per decision
    o = n_old
    for f, size in zip(shards, sizes):
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        c, p = tok(d["ctx"]), tok(d["ctx_prev"], 8)
        cur[o:o + size], prev[o:o + size] = c, p
        geom = registration_score(d["phys8"][..., 4:7])                                   # (D, 2, K)
        for b in range(2):
            src = np.concatenate([source_codes(enc, c[s:s + BATCH], p[s:s + BATCH], d["ctx_pos"][s:s + BATCH],
                                               tok(d["end"][s:s + BATCH, b]), tok(d["seg"][s:s + BATCH, b], 8),
                                               d["end_pos"][s:s + BATCH, b], d["end_prev_pos"][s:s + BATCH, b], device)
                                  for s in range(0, size, BATCH)])
            rows = {"root": d["root"], "decision": d["decision"], "t": d["t"], "chunk": d["chunk"][:, b],
                    "ctx_pos": d["ctx_pos"], "end_pos": d["end_pos"][:, b], "end_prev_pos": d["end_prev_pos"][:, b],
                    "geom": geom[:, b], "cov": d["cov8"][:, b]}
            add(rows, np.arange(o, o + size), 1 + b, src, d["root"] % 2 == 1)
        o += size
        print(f"{f.name}: {size} decisions", flush=True)
    cur.flush()
    prev.flush()
    out = {k: np.concatenate(v) for k, v in banks.items()}
    out["chunk"] = out["chunk"].astype(np.float32)
    for k in ("ctx_pos", "end_pos", "end_prev_pos", "geom", "cov"):
        out[k] = out[k].astype(np.float32)
    np.savez(a.out / "banks.npz", **out)
    for name in ("pca.pt", "goals.npy"):
        (a.out / name).symlink_to((a.old_features / name).resolve())
    report = {"decisions": int(n_dec), "old_decisions": int(n_old), "banks": int(len(out["root"])),
              "banks_by_source": {int(s): int((out["source"] == s).sum()) for s in np.unique(out["source"])},
              "new_shards": [f.name for f in shards], "parent": str(a.parent), "encoder_m": cfg["m"],
              "label_spread": {int(s): float(np.mean(np.ptp(out["geom"][out["source"] == s], axis=1) > 1e-3))
                               for s in np.unique(out["source"])},
              "code_distinct_pairs": {}, "seconds": time.perf_counter() - t0}
    for s in np.unique(out["source"]):
        idx = out["src"][out["source"] == s]
        same = (idx[:, :, None] == idx[:, None, :]).all(-1)
        report["code_distinct_pairs"][int(s)] = float(1 - same[:, ~np.eye(K, dtype=bool)].mean())
    (a.out / "features_plus.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("collect", "old-features", "parent", "out"):
        p.add_argument(f"--{key}", type=Path, required=True)
    p.add_argument("--old-limit", type=int, default=None, help="smoke only")
    main(p.parse_args())
