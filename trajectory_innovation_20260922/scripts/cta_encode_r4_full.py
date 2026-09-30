"""Full-token features of the Round-4 PushT banks (collection 55147), for retraining the source encoder and readers.

scripts/cta_encode_plus.py kept only the frozen encoder's code indices of these banks, so the codec and readers were
never trained on them. This writes them in the format of scripts/cta_encode.py:encode_split (the format of the Round-0
and on-policy feature caches), with the ORIGINAL PCA basis (cta_feat_54490), one split per bank type:
  r4std   the policy's 8 seeded samples (the deployment bank)
  r4pert  their 8 perturbed copies
meta.npz: root, decision, t, ctx_pos, chunk, end_pos, end_prev_pos, cov8 = registration (geometry) label from the
simulated end poses (as cta_encode_plus / cta_geometry), native_cov8 = native coverage, done8. Compute node only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cta_encode import Tokens  # noqa: E402
from ti_wm.contract import require_compute  # noqa: E402
from ti_wm.cta_geometry import registration_score  # noqa: E402
from ti_wm.pusht_runtime import VisualScorer  # noqa: E402
from ti_wm.sibling import PCA_DIM, TOKENS  # noqa: E402

K, SMALL, N_SEG, BATCH = 8, 64, 3, 64
SPLITS = {"r4std": 0, "r4pert": 1}


def main(a):
    require_compute()
    t0 = time.perf_counter()
    pca = torch.load(a.pca, map_location="cpu")
    tok = Tokens(VisualScorer("cuda"))
    tok.mean, tok.basis = pca["pca_mean"].cuda(), pca["pca_basis"].cuda()
    shards = sorted(Path(a.collect).glob("shard_*.npz"))[: a.max_shards or None]
    sizes = []
    for f in shards:
        with np.load(f) as z:
            sizes.append(len(z["root"]))
    n = sum(sizes)
    out = {}
    for name in SPLITS:
        d = a.out / name
        d.mkdir(parents=True, exist_ok=True)
        out[name] = {"cur": open_memmap(d / "cur.npy", "w+", np.float16, (n, TOKENS, PCA_DIM)),
                     "prev": open_memmap(d / "prev.npy", "w+", np.float16, (n, SMALL, PCA_DIM)),
                     "end": open_memmap(d / "end.npy", "w+", np.float16, (n, K, TOKENS, PCA_DIM)),
                     "seg": open_memmap(d / "seg.npy", "w+", np.float16, (n, K, N_SEG, SMALL, PCA_DIM))}
    meta = {name: {k: [] for k in ("root", "decision", "t", "ctx_pos", "chunk", "end_pos", "end_prev_pos", "cov8",
                                   "native_cov8", "done8")} for name in SPLITS}
    o = 0
    for f, size in zip(shards, sizes):
        with np.load(f) as z:
            d = {k: z[k] for k in z.files}
        geom = registration_score(d["phys8"][..., 4:7])                                     # (D, 2, K)
        for s in range(0, size, BATCH):
            e = min(s + BATCH, size)
            c, p = tok(d["ctx"][s:e]), tok(d["ctx_prev"][s:e], 8)
            for name, b in SPLITS.items():
                out[name]["cur"][o + s:o + e] = c
                out[name]["prev"][o + s:o + e] = p
                out[name]["end"][o + s:o + e] = tok(d["end"][s:e, b])
                out[name]["seg"][o + s:o + e] = tok(d["seg"][s:e, b], 8)
        for name, b in SPLITS.items():
            for k, v in (("root", d["root"]), ("decision", d["decision"]), ("t", d["t"]), ("ctx_pos", d["ctx_pos"]),
                         ("chunk", d["chunk"][:, b]), ("end_pos", d["end_pos"][:, b]),
                         ("end_prev_pos", d["end_prev_pos"][:, b]), ("cov8", geom[:, b]), ("native_cov8", d["cov8"][:, b]),
                         ("done8", d["done8"][:, b])):
                meta[name][k].append(np.asarray(v))
        o += size
        print(f"{f.name}: {size} decisions ({o}/{n}) {time.perf_counter() - t0:.0f}s", flush=True)
    report = {"collect": str(a.collect), "pca": str(a.pca), "decisions": n, "shards": [f.name for f in shards]}
    for name in SPLITS:
        for v in out[name].values():
            v.flush()
        m = {k: np.concatenate(v) for k, v in meta[name].items()}
        for k in ("ctx_pos", "end_pos", "end_prev_pos", "cov8", "native_cov8"):
            m[k] = m[k].astype(np.float32)
        m["chunk"] = m["chunk"].astype(np.float32)
        np.savez(a.out / name / "meta.npz", **m)
        report[name] = {"informative_banks": int((np.ptp(m["cov8"], axis=1) > 1e-3).sum())}
    report["seconds"] = time.perf_counter() - t0
    (a.out / "encode_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--collect", type=Path, required=True)
    p.add_argument("--pca", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-shards", type=int, default=0, help="smoke only")
    main(p.parse_args())
