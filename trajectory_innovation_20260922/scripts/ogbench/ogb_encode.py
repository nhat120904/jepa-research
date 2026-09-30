"""Frozen DINOv2 ViT-S/14 PCA-128 tokens of the branched OGBench data (docs/CTA_OGBENCH_PROTOCOL.md).

Same tokenizer as PushT (ti_wm.pusht_runtime.VisualScorer: frames upsampled to 224, patch tokens; PCA fitted on
TRAIN decision frames only). Per split (train / heldout = last 10% of each task's train episodes, used only for
checkpoint selection):
  cur (N, 256, 128)  prev (N, 64, 128) pooled 8x8  fut (N, K, 4, 256, 128) future frames after steps 2-5 at full grid
  goal (E, 256, 128) one per episode, goal_index (N,)
  meta.npz: task, ep, root, d, t, chunk (N, K, 5, 5), prog (N, K, 5), success (N, K), executed (N,), cubes
fp16 memmaps. Compute node only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ti_wm.codec import pool_grid  # noqa: E402
from ti_wm.contract import require_compute  # noqa: E402
from ti_wm.pusht_runtime import VisualScorer  # noqa: E402
from ti_wm.sibling import PCA_DIM, TOKENS, fit_pca, project  # noqa: E402

K, N_FUT, BATCH, PCA_FRAMES = 8, 4, 512, 2000     # N_FUT: stored future frames = steps 2, 3, 4, 5


class Tok:
    def __init__(self):
        self.visual = VisualScorer("cuda")
        self.mean = self.basis = None

    @torch.inference_mode()
    def fit(self, frames):
        raw = torch.cat([self.visual.features(frames[s:s + BATCH]) for s in range(0, len(frames), BATCH)])
        self.mean, self.basis = fit_pca(raw.reshape(-1, raw.shape[-1]))

    @torch.inference_mode()
    def __call__(self, frames, side=None):
        lead = frames.shape[:-3]
        flat = frames.reshape(-1, *frames.shape[-3:])
        out = []
        for s in range(0, len(flat), BATCH):
            x = project(self.visual.features(flat[s:s + BATCH]), self.mean, self.basis)
            out.append((pool_grid(x, side) if side else x).half().cpu())
        out = torch.cat(out).numpy()
        return out.reshape(*lead, *out.shape[1:])


def split_files(collect, heldout_frac=0.1):
    files = sorted(Path(collect).glob("task*_ep*.npz"))
    by_task = {}
    for f in files:
        task, ep = f.stem.split("_")
        by_task.setdefault(int(task[4:]), []).append((int(ep[2:]), f))
    train, held = [], []
    for task, items in sorted(by_task.items()):
        items.sort()
        n_held = max(1, int(round(heldout_frac * len(items))))
        train += [f for _, f in items[:-n_held]]
        held += [f for _, f in items[-n_held:]]
    return train, held


def encode(tok, files, out):
    out.mkdir(parents=True, exist_ok=True)
    sizes = []
    for f in files:
        with np.load(f) as z:
            sizes.append(len(z["t"]))
    n, e = sum(sizes), len(files)
    mm = {"cur": np.lib.format.open_memmap(out / "cur.npy", "w+", np.float16, (n, TOKENS, PCA_DIM)),
          "prev": np.lib.format.open_memmap(out / "prev.npy", "w+", np.float16, (n, 64, PCA_DIM)),
          "fut": np.lib.format.open_memmap(out / "fut.npy", "w+", np.float16, (n, K, N_FUT, TOKENS, PCA_DIM)),
          "goal": np.lib.format.open_memmap(out / "goal.npy", "w+", np.float16, (e, TOKENS, PCA_DIM))}
    meta = {k: [] for k in ("task", "ep", "root", "d", "t", "chunk", "prog", "success", "executed", "cubes",
                            "goal_index")}
    at = 0
    for gi, (f, m) in enumerate(zip(files, sizes)):
        with np.load(f) as z:
            mm["cur"][at:at + m] = tok(z["cur"])
            mm["prev"][at:at + m] = tok(z["prev"], 8)
            mm["fut"][at:at + m] = tok(z["fut"])
            mm["goal"][gi] = tok(z["goal"][None])[0]
            for k in ("chunks", "prog", "success", "executed", "cubes", "t"):
                meta["chunk" if k == "chunks" else k].append(z[k])
            meta["task"].append(np.full(m, int(z["task"])))
            meta["ep"].append(np.full(m, int(z["ep"])))
            meta["root"].append(np.full(m, int(z["root"])))
            meta["d"].append(np.arange(m))
            meta["goal_index"].append(np.full(m, gi))
        at += m
        print(f"{f.name}: {m} decisions ({at}/{n})", flush=True)
    for v in mm.values():
        v.flush()
    np.savez(out / "meta.npz", **{k: np.concatenate(v) for k, v in meta.items()})
    return {"decisions": n, "episodes": e}


def main(a):
    require_compute()
    train, held = split_files(a.collect)
    tok = Tok()
    rng = np.random.default_rng(0)
    frames = []
    for f in rng.permutation(train)[:200]:
        with np.load(f) as z:
            frames.append(z["cur"])
    frames = np.concatenate(frames)
    tok.fit(frames[rng.choice(len(frames), min(PCA_FRAMES, len(frames)), replace=False)])
    a.out.mkdir(parents=True, exist_ok=True)
    torch.save({"pca_mean": tok.mean.cpu(), "pca_basis": tok.basis.cpu()}, a.out / "pca.pt")
    report = {"collect": str(a.collect), "train_files": len(train), "heldout_files": len(held),
              "train": encode(tok, train, a.out / "train"), "heldout": encode(tok, held, a.out / "heldout")}
    (a.out / "encode_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--collect", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
