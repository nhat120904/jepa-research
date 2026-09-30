"""Frozen DINOv2 PCA-128 tokens of the official OGBench play dataset (visual-cube-single-play-v0, 1M transitions).

Uses the PCA basis of the branched data (enc/<env>/pca.pt) so play-trained modules read the branched held-out banks
with the same features. Output (fp16 memmaps):
  full.npy   (N, 256, 128)  every frame on the 16x16 grid
  small.npy  (N, 64, 128)   every frame pooled to 8x8
  meta.npz   actions (N, 5), cube (N, 3) cube position from qpos, ep (N,) episode id, t (N,) step in episode
Cube position: qpos[:, CUBE] (object free joint xyz), checked against the resting height in the report.
Compute node only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from ti_wm.codec import pool_grid  # noqa: E402
from ti_wm.contract import require_compute  # noqa: E402
from ti_wm.pusht_runtime import VisualScorer  # noqa: E402
from ti_wm.sibling import PCA_DIM, TOKENS, project  # noqa: E402

CUBE = slice(14, 17)      # ur5e arm (6) + robotiq gripper (8) joints, then the cube's free joint (xyz, quat)
BATCH = 1024


def main(a):
    require_compute()
    t0 = time.perf_counter()
    z = np.load(a.dataset)
    obs, actions, terminals, qpos = z["observations"], z["actions"], z["terminals"], z["qpos"]
    n = len(obs) if not a.limit else a.limit
    term = terminals[:n].astype(np.int32)
    ep = np.concatenate([[0], np.cumsum(term[:-1])]).astype(np.int32)      # a new episode starts after each terminal
    starts = np.concatenate([[0], np.flatnonzero(term[:-1]) + 1])
    t = np.arange(n) - starts[ep]
    # the cube's qpos address, read from the environment's MuJoCo model
    from ti_wm import ogb_runtime as ogb
    env = ogb.make_env(a.env)
    adr = int(env.unwrapped._model.joint("object_joint_0").qposadr[0])
    if adr != CUBE.start:
        raise ValueError(f"cube qpos address {adr} != {CUBE.start}")
    cube = qpos[:n, CUBE].astype(np.float32)
    report = {"dataset": str(a.dataset), "frames": int(n), "episodes": int(ep.max() + 1),
              "cube_z_percentiles": np.percentile(cube[:, 2], [1, 10, 50, 90, 99]).round(4).tolist(),
              "cube_xy_min": cube[:, :2].min(0).round(3).tolist(), "cube_xy_max": cube[:, :2].max(0).round(3).tolist()}
    print(json.dumps(report), flush=True)
    pca = torch.load(a.pca, map_location="cuda")
    mean, basis = pca["pca_mean"].cuda(), pca["pca_basis"].cuda()
    visual = VisualScorer("cuda")
    a.out.mkdir(parents=True, exist_ok=True)
    full = np.lib.format.open_memmap(a.out / "full.npy", "w+", np.float16, (n, TOKENS, PCA_DIM))
    small = np.lib.format.open_memmap(a.out / "small.npy", "w+", np.float16, (n, 64, PCA_DIM))
    with torch.inference_mode():
        for s in range(0, n, BATCH):
            e = min(s + BATCH, n)
            x = project(visual.features(np.asarray(obs[s:e])), mean, basis)
            full[s:e] = x.half().cpu().numpy()
            small[s:e] = pool_grid(x, 8).half().cpu().numpy()
            if (s // BATCH) % 50 == 0:
                print(f"{e}/{n} frames {time.perf_counter() - t0:.0f}s", flush=True)
    full.flush()
    small.flush()
    np.savez(a.out / "meta.npz", actions=actions[:n].astype(np.float32), cube=cube, ep=ep, t=t.astype(np.int32),
             terminals=terminals[:n])
    report["seconds"] = time.perf_counter() - t0
    (a.out / "encode_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, default=Path("/mnt/data/nhatnc129/jepa/ogbench/data/visual-cube-single-play-v0.npz"))
    p.add_argument("--pca", type=Path, default=Path("/mnt/data/nhatnc129/jepa/ogbench/enc/visual-cube-single-play-v0/pca.pt"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--env", default="visual-cube-single-play-v0")
    p.add_argument("--limit", type=int, default=0, help="smoke only")
    main(p.parse_args())
