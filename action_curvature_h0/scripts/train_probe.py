#!/usr/bin/env python3
"""Gate A: fit and validate the linear latent -> physical-state bridge.

The bridge exists so that the model's predicted outcome and the simulator's
true outcome can be compared in the same units (metres) instead of in an
arbitrary latent chart.  It is deliberately **affine**:

    P(z) = ((z - mu) / sigma) W + b

so that second differences pass through exactly,

    D2 (P o Phi) = W' . D2 Phi ,

and the probe cannot manufacture curvature of its own.  A nonlinear probe would
destroy that property and with it the whole argument, so if this gate fails the
answer is a different reference, never an MLP.

Calibration states are drawn from episodes NOT in the Stage-1 manifest, and the
held-out split is by episode.  Frames come from the same renderer path the
diagnostic uses (restore -> render -> encode), not from the dataset's stored
pixels, so the probe is never asked to cross a renderer domain gap at eval time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "action_curvature_h0" / "scripts"))
from measure_curvature import (  # noqa: E402
    build_contact_classifier,
    effector_position,
    encode_images_dtype,
    load_script_module,
    object_position,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path,
                   default=ROOT / "physical_search_distillation/outputs/h0/manifest.json")
    p.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    p.add_argument("--checkpoint", default="quentinll/lewm-cube")
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--n-episodes", type=int, default=120)
    p.add_argument("--steps-per-episode", type=int, default=4)
    p.add_argument("--heldout-fraction", type=float, default=0.25)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--encode-batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=20260825)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("probe calibration requires the GPU nodes")

    import stable_worldmodel as swm
    from stable_worldmodel.world.world import _extract_init_goal

    audit = load_script_module("72_ogb_stage0_candidate_audit.py", "probe_audit")
    corrected = load_script_module("76_ogb_true_endpoint_corrected.py", "probe_corrected")
    corrected.load_stage0_transform_images = audit.transform_images

    manifest = json.loads(args.manifest.read_text())
    excluded = {int(r["episode"]) for r in manifest}

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    rng = np.random.default_rng(args.seed)

    # Episodes are array indices into dataset.lengths / dataset.offsets; there
    # is no episode column.  Mirrors build_manifest in script 72.
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    candidates = np.array(
        [i for i in range(lengths.size)
         if i not in excluded and int(lengths[i]) - args.goal_offset - 1 > 1]
    )
    if candidates.size < args.n_episodes:
        raise RuntimeError(
            f"only {candidates.size} usable episodes outside the manifest"
        )
    chosen = np.sort(rng.choice(candidates, size=args.n_episodes, replace=False))

    episodes: list[int] = []
    steps: list[int] = []
    for ep in chosen:
        usable = int(lengths[ep]) - args.goal_offset - 1
        take = min(args.steps_per_episode, usable)
        for s in rng.choice(usable, size=take, replace=False):
            episodes.append(int(ep))
            steps.append(int(s))

    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, episodes, steps, args.goal_offset
    )

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = audit.make_transform(224)

    snapshot = audit.Snapshot(**manifest[0])
    world, raw_env, visual_hash, _ = corrected.make_world(swm, snapshot)
    frames: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    try:
        build_contact_classifier(raw_env)  # same structural guard as the diagnostic
        for init_row, goal_row in zip(init_rows, goal_rows):
            corrected.restore_complete(
                raw_env, init_row["qpos"], init_row["qvel"], goal_row, audit
            )
            frames.append(audit.resize_render(raw_env.render()))
            labels.append(
                np.concatenate([object_position(raw_env), effector_position(raw_env)])
            )
        latents = encode_images_dtype(
            model, np.stack(frames), transform, args.encode_batch, audit
        )
    finally:
        world.close()

    X = np.asarray(latents, dtype=np.float64)
    Y = np.asarray(labels, dtype=np.float64)
    groups = np.asarray(episodes[: len(Y)], dtype=np.int64)

    uniq = np.unique(groups)
    n_heldout = max(1, int(round(args.heldout_fraction * uniq.size)))
    heldout_eps = set(rng.choice(uniq, size=n_heldout, replace=False).tolist())
    is_heldout = np.array([g in heldout_eps for g in groups])

    Xtr, Ytr = X[~is_heldout], Y[~is_heldout]
    Xte, Yte = X[is_heldout], Y[is_heldout]

    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma[sigma < 1e-12] = 1.0
    Ztr = (Xtr - mu) / sigma
    Zte = (Xte - mu) / sigma

    d = Ztr.shape[1]
    A = Ztr.T @ Ztr + args.ridge * Ztr.shape[0] * np.eye(d)
    W = np.linalg.solve(A, Ztr.T @ (Ytr - Ytr.mean(axis=0)))
    b = Ytr.mean(axis=0)

    def predict(Z: np.ndarray) -> np.ndarray:
        return Z @ W + b

    names = ["object_x", "object_y", "object_z", "effector_x", "effector_y", "effector_z"]
    metrics: dict[str, Any] = {"n_train": int(Xtr.shape[0]), "n_heldout": int(Xte.shape[0]),
                               "n_train_episodes": int(uniq.size - n_heldout),
                               "n_heldout_episodes": int(n_heldout),
                               "latent_dim": int(d), "ridge": args.ridge}
    for split, Z, Yt in (("train", Ztr, Ytr), ("heldout", Zte, Yte)):
        pred = predict(Z)
        resid = pred - Yt
        ss_res = np.sum(resid**2, axis=0)
        ss_tot = np.sum((Yt - Yt.mean(axis=0)) ** 2, axis=0)
        metrics[f"{split}_r2_per_axis"] = {
            n: float(1.0 - ss_res[i] / max(ss_tot[i], 1e-30)) for i, n in enumerate(names)
        }
        metrics[f"{split}_median_err_mm"] = {
            "object": float(np.median(np.linalg.norm(resid[:, :3], axis=1)) * 1000.0),
            "effector": float(np.median(np.linalg.norm(resid[:, 3:], axis=1)) * 1000.0),
        }
        metrics[f"{split}_p90_err_mm"] = {
            "object": float(np.percentile(np.linalg.norm(resid[:, :3], axis=1), 90) * 1000.0),
            "effector": float(np.percentile(np.linalg.norm(resid[:, 3:], axis=1), 90) * 1000.0),
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "probe.npz", W=W, b=b, mu=mu, sigma=sigma,
                        target_names=np.array(names))
    metrics["visual_signature"] = visual_hash
    metrics["excluded_manifest_episodes"] = len(excluded)
    (args.out_dir / "gate_a.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
