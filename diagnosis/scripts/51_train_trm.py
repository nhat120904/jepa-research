"""Train a leakage-safe, horizon-matched TRM-style terminal metric.

The fixed encoder cache is split once at trajectory level (70/15/15).  Only the
training trajectories produce gradient updates, validation selects the saved
head, and the reserved test trajectories are read once after selection for an
offline generalization audit.  No simulator-state labels are used.

This follows the primary TRM protocol (arXiv:2605.22164): balanced temporal-gap
sampling, random pair order, a 2x256 SiLU/Softplus pair head, Smooth-L1 labels,
and a 100k-pair default budget.  ``TrajectoryReachabilityMetric`` documents the
single adaptation required by the large JEPA token grids: parameter-free
mean/max token pooling before the pair MLP.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import (LatentCache, build_trajectory_manifest, latent_cache_path,  # noqa: E402
                  load_manifest, write_manifest_once)
from models.heads.trajectory_reachability import TrajectoryReachabilityMetric  # noqa: E402
from trm_protocol import (materialize_temporal_pairs,  # noqa: E402
                          sample_balanced_temporal_pairs,
                          spearman_rank_correlation)


def _trajectory_lengths(cache: LatentCache, tids: list[str]) -> dict[str, int]:
    """Read HDF5 shapes only; do not materialize the latent arrays."""
    assert cache.h5 is not None
    group = cache.h5["trajectories"]
    return {
        tid: int(group[cache._safe_key(tid)]["z"].shape[0])  # type: ignore[attr-defined]
        for tid in tids
    }


def _resolve_manifest(cache_tids: list[str], path: Path, *, dataset: str, model: str,
                      split_seed: int, val_frac: float, test_frac: float) -> dict:
    if path.exists():
        manifest = load_manifest(path)
        if manifest.get("dataset") != dataset or manifest.get("model") != model:
            raise ValueError(
                f"split manifest provenance mismatch: expected {dataset}/{model}, got "
                f"{manifest.get('dataset')}/{manifest.get('model')}"
            )
        present = set().union(*(set(manifest["splits"][s]) for s in ("train", "val", "test")))
        if present != set(cache_tids):
            raise ValueError("split manifest trajectory IDs do not match the current cache")
        return manifest
    manifest = build_trajectory_manifest(
        cache_tids, seed=split_seed, dataset=dataset, model=model,
        val_frac=val_frac, test_frac=test_frac,
    )
    write_manifest_once(path, manifest)
    return manifest


def _run_pair_batches(metric, cache, lengths, *, n_pairs, batch_size, max_gap,
                      label_scale, rng, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    metric.train(training)
    losses, preds, targets = [], [], []
    remaining = int(n_pairs)
    while remaining:
        n = min(int(batch_size), remaining)
        pairs = sample_balanced_temporal_pairs(lengths, n, max_gap, rng)
        z_i, z_j, gap = materialize_temporal_pairs(cache, pairs)
        z_i = z_i.to(device, non_blocking=True).float()
        z_j = z_j.to(device, non_blocking=True).float()
        target = gap.to(device, non_blocking=True) / float(label_scale)
        with torch.set_grad_enabled(training):
            prediction = metric(z_i, z_j)
            loss = torch.nn.functional.smooth_l1_loss(prediction, target)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append((float(loss.detach()), n))
        preds.append(prediction.detach().cpu().numpy())
        targets.append(gap.numpy())
        remaining -= n
        del z_i, z_j, gap, target, prediction
    mean_loss = sum(loss * n for loss, n in losses) / sum(n for _, n in losses)
    spearman = spearman_rank_correlation(np.concatenate(preds), np.concatenate(targets))
    return mean_loss, spearman


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--planning-horizon", type=int, default=6,
                    help="number of model actions in the oracle planner")
    ap.add_argument("--frameskip", type=int, default=5,
                    help="raw cached transitions represented by one model action")
    ap.add_argument("--max-gap", type=int, default=None,
                    help="override horizon-matched gap; default planning_horizon*frameskip")
    ap.add_argument("--label-scale", type=float, default=224.0,
                    help="TRM paper distance-label scale")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--pairs-per-epoch", type=int, default=10000)
    ap.add_argument("--eval-pairs", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0,
                    help="head initialization and pair-sampling seed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.label_scale <= 0:
        raise SystemExit("--label-scale must be positive")
    max_gap = args.max_gap or args.planning_horizon * args.frameskip
    if max_gap <= 0:
        raise SystemExit("effective --max-gap must be positive")
    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "4")))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = yaml.safe_load(open(args.config))
    dataset = str(cfg["dataset"]["name"])
    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, dataset)
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    manifest_path = Path(args.split_manifest)

    with LatentCache(cache_path, mode="r") as cache:
        all_tids = sorted(cache.trajectory_ids())
        manifest = _resolve_manifest(
            all_tids, manifest_path, dataset=dataset, model=args.model,
            split_seed=args.split_seed, val_frac=args.val_frac, test_frac=args.test_frac,
        )
        split_lengths = {
            split: _trajectory_lengths(cache, manifest["splits"][split])
            for split in ("train", "val", "test")
        }
        first_tid = next(iter(split_lengths["train"]))
        first = cache.read_trajectory(first_tid)["z"]
        latent_dim = int(first.shape[-1])
        del first
        print(json.dumps({
            "cache": str(cache_path), "model": args.model, "device": str(device),
            "latent_dim": latent_dim, "max_gap": max_gap,
            "manifest": str(manifest_path), "manifest_sha256": manifest["manifest_sha256"],
            "split_trajectories": {s: len(split_lengths[s]) for s in split_lengths},
        }, sort_keys=True), flush=True)

        metric = TrajectoryReachabilityMetric(latent_dim, hidden=args.hidden).to(device)
        optimizer = torch.optim.AdamW(
            metric.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        train_rng = np.random.default_rng(args.seed)
        best = None
        for epoch in range(1, args.epochs + 1):
            train_loss, train_sp = _run_pair_batches(
                metric, cache, split_lengths["train"], n_pairs=args.pairs_per_epoch,
                batch_size=args.batch_size, max_gap=max_gap, label_scale=args.label_scale,
                rng=train_rng, device=device, optimizer=optimizer,
            )
            # A fixed validation stream makes checkpoint selection comparable
            # across epochs and never touches the test trajectories.
            val_loss, val_sp = _run_pair_batches(
                metric, cache, split_lengths["val"], n_pairs=args.eval_pairs,
                batch_size=args.batch_size, max_gap=max_gap, label_scale=args.label_scale,
                rng=np.random.default_rng(args.split_seed + 1009), device=device,
            )
            print(f"epoch={epoch:02d}/{args.epochs} train_loss={train_loss:.6f} "
                  f"train_spearman={train_sp:.4f} val_loss={val_loss:.6f} "
                  f"val_spearman={val_sp:.4f}", flush=True)
            if best is None or val_loss < best["val_loss"]:
                best = {
                    "epoch": epoch, "val_loss": val_loss, "val_spearman": val_sp,
                    "state_dict": copy.deepcopy(metric.state_dict()),
                }

        assert best is not None
        metric.load_state_dict(best["state_dict"])
        test_loss, test_sp = _run_pair_batches(
            metric, cache, split_lengths["test"], n_pairs=args.eval_pairs,
            batch_size=args.batch_size, max_gap=max_gap, label_scale=args.label_scale,
            rng=np.random.default_rng(args.split_seed + 2017), device=device,
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "trm_style_v1",
        "state_dict": {k: v.detach().cpu() for k, v in best["state_dict"].items()},
        "latent_dim": latent_dim,
        "hidden": args.hidden,
        "model": args.model,
        "dataset": dataset,
        "head_seed": args.seed,
        "planning_horizon": args.planning_horizon,
        "frameskip": args.frameskip,
        "max_gap": max_gap,
        "label_scale": args.label_scale,
        "training_pairs": args.epochs * args.pairs_per_epoch,
        "best_epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "val_spearman": best["val_spearman"],
        "test_loss": test_loss,
        "test_spearman": test_sp,
        "test_used_for_selection": False,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "primary_reference": "https://arxiv.org/abs/2605.22164",
        "adaptation": "parameter-free mean/max token pooling before symmetric pair MLP",
    }
    torch.save(payload, out)
    print(f"selected_epoch={best['epoch']} val_spearman={best['val_spearman']:.4f} "
          f"reserved_test_spearman={test_sp:.4f} test_used_for_selection=false", flush=True)
    print(f"saved {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
