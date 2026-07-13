"""Train the ACID-style inverse-dynamics verifier on cached MetaWorld transitions.

This is a cache/GPU job and must run through Slurm.  A deterministic immutable
70/15/15 trajectory manifest prevents inverse-verifier train/eval leakage.  The
test split is touched only once after validation-selected training has finished.

Important fidelity boundary: ACID (arXiv:2607.02403) uses a 4-layer, width-192
flow-matching prefix/suffix transformer.  No official code or verifier checkpoint
was publicly linked as of 2026-07-13.  We therefore train a deterministic pooled
MLP approximation with the same input/output contract G(z_t,z_t1)->a_t.  The
evaluation runner implements ACID's planning residual and adaptive weight exactly.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import (  # noqa: E402
    LatentCache,
    build_trajectory_manifest,
    filter_records,
    latent_cache_path,
    load_manifest,
    read_regimes,
    write_manifest_once,
)
from models.adapters import build_adapter  # noqa: E402
from models.heads.acid_idm import ACIDInverseDynamics, transition_features  # noqa: E402


def _load_runner_helpers():
    spec = importlib.util.spec_from_file_location(
        "_acid_cache_helpers", ROOT / "scripts" / "05_run_diagnostic.py"
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _subsample(records: list[dict], n: int, seed: int) -> list[dict]:
    if n <= 0 or len(records) <= n:
        return list(records)
    idx = np.random.default_rng(seed).choice(len(records), size=n, replace=False)
    return [records[int(i)] for i in sorted(idx)]


@torch.no_grad()
def materialize_features(cache, records, step, latent_dim, helpers, chunk):
    """Pool immediately so the multi-GB token grids never accumulate in RAM."""
    feats, actions = [], []
    for lo in range(0, len(records), chunk):
        data = helpers.materialize_records(
            cache, records[lo : lo + chunk], step,
            want_proprio=False, want_state=False,
        )
        feats.append(transition_features(data["z_t"], data["z_t1"], latent_dim).cpu())
        actions.append(data["a_t"].float().cpu())
        del data
    if not feats:
        raise ValueError("split has no transitions")
    return torch.cat(feats), torch.cat(actions)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--train-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--split-manifest", default=None)
    ap.add_argument("--max-train", type=int, default=30000)
    ap.add_argument("--max-val", type=int, default=5000)
    ap.add_argument("--max-test", type=int, default=5000)
    ap.add_argument("--materialize-chunk", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "8")))
    torch.manual_seed(args.train_seed)
    np.random.seed(args.train_seed)
    cfg = yaml.safe_load(open(args.config))
    dataset = cfg["dataset"]["name"]
    if dataset != "metaworld":
        raise SystemExit("this controlled baseline currently targets MetaWorld only")
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    adapter = build_adapter(args.model, device=str(device)).eval()
    step = int(adapter.frames_per_step)
    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, dataset)
    helpers = _load_runner_helpers()

    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(
            cache, read_regimes(cache_path), step, per_task=True
        )
        if not records:
            raise SystemExit(f"no transitions in {cache_path}")
        manifest_path = Path(args.split_manifest) if args.split_manifest else (
            ROOT / "checkpoints" / "splits" /
            f"acid_{args.model}_split{args.split_seed}.json"
        )
        if manifest_path.exists():
            manifest = load_manifest(manifest_path)
        else:
            manifest = build_trajectory_manifest(
                (r["tid"] for r in records), seed=args.split_seed,
                dataset=dataset, model=args.model,
                val_frac=args.val_frac, test_frac=args.test_frac,
            )
            write_manifest_once(manifest_path, manifest)
        if manifest["model"] != args.model or manifest["dataset"] != dataset:
            raise ValueError("split manifest model/dataset does not match this run")

        split_records = {
            split: filter_records(records, manifest, split)
            for split in ("train", "val", "test")
        }
        selected = {
            "train": _subsample(split_records["train"], args.max_train, args.train_seed + 11),
            "val": _subsample(split_records["val"], args.max_val, args.train_seed + 12),
            "test": _subsample(split_records["test"], args.max_test, args.train_seed + 13),
        }
        first = cache.read_trajectory(records[0]["tid"])["z"]
        latent_dim = int(first.shape[-1])
        print(
            f"cache={cache_path} step={step} latent_dim={latent_dim} "
            f"manifest={manifest_path} hash={manifest['manifest_sha256']}", flush=True,
        )
        print(
            "selected transitions: " + " ".join(f"{k}={len(v)}" for k, v in selected.items()),
            flush=True,
        )
        x_train, a_train = materialize_features(
            cache, selected["train"], step, latent_dim, helpers, args.materialize_chunk
        )
        x_val, a_val = materialize_features(
            cache, selected["val"], step, latent_dim, helpers, args.materialize_chunk
        )
        # Test remains unmaterialized until model selection is over.

    action_dim = int(a_train.shape[-1])
    idm = ACIDInverseDynamics(latent_dim, action_dim, hidden=args.hidden).to(device)
    optimizer = torch.optim.AdamW(
        idm.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mean_action = a_train.mean(dim=0)

    @torch.no_grad()
    def metrics(x, a):
        idm.eval()
        sq_sum = l2_sum = 0.0
        n = 0
        for lo in range(0, len(x), args.batch_size):
            xb = x[lo : lo + args.batch_size].to(device)
            ab = a[lo : lo + args.batch_size].to(device)
            residual = idm.forward_features(xb) - ab
            sq_sum += float((residual ** 2).sum())
            l2_sum += float((residual ** 2).sum(dim=-1).sum())
            n += len(xb)
        return {"mse": sq_sum / max(n * action_dim, 1), "mean_l2_sq": l2_sum / max(n, 1)}

    best_val = float("inf")
    rng = np.random.default_rng(args.train_seed)
    for epoch in range(args.epochs):
        idm.train()
        order = rng.permutation(len(x_train))
        total = 0.0
        for lo in range(0, len(order), args.batch_size):
            idx = torch.as_tensor(order[lo : lo + args.batch_size], dtype=torch.long)
            xb = x_train[idx].to(device)
            ab = a_train[idx].to(device)
            loss = torch.nn.functional.mse_loss(idm.forward_features(xb), ab)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(idx)
        val = metrics(x_val, a_val)
        print(
            f"epoch {epoch + 1:02d}/{args.epochs}: "
            f"train_mse={total / len(x_train):.6f} val_mse={val['mse']:.6f}",
            flush=True,
        )
        if val["mse"] < best_val:
            best_val = val["mse"]
            torch.save(
                {
                    "model": args.model,
                    "dataset": dataset,
                    "latent_dim": latent_dim,
                    "action_dim": action_dim,
                    "frames_per_step": step,
                    "hidden": args.hidden,
                    "state_dict": idm.state_dict(),
                    "architecture": "deterministic_meanmax_mlp_approximation",
                    "acid_reference": "arXiv:2607.02403v1",
                    "official_idm_difference": (
                        "Paper uses a 4-layer width-192 3-head flow-matching "
                        "prefix/suffix transformer; no official code/checkpoint was "
                        "available on 2026-07-13."
                    ),
                    "split_manifest": str(manifest_path),
                    "split_manifest_sha256": manifest["manifest_sha256"],
                    "split_seed": args.split_seed,
                    "train_seed": args.train_seed,
                    "sample_counts": {k: len(v) for k, v in selected.items()},
                    "val_action_mse": val["mse"],
                    "val_action_l2_sq": val["mean_l2_sq"],
                    "mean_action": mean_action,
                },
                out,
            )

    # Load the validation-selected checkpoint before the one permitted test pass.
    best = torch.load(out, map_location=device, weights_only=False)
    idm.load_state_dict(best["state_dict"])
    with LatentCache(cache_path, mode="r") as cache:
        x_test, a_test = materialize_features(
            cache, selected["test"], step, latent_dim, helpers, args.materialize_chunk
        )
    test = metrics(x_test, a_test)
    baseline_mse = float(((a_test - mean_action) ** 2).mean())
    best["test_action_mse"] = test["mse"]
    best["test_action_l2_sq"] = test["mean_l2_sq"]
    best["test_constant_mean_mse"] = baseline_mse
    torch.save(best, out)
    print(
        f"wrote {out}\nval_mse={best_val:.6f} test_mse={test['mse']:.6f} "
        f"constant_mean_test_mse={baseline_mse:.6f}", flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
