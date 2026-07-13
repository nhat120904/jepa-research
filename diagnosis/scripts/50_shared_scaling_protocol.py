"""Apples-to-apples 22M->1B DROID action-grounding rerun.

The original diagnostic used each model's latent to choose both its effect
subset and hard negatives.  That makes a cross-model scaling curve ambiguous.
This runner first creates one immutable, sensor-only manifest from the common
DROID transitions, then evaluates every checkpoint on exactly the same:

* trajectory/time anchors;
* physical-effect mask (raw proprio/gripper change; no latent);
* K negative trajectory/time indices, chosen by physical-state proximity.

The negatives remain observational, not causal.  Exact same-state causality is
handled separately by ``49_same_state_intervention.py`` on MetaWorld.
Run via ``slurm_shared_scaling_protocol.sh`` only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path  # noqa: E402
from metrics.bootstrap import bootstrap_ci  # noqa: E402
from metrics.cra import cra_per_transition  # noqa: E402
from models.adapters import build_adapter  # noqa: E402
from scripts._shared_scaling_protocol import (  # noqa: E402
    fixed_physical_neighbours,
    physical_effect_scores,
    physical_regimes,
    physical_state_features,
)


def require_slurm_gpu() -> None:
    if not os.environ.get("SLURM_JOB_ID") and os.environ.get("CAI_JEPA_ALLOW_LOGIN_HEAVY") != "1":
        raise RuntimeError(
            "Refusing the shared-scaling checkpoint run outside Slurm. Submit "
            "scripts/slurm_shared_scaling_protocol.sh."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("shared scaling evaluation requires a CUDA compute node")


def _group(cache: LatentCache, tid: str):
    assert cache.h5 is not None
    return cache.h5["trajectories"][LatentCache._safe_key(tid)]


def build_common_manifest(cache_paths: dict[str, Path], *, reference_model: str,
                          K: int, pool_size: int, seed: int, effect_quantile: float):
    """Read raw arrays only and return a fixed common transition manifest."""
    caches = {name: LatentCache(path, "r").__enter__() for name, path in cache_paths.items()}
    try:
        common = set(caches[reference_model].trajectory_ids())
        for cache in caches.values():
            common &= set(cache.trajectory_ids())
        tids = sorted(common)
        if not tids:
            raise RuntimeError("no trajectory IDs are common to all requested caches")

        refs, p0s, p1s, actions = [], [], [], []
        raw_hash = hashlib.sha256()
        for tid in tids:
            ref_grp = _group(caches[reference_model], tid)
            ref_a = np.asarray(ref_grp["action"], dtype=np.float32)
            ref_p = np.asarray(ref_grp["proprio"], dtype=np.float32)
            n = min(len(ref_a), len(ref_p) - 1)
            for name, cache in caches.items():
                grp = _group(cache, tid)
                a = np.asarray(grp["action"][:n], dtype=np.float32)
                p = np.asarray(grp["proprio"][:n + 1], dtype=np.float32)
                if a.shape != ref_a[:n].shape or p.shape != ref_p[:n + 1].shape:
                    raise RuntimeError(f"raw alignment shape mismatch: model={name} tid={tid}")
                if not np.allclose(a, ref_a[:n], atol=1e-6, rtol=0) or not np.allclose(
                        p, ref_p[:n + 1], atol=1e-6, rtol=0):
                    raise RuntimeError(f"raw action/proprio mismatch: model={name} tid={tid}")
            raw_hash.update(tid.encode())
            raw_hash.update(ref_a[:n].tobytes()); raw_hash.update(ref_p[:n + 1].tobytes())
            for t in range(n):
                refs.append((tid, t))
            p0s.append(ref_p[:n]); p1s.append(ref_p[1:n + 1]); actions.append(ref_a[:n])

        p0, p1, action = np.concatenate(p0s), np.concatenate(p1s), np.concatenate(actions)
        effect, effect_scales = physical_effect_scores(p0, p1)
        effect_threshold = float(np.quantile(effect, effect_quantile))
        regimes = physical_regimes(
            p0, p1, gripper_dim=-1, motion_scores=effect,
            motion_threshold=effect_threshold)

        # Shared current-state feature space.  One robust centering/scaling is
        # computed once on raw proprio and reused by every model.
        features, state_median, state_scales = physical_state_features(p0)
        rng = np.random.default_rng(seed)
        pool_global = np.sort(rng.choice(
            len(refs), size=min(pool_size, len(refs)), replace=False))
        if len(pool_global) <= K:
            raise RuntimeError(f"pool_size={len(pool_global)} must exceed K={K}")
        neg_pool_rows = fixed_physical_neighbours(
            features, features[pool_global], K=K,
            anchor_global_indices=np.arange(len(refs)), pool_global_indices=pool_global)
        neg_global = pool_global[neg_pool_rows]

        anchor_rows = [{
            "global_index": i, "transition_id": f"{tid}__t{t:03d}",
            "traj_id": tid, "t": t, "regime": str(regimes[i]),
            "physical_effect_score": float(effect[i]),
            "shared_effect_mask": int(effect[i] > effect_threshold),
        } for i, (tid, t) in enumerate(refs)]
        negative_rows = []
        for i, rows in enumerate(neg_global):
            for rank, j in enumerate(rows):
                ntid, nt = refs[int(j)]
                negative_rows.append({
                    "global_index": i, "negative_rank": rank,
                    "negative_global_index": int(j),
                    "negative_transition_id": f"{ntid}__t{nt:03d}",
                    "negative_traj_id": ntid, "negative_t": nt,
                })
        protocol = {
            "name": "shared_physical_effect_fixed_negative_v1",
            "reference_model_raw_arrays": reference_model,
            "models": list(cache_paths), "n_common_trajectories": len(tids),
            "n_common_transitions": len(refs), "K": K, "pool_size": len(pool_global),
            "pool_seed": seed, "effect_quantile": effect_quantile,
            "effect_threshold": effect_threshold,
            "effect_scales": effect_scales.tolist(),
            "state_median": state_median.tolist(), "state_scales": state_scales.tolist(),
            "negative_rule": (
                "K nearest states by robust-standardized raw proprio with sin/cos Euler embedding; "
                "no latent/action reranking"),
            "effect_rule": (
                "L2 norm of robust-standardized raw proprio delta with wrapped Euler deltas, "
                "shared quantile threshold"),
            "regime_rule": "raw gripper sensor first; open-gripper pre_grasp_proxy uses shared physical-motion score",
            "raw_common_manifest_sha256": raw_hash.hexdigest(),
            "causal_scope": "observational matched-state diagnostic; not exact same-state intervention",
        }
        return anchor_rows, negative_rows, protocol, refs, action, neg_global
    finally:
        for cache in caches.values():
            cache.__exit__(None, None, None)


def evaluate_model(model_name: str, cache_path: Path, refs, action: np.ndarray,
                   neg_global: np.ndarray, anchor_rows: list[dict], *,
                   device: torch.device, batch_size: int):
    adapter = build_adapter(model_name, device=str(device)).eval()
    if adapter.frames_per_step != 1:
        raise RuntimeError(
            f"shared DROID manifest is one raw transition per step, but {model_name} "
            f"reports frames_per_step={adapter.frames_per_step}")
    rows = []
    with LatentCache(cache_path, "r") as cache:
        for start in range(0, len(refs), batch_size):
            stop = min(start + batch_size, len(refs))
            batch_refs = refs[start:stop]
            z0, z1, prop = [], [], []
            for tid, t in batch_refs:
                grp = _group(cache, tid)
                z0.append(np.asarray(grp["z"][t], dtype=np.float32))
                z1.append(np.asarray(grp["z"][t + 1], dtype=np.float32))
                prop.append(np.asarray(grp["proprio"][t], dtype=np.float32))
            z0t = torch.from_numpy(np.stack(z0)).to(device)
            z1t = torch.from_numpy(np.stack(z1)).to(device)
            at = torch.from_numpy(action[start:stop]).to(device)
            aneg = torch.from_numpy(action[neg_global[start:stop]]).to(device)
            pt = torch.from_numpy(np.stack(prop)).to(device) if adapter.uses_proprio() else None
            correct, mrr, df, dn = cra_per_transition(
                adapter, z0t, at, z1t, aneg, distance="l2", proprio_t=pt)
            for local, i in enumerate(range(start, stop)):
                base = anchor_rows[i]
                rows.append({
                    "model": model_name, **base,
                    "cra_top1": float(correct[local]), "cra_mrr": float(mrr[local]),
                    "factual_l2": float(df[local].item()),
                    "negative_l2_mean": float(dn[local].mean().item()),
                })
            print(f"  {model_name}: {stop}/{len(refs)} transitions", flush=True)
    del adapter
    torch.cuda.empty_cache()
    return rows


def aggregate(per_transition: pd.DataFrame, *, n_resamples: int) -> pd.DataFrame:
    rows = []
    for model in sorted(per_transition.model.unique()):
        mdf = per_transition[per_transition.model == model]
        for regime in ["all", *sorted(mdf.regime.unique())]:
            rdf = mdf if regime == "all" else mdf[mdf.regime == regime]
            for subset in ("all", "shared_effect"):
                sdf = rdf if subset == "all" else rdf[rdf.shared_effect_mask == 1]
                for metric in ("cra_top1", "cra_mrr"):
                    ci = bootstrap_ci(
                        sdf[metric].to_numpy(), groups=sdf.traj_id.to_numpy(),
                        n_resamples=n_resamples, seed=13 if metric == "cra_top1" else 17)
                    rows.append({
                        "estimate_type": "model_mean", "model": model,
                        "model_a": "", "model_b": "", "regime": regime, "subset": subset,
                        "metric": metric, "n_transitions": len(sdf),
                        "n_trajectories": sdf.traj_id.nunique(),
                        "point": ci.point, "ci95_low": ci.low, "ci95_high": ci.high,
                    })
    # Because the manifest is exactly paired, differences should also be paired
    # at transition level (then trajectory-cluster bootstrapped).  This is the
    # statistically efficient test of a scaling trend; overlapping marginal CIs
    # are not a valid no-difference test.
    models = sorted(per_transition.model.unique())
    for model_a, model_b in itertools.combinations(models, 2):
        a = per_transition[per_transition.model == model_a].set_index("global_index")
        b = per_transition[per_transition.model == model_b].set_index("global_index")
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        for regime in ["all", *sorted(a.regime.unique())]:
            regime_mask = np.ones(len(a), dtype=bool) if regime == "all" else a.regime.to_numpy() == regime
            for subset in ("all", "shared_effect"):
                mask = regime_mask.copy()
                if subset == "shared_effect":
                    mask &= a.shared_effect_mask.to_numpy() == 1
                for metric in ("cra_top1", "cra_mrr"):
                    diff = b[metric].to_numpy()[mask] - a[metric].to_numpy()[mask]
                    groups = a.traj_id.to_numpy()[mask]
                    ci = bootstrap_ci(
                        diff, groups=groups, n_resamples=n_resamples,
                        seed=29 if metric == "cra_top1" else 31)
                    rows.append({
                        "estimate_type": "paired_difference_b_minus_a",
                        "model": f"{model_b}-minus-{model_a}",
                        "model_a": model_a, "model_b": model_b,
                        "regime": regime, "subset": subset, "metric": metric,
                        "n_transitions": len(diff), "n_trajectories": len(np.unique(groups)),
                        "point": ci.point, "ci95_low": ci.low, "ci95_high": ci.high,
                    })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--reference-model", default="dino_wm_droid")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--pool-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--effect-quantile", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--bootstrap-resamples", type=int, default=2000)
    ap.add_argument("--out-prefix", default="results/droid_shared_scaling")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    require_slurm_gpu()
    cfg = yaml.safe_load(open(args.config))
    models = args.models or cfg["models"]
    if args.reference_model not in models:
        raise SystemExit("--reference-model must be included in --models")
    cache_paths = {
        model: latent_cache_path(cfg["latent_cache"]["root"], model, cfg["dataset"]["name"])
        for model in models
    }
    missing = [str(path) for path in cache_paths.values() if not path.exists()]
    if missing:
        raise SystemExit("missing caches:\n  " + "\n  ".join(missing))
    prefix = Path(args.out_prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": Path(f"{prefix}_manifest.csv"),
        "negatives": Path(f"{prefix}_negatives.csv"),
        "protocol": Path(f"{prefix}_protocol.json"),
        "transitions": Path(f"{prefix}_per_transition.csv"),
        "summary": Path(f"{prefix}_summary.csv"),
    }
    if not args.overwrite and any(path.exists() for path in paths.values()):
        raise SystemExit(f"output exists under {prefix}; pass --overwrite to replace")

    anchors, negatives, protocol, refs, action, neg_global = build_common_manifest(
        cache_paths, reference_model=args.reference_model, K=args.K,
        pool_size=args.pool_size, seed=args.seed, effect_quantile=args.effect_quantile)
    pd.DataFrame(anchors).to_csv(paths["manifest"], index=False)
    pd.DataFrame(negatives).to_csv(paths["negatives"], index=False)
    with open(paths["protocol"], "w") as f:
        json.dump(protocol, f, indent=2)
    print(f"shared manifest: {len(anchors)} transitions, {len(negatives)} fixed negatives", flush=True)

    device = torch.device(cfg["eval"].get("device", "cuda"))
    all_rows = []
    # Write incrementally per model so a completed checkpoint survives a later failure.
    for model in models:
        rows = evaluate_model(
            model, cache_paths[model], refs, action, neg_global, anchors,
            device=device, batch_size=args.batch_size)
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(paths["transitions"], index=False)
    summary = aggregate(pd.DataFrame(all_rows), n_resamples=args.bootstrap_resamples)
    summary.to_csv(paths["summary"], index=False)
    print(f"wrote {paths['summary']} ({len(summary)} CI rows)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
