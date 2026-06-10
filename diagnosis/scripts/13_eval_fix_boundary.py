"""BB before/after the C1 fix — the success criterion (design 2026-06-09 §4.3).

Re-runs the boundary diagnostic on the *same* cache, pool, and cells as
``12_boundary_diagnostic.py`` for three adapter variants:

    base       — the frozen baseline predictor (sanity anchor; should reproduce
                 results/{dataset}_boundary.csv up to pool-sampling noise)
    +mdn_K1    — frozen trunk + trained *unimodal* head (ablation: same head
                 capacity/schedule, K=1 — isolates "distributional" as the cause)
    +mdn_K<k>  — frozen trunk + trained mixture head (the C1 fix, mode-scored)

Writes ``results/{dataset}_boundary_fix.csv``. The claim PASSES if the K≥2 row
has lower ``bb_boundary`` than both the base and the K1 rows in the boundary
regimes (pre_grasp / contact), per-model-standardised exactly like production.

    python scripts/13_eval_fix_boundary.py --config configs/diagnostic_metaworld.yaml \
        --model dino_wm_metaworld --ckpt checkpoints/mdn_dino_wm_metaworld_K3.pt \
        --ckpt-k1 checkpoints/mdn_dino_wm_metaworld_K1.pt
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import LatentCache, latent_cache_path, read_regimes  # noqa: E402
from data.latent_cache import REGIME_TO_ID  # noqa: E402
from models.adapters import build_adapter  # noqa: E402
from models.heads import MixtureDensityHead, MixturePredictorAdapter  # noqa: E402
from scripts._boundary_diagnostic import (  # noqa: E402
    _load_runner_helpers,
    accumulate_cell,
    compute_true_outcome,
    finalize_rows,
)


def load_head(ckpt_path: str, device) -> MixtureDensityHead:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    head = MixtureDensityHead(latent_dim=ckpt["latent_dim"], action_dim=ckpt["action_dim"],
                              K=ckpt["K"], hidden=ckpt["hidden"], ctx_dim=ckpt["ctx_dim"],
                              state_dim=ckpt.get("state_dim", 0))
    head.load_state_dict(ckpt["state_dict"])
    return head.to(device).eval()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ckpt", required=True, help="mixture (K>=2) head checkpoint")
    ap.add_argument("--ckpt-k1", default=None, help="optional unimodal-baseline checkpoint")
    ap.add_argument("--skip-base", action="store_true",
                    help="skip the frozen-base variant (already in {dataset}_boundary.csv)")
    args = ap.parse_args()

    torch.set_num_threads(int(os.environ.get("CAI_JEPA_TORCH_THREADS", "2")))
    cfg = yaml.safe_load(open(args.config))
    helpers = _load_runner_helpers()

    dataset_name = cfg["dataset"]["name"]
    device = torch.device(cfg["eval"]["device"] if torch.cuda.is_available() else "cpu")
    min_n = cfg["eval"]["min_transitions_per_cell"]
    similarity_radius = cfg["hard_nn"]["similarity_radius"]
    pool_size = cfg["hard_nn"]["pool_size"]
    bcfg = cfg.get("boundary", {})
    max_neighbours = int(bcfg.get("max_neighbours", cfg["cra"]["K"]))
    boundary_quantile = float(bcfg.get("quantile", 0.75))
    n_resamples = cfg["bootstrap"]["n_resamples"]
    ci = cfg["bootstrap"]["ci"]

    cache_path = latent_cache_path(cfg["latent_cache"]["root"], args.model, dataset_name)
    if not cache_path.exists():
        print(f"[error] cache missing: {cache_path}")
        return 1

    base = build_adapter(args.model, device=str(device)).eval()
    step = base.frames_per_step
    want_state = dataset_name == "metaworld"

    def make_variant(ckpt_path, tag_extra=""):
        h = load_head(ckpt_path, device)
        state_cond = h.state_dim > 0
        name = f"{args.model}+mdn_K{h.K}{tag_extra}" + ("+state" if state_cond else "")
        return name, MixturePredictorAdapter(base, h, device=str(device),
                                             state_via_proprio=state_cond), state_cond

    variants = {}        # name -> (adapter, state_conditioned)
    if not args.skip_base:
        variants[args.model] = (base, False)
    if args.ckpt_k1:
        n, a, s = make_variant(args.ckpt_k1)
        variants[n] = (a, s)
    n, a, s = make_variant(args.ckpt)
    variants[n] = (a, s)
    any_state = any(s for _, s in variants.values())

    regime_by_traj = read_regimes(cache_path)
    all_rows = []
    with LatentCache(cache_path, mode="r") as cache:
        records = helpers.build_transition_records(
            cache, regime_by_traj, step, per_task=dataset_name == "metaworld")
        # Same pool construction (seed 0) as 12_boundary_diagnostic.py so the
        # neighbourhoods are identical across before/after.
        rng = np.random.default_rng(0)
        pool_idx = rng.choice(np.arange(len(records)),
                              size=min(pool_size, len(records)), replace=False)
        pool_records = [records[int(i)] for i in pool_idx]
        pool_data = helpers.materialize_records(cache, pool_records, step,
                                                want_proprio=False, want_state=want_state)
        pool_outcome = compute_true_outcome(
            dataset_name, pool_data["z_t"], pool_data["z_t1"],
            pool_data.get("state_t"), pool_data.get("state_t1"))
        pool_z, pool_a = pool_data["z_t"], pool_data["a_t"]
        tasks = sorted({r["task"] for r in records})

        for name, (adapter, state_cond) in variants.items():
            print(f"\n=== Boundary eval: {name} ===", flush=True)
            acc = {"s_true": [], "s_model": [], "boundary_score": [],
                   "traj_tag": [], "task": [], "regime": []}
            for regime_name in cfg["regimes"]:
                regime_id = REGIME_TO_ID[regime_name]
                for task in tasks:
                    cell_records = [r for r in records
                                    if r["regime"] == regime_id and r["task"] == task]
                    if len(cell_records) < min_n:
                        continue
                    cell_data = helpers.materialize_records(
                        cache, cell_records, step,
                        want_proprio=base.uses_proprio(), want_state=any_state)
                    if state_cond:
                        # C1+D: the wrapper expects the full raw state through the
                        # proprio channel (it forwards state[:, :4] to the base).
                        cell_data = {**cell_data, "proprio_t": cell_data["state_t"]}
                    out = accumulate_cell(
                        adapter, cell_data, pool_z, pool_a, pool_outcome, device=device,
                        similarity_radius=similarity_radius, max_neighbours=max_neighbours)
                    n = len(out["s_true"])
                    acc["s_true"].append(out["s_true"])
                    acc["s_model"].append(out["s_model"])
                    acc["boundary_score"].append(out["boundary_score"])
                    acc["traj_tag"].append(out["traj_tag"])
                    acc["task"].append(np.array([task] * n))
                    acc["regime"].append(np.array([regime_name] * n))
                    del cell_data
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            rows, _ = finalize_rows(
                dataset_name, name,
                s_true=np.concatenate(acc["s_true"]),
                s_model=np.concatenate(acc["s_model"]),
                boundary_score=np.concatenate(acc["boundary_score"]),
                task=np.concatenate(acc["task"]),
                regime=np.concatenate(acc["regime"]),
                traj_tag=np.concatenate(acc["traj_tag"]),
                boundary_quantile=boundary_quantile, n_resamples=n_resamples, ci=ci)
            all_rows.extend(rows)
            for r in rows:
                print(f"  {r['task']:22s} {r['regime']:20s} n={r['n_transitions']:5d} "
                      f"BB={r['bb']:+.3f} BB_boundary={r['bb_boundary']:+.3f} "
                      f"(n_b={r['n_boundary']})", flush=True)

    import pandas as pd
    out_path = Path(cfg["output"]["csv"]).with_name(f"{dataset_name}_boundary_fix.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
