#!/usr/bin/env python3
"""Score one arm on the shared CEM populations (fifteenth amendment).

Every arm sees the identical candidate tensor, cached once from the original
frozen checkpoint at the deployed budget (N=300, K=30, T=30, H=5).  This script
only re-scores and refits; it never samples.  That is what makes the test a
counterfactual reranking on a common proposal rather than a per-arm CEM run,
and the endpoint that would justify a planning claim is a separate experiment.

Primary outcome, per arena, mirroring the deployed CEM update exactly
(``cem.py``: ``batch_mean = topk_candidates.mean(dim=1)``, then
``outputs['actions'] = mean``):

    score 300 candidates -> take the 30 lowest -> elite_mean = their mean
    -> restore the simulator -> execute elite_mean -> physical goal distance.

Secondaries come from the cached per-candidate physical distances, so they cost
no extra simulation: top-1 physical cost, mean physical cost of the 30 elites,
and the model-vs-physical rank correlation over all 300 candidates.

Arena 0 is the primary (drawn from the initial proposal, conditioned on no arm);
arena 1 is the original CEM's own final population, a secondary robustness
arena.  They are never pooled.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DIAG = REPO / "diagnosis"
PERD = REPO / "physical_search_distillation"
sys.path.insert(0, str(REPO))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot-index", type=int, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--populations-dir", type=Path, required=True)
    p.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    p.add_argument("--checkpoint", default="quentinll/lewm-cube")
    p.add_argument("--arm", action="append", required=True, metavar="NAME[=CKPT]",
                   help="arm to score; NAME alone uses the released checkpoint, "
                        "NAME=path loads that state dict strict=True on top of it. "
                        "All arms in one task share the simulator and the "
                        "population, so they cannot diverge in setup.")
    p.add_argument("--goal-offset", type=int, default=25)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--out-dir", type=Path, required=True)
    return p.parse_args()


def load_module(path: Path, alias: str) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def sha256_f4(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def expand_info(info: dict[str, Any], num_samples: int, device: str,
                dtype: torch.dtype) -> dict[str, Any]:
    """Replicate CEMSolver's per-candidate expansion exactly (cem.py)."""
    expanded: dict[str, Any] = {}
    for key, value in info.items():
        if torch.is_tensor(value):
            target = dtype if value.is_floating_point() else None
            value = (value.to(device=device, dtype=target)
                     .unsqueeze(1)
                     .expand(value.shape[0], num_samples, *value.shape[1:]))
        elif isinstance(value, np.ndarray):
            value = np.repeat(value[:, None, ...], num_samples, axis=1)
        expanded[key] = value
    return expanded


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without a scipy dependency; ties averaged."""
    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), dtype=np.float64)
        r[order] = np.arange(len(x), dtype=np.float64)
        # average ranks within tied groups
        sx = x[order]
        start = 0
        for i in range(1, len(x) + 1):
            if i == len(x) or sx[i] != sx[start]:
                if i - start > 1:
                    r[order[start:i]] = np.mean(r[order[start:i]])
                start = i
        return r
    ra, rb = ranks(np.asarray(a, dtype=np.float64)), ranks(np.asarray(b, dtype=np.float64))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return float(ra @ rb / denom) if denom > 0 else float("nan")


def parse_arm(spec: str) -> tuple[str, Path | None]:
    name, sep, path = spec.partition("=")
    if not name:
        raise ValueError(f"empty arm name in {spec!r}")
    return name, (Path(path) if sep else None)


def load_arm(swm: Any, checkpoint: str, state_dict: Path | None,
             torch_mod: Any) -> tuple[Any, dict[str, Any] | None]:
    model = swm.wm.utils.load_pretrained(checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    meta = None
    if state_dict is not None:
        blob = torch_mod.load(state_dict, map_location="cuda")
        model.load_state_dict(blob["model"], strict=True)
        meta = {k: blob.get(k) for k in ("lambda_as", "seed", "steps")}
    return model, meta


def score_arena(arena: int, populations: Any, evaluator: Any, info: dict[str, Any],
                scaler: Any, raw_env: Any, init_row: Any, goal_row: Any,
                audit: Any, corrected: Any, perd: Any, args: argparse.Namespace,
                raw_dim: int, low: np.ndarray, high: np.ndarray,
                device: str, dtype: torch.dtype) -> dict[str, Any]:
    candidates = np.asarray(populations["actions_normalized"][arena], dtype=np.float32)
    cached_physical = np.asarray(populations["physical_distance_m"][arena], dtype=np.float64)
    num_samples = candidates.shape[0]

    tensor = torch.as_tensor(candidates, device=device, dtype=dtype).unsqueeze(0)
    with torch.inference_mode():
        costs = evaluator.get_cost(expand_info(info, num_samples, device, dtype), tensor)
    costs = np.asarray(costs[0].detach().float().cpu().numpy(), dtype=np.float64)
    if costs.shape != (num_samples,):
        raise RuntimeError(f"cost shape {costs.shape} != ({num_samples},)")

    # Deployed CEM update: mean of the topk lowest-cost candidates.
    elite_inds = np.argsort(costs, kind="mergesort")[: args.topk]
    elite_mean = candidates[elite_inds].mean(axis=0)

    elite_mean_raw = perd.normalized_to_raw(
        elite_mean[None], scaler, args.horizon, args.action_block, raw_dim)
    flat = elite_mean_raw.reshape(-1, raw_dim)
    n_out = int(np.sum((flat < low) | (flat > high)))
    clipped = np.clip(elite_mean_raw, low, high)

    _, dist, succ, n_exec = corrected.rollout_population(
        raw_env, init_row, goal_row, clipped, audit)

    top1 = int(elite_inds[0])
    return {
        "arena": arena,
        "cem_step": int(populations["step"][arena]),
        "num_samples": num_samples,
        "candidate_sha256": sha256_f4(candidates),
        # primary
        "elite_mean_physical_distance_m": float(dist[0]),
        "elite_mean_success": bool(succ[0]),
        "elite_mean_executed_steps": int(n_exec[0]),
        "elite_mean_out_of_bounds_components": n_out,
        "elite_mean_out_of_bounds_fraction": float(n_out / flat.size),
        # secondaries
        "top1_physical_distance_m": float(cached_physical[top1]),
        "elite_mean_of_physical_distances_m": float(cached_physical[elite_inds].mean()),
        "rank_correlation_model_vs_physical": spearman(costs, cached_physical),
        # context, for diagnosing a floor rather than reporting one as a null
        "population_physical_min_m": float(cached_physical.min()),
        "population_physical_median_m": float(np.median(cached_physical)),
        "population_physical_max_m": float(cached_physical.max()),
        "model_cost_min": float(costs.min()),
        "model_cost_median": float(np.median(costs)),
        "model_cost_max": float(costs.max()),
        "elite_indices": [int(i) for i in elite_inds],
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("scoring requires a GPU Slurm allocation")

    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "acm_audit")
    corrected = load_module(DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "acm_corrected")
    perd = load_module(PERD / "scripts/collect_populations.py", "acm_perd_collect")

    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("manifest order/index mismatch")

    pop_path = args.populations_dir / f"snapshot_{snapshot.order:03d}/populations.npz"
    populations = np.load(pop_path)

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset)
    init_row, goal_row = init_rows[0], goal_rows[0]
    action_data = np.asarray(dataset.get_col_data("action"))
    scaler = StandardScaler().fit(action_data[~np.isnan(action_data).any(axis=1)])

    arms = [parse_arm(spec) for spec in args.arm]
    names = [n for n, _ in arms]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate arm names: {names}")

    transform = audit.make_transform(224)
    world, raw_env, visual_hash, _ = corrected.make_world(swm, snapshot)
    results: list[dict[str, Any]] = []
    try:
        raw_dim = int(np.prod(world.envs.single_action_space.shape))
        low = np.asarray(world.envs.single_action_space.low,
                         dtype=np.float32).reshape(-1)
        high = np.asarray(world.envs.single_action_space.high,
                          dtype=np.float32).reshape(-1)
        raw_info = {
            "pixels": np.asarray(init_row["pixels"])[None, None],
            "goal": np.asarray(goal_row["goal"])[None, None],
            "action": np.full((1, 1, raw_dim), np.nan, dtype=np.float32),
        }
        n_arenas = int(populations["actions_normalized"].shape[0])

        for name, state_dict in arms:
            model, meta = load_arm(swm, args.checkpoint, state_dict, torch)
            evaluator = swm.planning.ShootingCostEvaluator(
                model, swm.planning.GoalMSE())
            solver = swm.planning.CEMSolver(
                cost=evaluator, batch_size=1, num_samples=1, n_steps=1, topk=1,
                device="cuda", seed=0)
            config = swm.PlanConfig(
                horizon=args.horizon, receding_horizon=args.horizon,
                action_block=args.action_block, history_len=1, warm_start=True)
            policy = swm.policy.WorldModelPolicy(
                solver=solver, config=config, process={"action": scaler},
                transform={"pixels": transform, "goal": transform})
            policy.set_env(world.envs)
            info = policy._prepare_info(raw_info)
            dtype = next(model.parameters()).dtype
            arenas = [
                score_arena(a, populations, evaluator, info, scaler, raw_env,
                            init_row, goal_row, audit, corrected, perd, args,
                            raw_dim, low, high, "cuda", dtype)
                for a in range(n_arenas)
            ]
            results.append({
                "arm": name,
                "state_dict": str(state_dict) if state_dict else None,
                "state_dict_meta": meta,
                "arenas": arenas,
            })
            del model, evaluator, policy, solver
            torch.cuda.empty_cache()
    finally:
        world.close()

    # Every arm must have scored byte-identical candidates.
    for arena in range(len(results[0]["arenas"])):
        hashes = {r["arm"]: r["arenas"][arena]["candidate_sha256"] for r in results}
        if len(set(hashes.values())) != 1:
            raise RuntimeError(f"arena {arena} candidate mismatch across arms: {hashes}")

    out = args.out_dir / f"snapshot_{snapshot.order:03d}"
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "snapshot": snapshot.order, "episode": snapshot.episode,
        "checkpoint": args.checkpoint, "topk": args.topk,
        "horizon": args.horizon, "action_block": args.action_block,
        "visual_signature": visual_hash, "populations": str(pop_path),
        "candidate_sha256": [results[0]["arenas"][a]["candidate_sha256"]
                             for a in range(len(results[0]["arenas"]))],
        "arms": results,
    }
    (out / "cem_score.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(record, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
