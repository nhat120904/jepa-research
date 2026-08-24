#!/usr/bin/env python3
"""Mine same-state physical counterfactual action chunks on OGBench-Cube.

This Phase-0 builder deliberately reuses only *persisted* CEM populations from
the corrected LeWM audit.  For each locked snapshot it restores the entire
MuJoCo state before every candidate rollout, labels all candidates by physical
endpoint distance, and identifies a low-proxy/high-regret final-CEM example.
The saved pool is the input to a later policy-training phase; no policy is
trained here.
"""

from __future__ import annotations

import argparse
import csv
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
DEFAULT_ARTIFACTS = Path("/mnt/data/nhatnc129/jepa/lewm_stage0/artifacts/audit_locked_array")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument(
        "--artifact-root", type=Path, default=DEFAULT_ARTIFACTS,
        help="directory containing snapshot_XXX_{initial,final}.npz",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=DIAG / "results/ogb_stage0/audit_locked/manifest.json",
    )
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--physical-atol", type=float, default=1e-5)
    parser.add_argument("--proxy-top-frac", type=float, default=0.10)
    parser.add_argument("--min-regret-m", type=float, default=0.02)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_module(path: Path, alias: str) -> Any:
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def hash_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.sha256(value.tobytes()).hexdigest()


def rank_fraction(costs: np.ndarray) -> np.ndarray:
    """Stable ascending rank normalized to [0, 1]."""
    order = np.argsort(costs, kind="mergesort")
    rank = np.empty(len(costs), dtype=np.int64)
    rank[order] = np.arange(len(costs), dtype=np.int64)
    return rank / max(len(costs) - 1, 1)


def load_population(root: Path, snapshot: int, source: str) -> dict[str, np.ndarray]:
    path = root / f"snapshot_{snapshot:03d}_{source}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as artifact:
        needed = ("actions_normalized", "actions_raw", "learned_cost")
        missing = [name for name in needed if name not in artifact.files]
        if missing:
            raise RuntimeError(f"{path} missing {missing}")
        result = {name: np.asarray(artifact[name]) for name in needed}
    if result["actions_raw"].shape[0] != len(result["learned_cost"]):
        raise RuntimeError(f"candidate count mismatch in {path}")
    if result["actions_normalized"].shape[0] != len(result["learned_cost"]):
        raise RuntimeError(f"normalized candidate count mismatch in {path}")
    return result


def select_group(
    source: np.ndarray,
    proxy: np.ndarray,
    physical: np.ndarray,
    proxy_top_frac: float,
    min_regret_m: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return physical best, deceptive final-CEM, and matched initial control.

    The control has comparable physical regret but is forced outside the best
    proxy decile whenever possible.  This prevents a later policy experiment
    from confusing generic bad outcomes with the proxy-deception mechanism.
    """
    regret = physical - float(physical.min())
    proxy_rank = rank_fraction(proxy)
    positive = int(np.argmin(physical))
    final = np.flatnonzero(source == "cem_final")
    initial = np.flatnonzero(source == "cem_initial")
    deceptive_pool = final[
        (proxy_rank[final] <= proxy_top_frac) & (regret[final] >= min_regret_m)
    ]
    if len(deceptive_pool) == 0:
        return {
            "positive_index": positive,
            "deceptive_index": None,
            "control_index": None,
            "control_relaxed": None,
            "has_deceptive": False,
        }, regret

    # Prefer the candidate whose false optimism is most harmful physically;
    # ties use the lower proxy cost, then stable index order.
    deceptive = int(
        deceptive_pool[np.lexsort((deceptive_pool, proxy[deceptive_pool], -regret[deceptive_pool]))[0]]
    )
    target_regret = regret[deceptive]
    non_elite_initial = initial[proxy_rank[initial] > proxy_top_frac]
    control_pool = non_elite_initial if len(non_elite_initial) else initial
    control = int(control_pool[np.argmin(np.abs(regret[control_pool] - target_regret))])
    return {
        "positive_index": positive,
        "deceptive_index": deceptive,
        "control_index": control,
        "control_relaxed": bool(len(non_elite_initial) == 0),
        "has_deceptive": True,
    }, regret


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("counterfactual mining requires a GPU Slurm allocation")
    if not 0 < args.proxy_top_frac < 1:
        raise ValueError("proxy-top-frac must lie strictly between zero and one")
    if args.min_regret_m <= 0 or args.physical_atol <= 0:
        raise ValueError("regret and tolerance thresholds must be positive")

    # Import the corrected reset/render helpers, not the historical evaluator.
    audit = load_module(DIAG / "scripts/72_ogb_stage0_candidate_audit.py", "cf_ogb_audit")
    corrected = load_module(
        DIAG / "scripts/76_ogb_true_endpoint_corrected.py", "cf_ogb_corrected"
    )
    import stable_worldmodel as swm
    from stable_worldmodel.world.world import _extract_init_goal

    manifest = json.loads(args.manifest.read_text())
    if not 0 <= args.snapshot_index < len(manifest):
        raise ValueError("snapshot-index outside manifest")
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("manifest order/index mismatch")

    populations = {
        "cem_initial": load_population(args.artifact_root, snapshot.order, "initial"),
        "cem_final": load_population(args.artifact_root, snapshot.order, "final"),
    }
    raw_shape = populations["cem_initial"]["actions_raw"].shape[1:]
    if populations["cem_final"]["actions_raw"].shape[1:] != raw_shape:
        raise RuntimeError("initial/final action shapes differ")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset
    )
    init_row, goal_row = init_rows[0], goal_rows[0]

    world, raw_env, visual_hash, visual_shapes = corrected.make_world(swm, snapshot)
    try:
        all_physical: dict[str, np.ndarray] = {}
        all_success: dict[str, np.ndarray] = {}
        all_executed: dict[str, np.ndarray] = {}
        for source, population in populations.items():
            # Uses full mjData reset and disabled warm-start before every action
            # sequence. Endpoint images are intentionally discarded here.
            _, physical, success, executed = corrected.rollout_population(
                raw_env,
                init_row,
                goal_row,
                population["actions_raw"],
                audit,
            )
            all_physical[source] = physical
            all_success[source] = success
            all_executed[source] = executed

        # Integrity replay of a deterministic candidate in an independently
        # reset state; the selected candidate is enough to catch reset leakage.
        check_actions = populations["cem_final"]["actions_raw"][:1]
        _, p1, s1, e1 = corrected.rollout_population(
            raw_env, init_row, goal_row, check_actions, audit
        )
        _, p2, s2, e2 = corrected.rollout_population(
            raw_env, init_row, goal_row, check_actions, audit
        )
    finally:
        world.close()

    repeat_gate = {
        "physical_max_abs": corrected.max_abs(p1, p2),
        "success_disagreements": int(np.sum(s1 != s2)),
        "executed_max_abs": corrected.max_abs(e1, e2),
    }
    repeat_gate["pass"] = bool(
        repeat_gate["physical_max_abs"] <= args.physical_atol
        and repeat_gate["success_disagreements"] == 0
        and repeat_gate["executed_max_abs"] == 0
    )
    if not repeat_gate["pass"]:
        raise RuntimeError(f"repeatability gate failed: {repeat_gate}")

    source = np.concatenate(
        [np.full(len(populations[name]["learned_cost"]), name) for name in populations]
    )
    proxy = np.concatenate([populations[name]["learned_cost"] for name in populations])
    physical = np.concatenate([all_physical[name] for name in populations])
    success = np.concatenate([all_success[name] for name in populations])
    executed = np.concatenate([all_executed[name] for name in populations])
    actions_raw = np.concatenate([populations[name]["actions_raw"] for name in populations])
    actions_normalized = np.concatenate(
        [populations[name]["actions_normalized"] for name in populations]
    )
    group, regret = select_group(
        source, proxy, physical, args.proxy_top_frac, args.min_regret_m
    )
    proxy_rank = rank_fraction(proxy)
    physical_rank = rank_fraction(physical)

    labels = np.full(len(source), "candidate", dtype="U20")
    labels[group["positive_index"]] = "physical_positive"
    if group["has_deceptive"]:
        labels[int(group["deceptive_index"])] = "proxy_deceptive"
        labels[int(group["control_index"])] = "matched_control"

    np.savez_compressed(
        args.out_dir / "counterfactual_pool.npz",
        actions_raw=actions_raw,
        actions_normalized=actions_normalized,
        source=source,
        learned_proxy_cost=proxy,
        physical_distance_m=physical,
        physical_regret_m=regret,
        proxy_rank_fraction=proxy_rank,
        physical_rank_fraction=physical_rank,
        success=success,
        executed_steps=executed,
        label=labels,
    )
    fields = [
        "snapshot", "candidate", "source", "label", "learned_proxy_cost",
        "physical_distance_m", "physical_regret_m", "proxy_rank_fraction",
        "physical_rank_fraction", "success", "executed_steps",
    ]
    with (args.out_dir / "candidates.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in range(len(source)):
            writer.writerow({
                "snapshot": snapshot.order,
                "candidate": index,
                "source": source[index],
                "label": labels[index],
                "learned_proxy_cost": float(proxy[index]),
                "physical_distance_m": float(physical[index]),
                "physical_regret_m": float(regret[index]),
                "proxy_rank_fraction": float(proxy_rank[index]),
                "physical_rank_fraction": float(physical_rank[index]),
                "success": int(success[index]),
                "executed_steps": int(executed[index]),
            })

    summary: dict[str, Any] = {
        "snapshot": snapshot.order,
        "episode": int(snapshot.episode),
        "start_step": int(snapshot.start_step),
        "goal_offset": args.goal_offset,
        "n_candidates": int(len(source)),
        "visual_signature": visual_hash,
        "visual_signature_shapes": visual_shapes,
        "repeat_gate": repeat_gate,
        "thresholds": {
            "proxy_top_frac": args.proxy_top_frac,
            "min_regret_m": args.min_regret_m,
        },
        "source_metrics": {
            name: {
                "n": int(len(all_physical[name])),
                "best_physical_distance_m": float(all_physical[name].min()),
                "mean_physical_distance_m": float(all_physical[name].mean()),
                "success_available": int(all_success[name].any()),
            }
            for name in populations
        },
        "group": group,
        "pool_physical_best_m": float(physical.min()),
        "pool_success_available": int(success.any()),
        "pool_sha256": hash_array(actions_raw),
    }
    if group["has_deceptive"]:
        deceptive = int(group["deceptive_index"])
        control = int(group["control_index"])
        summary["deceptive"] = {
            "proxy_cost": float(proxy[deceptive]),
            "proxy_rank_fraction": float(proxy_rank[deceptive]),
            "physical_distance_m": float(physical[deceptive]),
            "physical_regret_m": float(regret[deceptive]),
            "success": int(success[deceptive]),
        }
        summary["matched_control"] = {
            "proxy_cost": float(proxy[control]),
            "proxy_rank_fraction": float(proxy_rank[control]),
            "physical_distance_m": float(physical[control]),
            "physical_regret_m": float(regret[control]),
            "success": int(success[control]),
            "absolute_regret_gap_m": float(abs(regret[control] - regret[deceptive])),
        }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
