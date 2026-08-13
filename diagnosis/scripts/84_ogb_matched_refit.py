#!/usr/bin/env python3
"""Matched-refit branched CEM on OGBench-Cube under exact simulator dynamics.

Two CEM branches start from a bitwise identical population and share the
standard-normal noise of every iteration.  They differ only in the scalar used
to pick elites: the frozen LeWM encoder's terminal squared L2 to the goal, or a
physical cost built from simulator state.  Every candidate of every iteration is
executed in MuJoCo from the same restored snapshot, rendered, and encoded, so
the learned predictor is never called and dynamics error cannot explain a gap.

The reset/render protocol and its determinism gates are imported from
``76_ogb_true_endpoint_corrected.py`` rather than reimplemented: the retracted
historical Stage-0 artifact was caused by a weaker reset, and this experiment
must not reintroduce it.

Requires a GPU Slurm allocation; see
``docs/plans/2026-08-12-ogb-matched-refit-design.md`` for the locked protocol.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BRANCHES = ("latent", "physical")

ITERATION_FIELDS = [
    "snapshot", "iter", "branch", "n_candidates", "n_elite",
    "noise_hash", "identical_to_other_branch",
    "pre_mean_l2", "pre_var_mean", "post_mean_l2", "post_var_mean",
    "best_task_distance_m", "best_shaped_cost",
    "selected_task_distance_m", "selected_shaped_cost", "selection_regret_m",
    "mean_task_distance_m", "median_task_distance_m",
    "latent_physical_spearman", "latent_task_spearman",
    "latent_top10pct_recall_physical", "latent_top10pct_recall_task",
    "n_success", "success_any",
]

CANDIDATE_FIELDS = [
    "snapshot", "iter", "branch", "candidate", "action_hash", "noise_hash",
    "latent_rendered_goal_cost", "latent_dataset_goal_cost",
    "shaped_physical_cost", "cube_goal_distance_m", "hand_cube_distance_m",
    "success", "executed_steps", "elite_by_latent", "elite_by_physical",
    "argmin_by_latent", "argmin_by_physical",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/ogb_stage0/audit_locked/manifest.json"),
    )
    parser.add_argument("--snapshot-index", type=int, required=True)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--cem-iterations", type=int, default=30)
    parser.add_argument("--w-hand", type=float, default=0.5)
    parser.add_argument("--noise-seed", type=int, default=20260812)
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--physical-atol", type=float, default=1e-5)
    parser.add_argument("--latent-atol", type=float, default=1e-5)
    parser.add_argument(
        "--provenance-artifacts",
        type=Path,
        help="locked Stage-0 candidate npz directory; enables the executor gate",
    )
    parser.add_argument(
        "--provenance-shards",
        type=Path,
        help="corrected true-endpoint shard directory used as the physical reference",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def load_script_module(filename: str, alias: str) -> Any:
    source = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(alias, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


def hash_array(value: np.ndarray) -> str:
    import hashlib

    canonical = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    return hashlib.blake2b(canonical.tobytes(), digest_size=8).hexdigest()


def cube_position(raw_env: Any) -> np.ndarray:
    return np.asarray(raw_env._data.joint("object_joint_0").qpos[:3]).copy()


def pinch_position(raw_env: Any) -> np.ndarray:
    """Effector site OGBench reports as ``proprio/effector_pos``."""

    return np.asarray(raw_env._data.site_xpos[raw_env._pinch_site_id]).copy()


def execute_population(
    raw_env: Any,
    init_row: dict[str, Any],
    goal_row: dict[str, Any],
    actions_raw: np.ndarray,
    audit: Any,
    corrected: Any,
    target: np.ndarray,
) -> dict[str, np.ndarray]:
    """Roll every candidate from the same restored snapshot in the simulator."""

    endpoints: list[np.ndarray] = []
    cube_goal = np.empty(len(actions_raw), dtype=np.float64)
    hand_cube = np.empty(len(actions_raw), dtype=np.float64)
    success = np.zeros(len(actions_raw), dtype=bool)
    executed = np.empty(len(actions_raw), dtype=np.int64)
    for index, sequence in enumerate(actions_raw):
        corrected.restore_complete(
            raw_env, init_row["qpos"], init_row["qvel"], goal_row, audit
        )
        terminated = False
        n_executed = 0
        for action in sequence.reshape(-1, sequence.shape[-1]):
            _, _, terminated, truncated, _ = raw_env.step(action)
            n_executed += 1
            if terminated or truncated:
                break
        endpoints.append(audit.resize_render(raw_env.render()))
        cube = cube_position(raw_env)
        cube_goal[index] = float(np.linalg.norm(cube - target))
        hand_cube[index] = float(np.linalg.norm(pinch_position(raw_env) - cube))
        success[index] = bool(terminated) or cube_goal[index] <= 0.04
        executed[index] = n_executed
    return {
        "endpoints": np.stack(endpoints),
        "cube_goal": cube_goal,
        "hand_cube": hand_cube,
        "success": success,
        "executed": executed,
    }


def score_population(
    raw_env: Any,
    init_row: dict[str, Any],
    goal_row: dict[str, Any],
    actions_raw: np.ndarray,
    audit: Any,
    corrected: Any,
    model: Any,
    transform: Any,
    encode_batch: int,
    target: np.ndarray,
    rendered_goal_emb: np.ndarray,
    dataset_goal_emb: np.ndarray,
    w_hand: float,
) -> dict[str, np.ndarray]:
    rolled = execute_population(
        raw_env, init_row, goal_row, actions_raw, audit, corrected, target
    )
    emb = corrected.encode_images(model, rolled["endpoints"], transform, encode_batch)
    rolled["latent"] = corrected.squared_l2(emb, rendered_goal_emb)
    rolled["latent_dataset"] = corrected.squared_l2(emb, dataset_goal_emb)
    rolled["physical"] = rolled["cube_goal"] + w_hand * rolled["hand_cube"]
    del rolled["endpoints"]
    return rolled


def top_recall(costs: np.ndarray, reference: np.ndarray) -> float:
    n_top = max(1, int(np.ceil(0.10 * len(costs))))
    cost_top = set(np.argsort(costs, kind="mergesort")[:n_top].tolist())
    ref_top = set(np.argsort(reference, kind="mergesort")[:n_top].tolist())
    return len(cost_top & ref_top) / n_top


def iteration_row(
    snapshot_order: int,
    iteration: int,
    branch: str,
    scored: dict[str, np.ndarray],
    audit: Any,
    noise_hash: str,
    identical: bool,
    pre_mean: np.ndarray,
    pre_var: np.ndarray,
    post_mean: np.ndarray,
    post_var: np.ndarray,
    n_samples: int,
    n_elite: int,
) -> dict[str, Any]:
    own_cost = scored[branch]
    selected = int(np.argmin(own_cost))
    best_physical = int(np.argmin(scored["physical"]))
    return {
        "snapshot": snapshot_order,
        "iter": iteration,
        "branch": branch,
        "n_candidates": n_samples,
        "n_elite": n_elite,
        "noise_hash": noise_hash,
        "identical_to_other_branch": int(identical),
        "pre_mean_l2": float(np.linalg.norm(pre_mean)),
        "pre_var_mean": float(pre_var.mean()),
        "post_mean_l2": float(np.linalg.norm(post_mean)),
        "post_var_mean": float(post_var.mean()),
        "best_task_distance_m": float(scored["cube_goal"].min()),
        "best_shaped_cost": float(scored["physical"].min()),
        "selected_task_distance_m": float(scored["cube_goal"][selected]),
        "selected_shaped_cost": float(scored["physical"][selected]),
        "selection_regret_m": float(
            scored["physical"][selected] - scored["physical"][best_physical]
        ),
        "mean_task_distance_m": float(scored["cube_goal"].mean()),
        "median_task_distance_m": float(np.median(scored["cube_goal"])),
        "latent_physical_spearman": audit.spearman(scored["latent"], scored["physical"]),
        "latent_task_spearman": audit.spearman(scored["latent"], scored["cube_goal"]),
        "latent_top10pct_recall_physical": top_recall(scored["latent"], scored["physical"]),
        "latent_top10pct_recall_task": top_recall(scored["latent"], scored["cube_goal"]),
        "n_success": int(scored["success"].sum()),
        "success_any": int(scored["success"].any()),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("matched-refit audit requires a GPU Slurm allocation")
    if args.topk > args.num_samples:
        raise ValueError("topk cannot exceed num-samples")
    if args.cem_iterations < 2:
        raise ValueError("at least two iterations are needed for a refit to exist")

    import mujoco  # noqa: F401  (imported for the same side effects as the audits)
    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    audit = load_script_module(
        "72_ogb_stage0_candidate_audit.py", "ogb_stage0_audit_for_matched_refit"
    )
    corrected = load_script_module(
        "76_ogb_true_endpoint_corrected.py", "ogb_corrected_for_matched_refit"
    )
    corrected.load_stage0_transform_images = audit.transform_images

    manifest = json.loads(args.manifest.read_text())
    if not 0 <= args.snapshot_index < len(manifest):
        raise ValueError("snapshot index outside persisted manifest")
    snapshot = audit.Snapshot(**manifest[args.snapshot_index])
    if snapshot.order != args.snapshot_index:
        raise RuntimeError("persisted manifest order/index mismatch")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    init_rows, goal_rows, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], args.goal_offset
    )
    init_row, goal_row = init_rows[0], goal_rows[0]
    target = audit.goal_field(goal_row, "block_0_pos")

    action_scaler = StandardScaler()
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    action_scaler.fit(action_data)

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = audit.make_transform(224)

    world, raw_env, visual_hash, _ = corrected.make_world(swm, snapshot)
    action_dim = int(np.prod(world.envs.single_action_space.shape))
    flat_dim = action_dim * args.action_block

    gates: dict[str, Any] = {}
    iteration_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    try:
        rendered_goal = corrected.render_state(
            raw_env,
            corrected.resolve_goal(goal_row, "qpos"),
            corrected.resolve_goal(goal_row, "qvel"),
            goal_row,
            audit,
        )
        rendered_goal_emb = corrected.encode_images(
            model, rendered_goal, transform, args.encode_batch
        )[0]
        dataset_goal_emb = corrected.encode_images(
            model, np.asarray(goal_row["goal"]), transform, args.encode_batch
        )[0]
        corrected.restore_complete(
            raw_env, init_row["qpos"], init_row["qvel"], goal_row, audit
        )
        start_distance = float(np.linalg.norm(cube_position(raw_env) - target))

        # ---- gate 2: the executor must reproduce the corrected audit exactly
        if args.provenance_artifacts is not None and args.provenance_shards is not None:
            path = (
                args.provenance_artifacts
                / f"snapshot_{snapshot.order:03d}_final.npz"
            )
            with np.load(path, allow_pickle=False) as artifact:
                locked_actions = np.asarray(artifact["actions_raw"])
            replay = score_population(
                raw_env, init_row, goal_row, locked_actions, audit, corrected,
                model, transform, args.encode_batch, target, rendered_goal_emb,
                dataset_goal_emb, args.w_hand,
            )
            reference_physical, reference_success, reference_executed = (
                corrected.load_reference_physical(args.provenance_shards, snapshot.order)
            )
            reference_latent = load_reference_latent(
                args.provenance_shards, snapshot.order
            )
            gates["provenance"] = {
                "physical_max_abs": corrected.max_abs(replay["cube_goal"], reference_physical),
                "success_disagreements": int(
                    np.sum(replay["success"] != reference_success)
                ),
                "executed_max_abs": corrected.max_abs(replay["executed"], reference_executed),
                "latent_max_abs": corrected.max_abs(replay["latent"], reference_latent),
            }
        else:
            gates["provenance"] = None

        # ---- branched CEM
        rng = np.random.default_rng(args.noise_seed + snapshot.order)
        n_elite = args.topk
        state = {
            branch: {
                "mean": np.zeros((args.horizon, flat_dim), dtype=np.float64),
                "var": np.full((args.horizon, flat_dim), args.var_scale, dtype=np.float64),
            }
            for branch in BRANCHES
        }
        started = time.time()
        repeat_gate: dict[str, Any] | None = None

        for iteration in range(args.cem_iterations):
            eps = rng.standard_normal((args.num_samples, args.horizon, flat_dim))
            noise_hash = hash_array(eps)

            populations: dict[str, np.ndarray] = {}
            for branch in BRANCHES:
                # Released CEM rule: scale by ``var`` (an elite std), force
                # candidate 0 to the current mean, no clipping.
                samples = (
                    state[branch]["mean"][None] + eps * state[branch]["var"][None]
                ).astype(np.float32)
                samples[0] = state[branch]["mean"].astype(np.float32)
                populations[branch] = samples

            identical = np.array_equal(populations["latent"], populations["physical"])
            if iteration == 0 and not identical:
                raise RuntimeError("iteration-0 branch populations are not identical")

            scored_cache: dict[str, np.ndarray] | None = None
            for branch in BRANCHES:
                samples = populations[branch]
                actions_raw = action_scaler.inverse_transform(
                    samples.reshape(-1, action_dim)
                ).reshape(args.num_samples, args.horizon, args.action_block, action_dim)
                if identical and scored_cache is not None:
                    scored = scored_cache
                else:
                    scored = score_population(
                        raw_env, init_row, goal_row, actions_raw, audit, corrected,
                        model, transform, args.encode_batch, target,
                        rendered_goal_emb, dataset_goal_emb, args.w_hand,
                    )
                    if identical:
                        scored_cache = scored

                orders = {
                    name: np.argsort(scored[name], kind="mergesort")
                    for name in BRANCHES
                }
                elites = {name: order[:n_elite] for name, order in orders.items()}
                own_elite = elites[branch]
                flat_samples = samples.reshape(args.num_samples, -1).astype(np.float64)
                post_mean = flat_samples[own_elite].mean(axis=0).reshape(
                    args.horizon, flat_dim
                )
                post_var = flat_samples[own_elite].std(axis=0).reshape(
                    args.horizon, flat_dim
                )

                iteration_rows.append(iteration_row(
                    snapshot.order, iteration, branch, scored, audit, noise_hash,
                    identical, state[branch]["mean"], state[branch]["var"],
                    post_mean, post_var, args.num_samples, n_elite,
                ))

                is_last = iteration == args.cem_iterations - 1
                if iteration == 0 or is_last:
                    elite_sets = {
                        name: set(index.tolist()) for name, index in elites.items()
                    }
                    argmins = {name: int(order[0]) for name, order in orders.items()}
                    for candidate in range(args.num_samples):
                        candidate_rows.append({
                            "snapshot": snapshot.order,
                            "iter": iteration,
                            "branch": branch,
                            "candidate": candidate,
                            "action_hash": hash_array(samples[candidate]),
                            "noise_hash": noise_hash,
                            "latent_rendered_goal_cost": float(scored["latent"][candidate]),
                            "latent_dataset_goal_cost": float(
                                scored["latent_dataset"][candidate]
                            ),
                            "shaped_physical_cost": float(scored["physical"][candidate]),
                            "cube_goal_distance_m": float(scored["cube_goal"][candidate]),
                            "hand_cube_distance_m": float(scored["hand_cube"][candidate]),
                            "success": int(scored["success"][candidate]),
                            "executed_steps": int(scored["executed"][candidate]),
                            "elite_by_latent": int(candidate in elite_sets["latent"]),
                            "elite_by_physical": int(candidate in elite_sets["physical"]),
                            "argmin_by_latent": int(candidate == argmins["latent"]),
                            "argmin_by_physical": int(candidate == argmins["physical"]),
                        })

                # ---- gate 3: repeat determinism on the final latent population
                if is_last and branch == "latent":
                    repeat = score_population(
                        raw_env, init_row, goal_row, actions_raw, audit, corrected,
                        model, transform, args.encode_batch, target,
                        rendered_goal_emb, dataset_goal_emb, args.w_hand,
                    )
                    repeat_gate = {
                        "cube_goal_max_abs": corrected.max_abs(
                            repeat["cube_goal"], scored["cube_goal"]
                        ),
                        "hand_cube_max_abs": corrected.max_abs(
                            repeat["hand_cube"], scored["hand_cube"]
                        ),
                        "latent_max_abs": corrected.max_abs(
                            repeat["latent"], scored["latent"]
                        ),
                        "executed_max_abs": corrected.max_abs(
                            repeat["executed"], scored["executed"]
                        ),
                        "success_disagreements": int(
                            np.sum(repeat["success"] != scored["success"])
                        ),
                    }

                state[branch]["mean"], state[branch]["var"] = post_mean, post_var

            print(
                f"iter={iteration:02d} "
                + " ".join(
                    f"{row['branch']}_best={row['best_task_distance_m']:.4f}"
                    for row in iteration_rows[-len(BRANCHES):]
                )
                + f" minutes={(time.time() - started) / 60:.1f}",
                flush=True,
            )
    finally:
        world.close()

    gates["repeat"] = repeat_gate
    checks = {
        "iteration_zero_shared": bool(
            iteration_rows[0]["identical_to_other_branch"] == 1
        ),
        "repeat_cube_exact": repeat_gate is not None
        and repeat_gate["cube_goal_max_abs"] <= args.physical_atol,
        "repeat_hand_exact": repeat_gate is not None
        and repeat_gate["hand_cube_max_abs"] <= args.physical_atol,
        "repeat_latent_exact": repeat_gate is not None
        and repeat_gate["latent_max_abs"] <= args.latent_atol,
        "repeat_success_exact": repeat_gate is not None
        and repeat_gate["success_disagreements"] == 0,
        "repeat_executed_exact": repeat_gate is not None
        and repeat_gate["executed_max_abs"] == 0,
    }
    if gates["provenance"] is not None:
        provenance = gates["provenance"]
        checks.update({
            "provenance_physical_exact": provenance["physical_max_abs"] <= args.physical_atol,
            "provenance_success_exact": provenance["success_disagreements"] == 0,
            "provenance_executed_exact": provenance["executed_max_abs"] == 0,
            "provenance_latent_exact": provenance["latent_max_abs"] <= args.latent_atol,
        })
    gate = {"pass": bool(all(checks.values())), "checks": checks}

    final = {
        row["branch"]: row
        for row in iteration_rows
        if row["iter"] == args.cem_iterations - 1
    }
    primary = {
        "delta_best_task_distance_m": float(
            final["latent"]["best_task_distance_m"] - final["physical"]["best_task_distance_m"]
        ),
        "delta_best_shaped_cost": float(
            final["latent"]["best_shaped_cost"] - final["physical"]["best_shaped_cost"]
        ),
        "delta_selected_task_distance_m": float(
            final["latent"]["selected_task_distance_m"]
            - final["physical"]["selected_task_distance_m"]
        ),
        "delta_mean_task_distance_m": float(
            final["latent"]["mean_task_distance_m"]
            - final["physical"]["mean_task_distance_m"]
        ),
        "delta_success_any": int(
            final["latent"]["success_any"] - final["physical"]["success_any"]
        ),
        "latent_final_best_task_distance_m": float(final["latent"]["best_task_distance_m"]),
        "physical_final_best_task_distance_m": float(final["physical"]["best_task_distance_m"]),
    }

    with (args.out_dir / "iteration_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ITERATION_FIELDS)
        writer.writeheader()
        writer.writerows(iteration_rows)
    with gzip.open(args.out_dir / "candidate_costs.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(candidate_rows)

    summary = {
        "config": {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "snapshot_index": snapshot.order,
            "goal_offset": args.goal_offset,
            "horizon": args.horizon,
            "action_block": args.action_block,
            "num_samples": args.num_samples,
            "topk": args.topk,
            "var_scale": args.var_scale,
            "cem_iterations": args.cem_iterations,
            "w_hand": args.w_hand,
            "noise_seed": args.noise_seed,
            "goal_arm": "same_renderer",
            "warmstart_disabled": True,
            "visual_variation": [],
        },
        "snapshot": asdict(snapshot),
        "visual_signature": visual_hash,
        "start_distance_m": start_distance,
        "gates": gates,
        "gate": gate,
        "primary": primary,
        "runtime_minutes": (time.time() - started) / 60.0,
        "scope": (
            "cost-only CEM intervention with simulator-exact dynamics; simulator "
            "state is a positive control, never a deployable selector"
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    (args.out_dir / "manifest_row.json").write_text(
        json.dumps(asdict(snapshot), indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))
    print("OGB_MATCHED_REFIT_PASS" if gate["pass"] else "OGB_MATCHED_REFIT_BLOCKED")
    if not gate["pass"]:
        raise SystemExit(2)


def load_reference_latent(root: Path, snapshot_index: int) -> np.ndarray:
    """Same-renderer true-endpoint costs recorded by the corrected audit."""

    path = root / str(snapshot_index) / "candidate_costs.csv.gz"
    rows: list[dict[str, str]] = []
    with gzip.open(path, "rt", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    rows.sort(key=lambda row: int(row["candidate"]))
    return np.asarray([float(row["true_rendered_goal"]) for row in rows])


if __name__ == "__main__":
    main()
