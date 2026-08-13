#!/usr/bin/env python3
"""Matched-candidate oracle audit for LeWM on OGBench-Cube.

The planner sees only its released visual world model and latent goal cost.
MuJoCo is used after planning, offline, to evaluate exactly the same sampled
action sequences from a restored dataset state.  This separates learned
rollout error from representation-induced endpoint cost error without giving
simulator state or reward to the acting planner.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch


@dataclass(frozen=True)
class Snapshot:
    order: int
    episode: int
    start_step: int
    storage_row: int
    reset_seed: int


class PopulationRecorder:
    """Keep only the first and last CEM populations on CPU."""

    name = "matched_population"

    def __init__(self, final_step: int) -> None:
        self.final_step = final_step
        self.records: dict[str, dict[str, np.ndarray]] = {}
        self.history: list[Any] = []

    @property
    def output_key(self) -> str:
        return self.name

    def reset(self) -> None:
        self.records = {}
        self.history = []

    def start_batch(self) -> None:
        pass

    def end_solve(self) -> None:
        pass

    def __call__(self, **state: Any) -> None:
        step = int(state["step"])
        label = "initial" if step == 0 else "final" if step == self.final_step else None
        if label is None:
            return
        self.records[label] = {
            "actions_normalized": state["candidates"][0].detach().float().cpu().numpy(),
            "learned_cost": state["costs"][0].detach().float().cpu().numpy(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ogbench/cube_single_expert.h5")
    parser.add_argument("--checkpoint", default="quentinll/lewm-cube")
    parser.add_argument("--num-snapshots", type=int, default=32)
    parser.add_argument("--manifest-snapshots", type=int)
    parser.add_argument("--snapshot-index", type=int)
    parser.add_argument("--goal-offset", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--cem-steps", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--var-scale", type=float, default=1.0)
    parser.add_argument("--encode-batch", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    return parser.parse_args()


def goal_field(goal_row: dict, field: str) -> np.ndarray:
    """Resolve both the flattened HDF5 and slash-preserving column names."""
    candidates = (
        f"goal_privileged_{field}",
        f"goal_privileged/{field}",
    )
    for key in candidates:
        if key in goal_row:
            return np.asarray(goal_row[key])
    raise KeyError(f"none of {candidates} found; goal keys={sorted(goal_row)}")


def build_manifest(dataset: Any, n: int, goal_offset: int, seed: int) -> list[Snapshot]:
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    offsets = np.asarray(dataset.offsets, dtype=np.int64)
    counts = np.maximum(lengths - goal_offset, 0)
    total = int(counts.sum())
    if n > total:
        raise ValueError(f"requested {n} snapshots but only {total} valid starts exist")

    rng = np.random.default_rng(seed)
    ordinals = np.sort(rng.choice(total, size=n, replace=False))
    cumulative = np.cumsum(counts)
    rows: list[tuple[int, int, int]] = []
    for ordinal in ordinals:
        episode = int(np.searchsorted(cumulative, ordinal, side="right"))
        previous = int(cumulative[episode - 1]) if episode else 0
        start = int(ordinal - previous)
        rows.append((int(offsets[episode] + start), episode, start))
    rows.sort()
    return [
        Snapshot(
            order=i,
            episode=episode,
            start_step=start,
            storage_row=storage_row,
            reset_seed=seed + 10_000 + i,
        )
        for i, (storage_row, episode, start) in enumerate(rows)
    ]


def make_transform(size: int = 224):
    import stable_pretraining as spt
    from torchvision.transforms import v2 as transforms

    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=size),
        ]
    )


def transform_images(images: np.ndarray, transform: Any, device: str) -> torch.Tensor:
    from torchvision import tv_tensors

    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    chw = np.transpose(images, (0, 3, 1, 2))
    tensor = torch.stack([transform(tv_tensors.Image(x)) for x in chw])
    return tensor[:, None].to(device)


def resize_render(image: np.ndarray, size: tuple[int, int] = (224, 224)) -> np.ndarray:
    from PIL import Image

    image = np.asarray(image)
    if image.shape[:2] == size:
        return image.copy()
    return np.asarray(Image.fromarray(image).resize((size[1], size[0]), Image.BILINEAR))


def restore(raw_env: Any, init_row: dict, goal_row: dict, seed: int) -> np.ndarray:
    raw_env.reset(seed=seed)
    raw_env.set_state(qpos=init_row["qpos"], qvel=init_row["qvel"])
    raw_env.set_target_pos(
        cube_id=0,
        target_pos=goal_field(goal_row, "block_0_pos"),
        target_quat=goal_field(goal_row, "block_0_quat"),
    )
    # OGBench derives finite-difference observations from these fields.  The
    # released evaluator restores qpos/qvel only; aligning the previous state
    # avoids injecting the arbitrary reset state into the first transition.
    if hasattr(raw_env, "_prev_qpos"):
        raw_env._prev_qpos = raw_env._data.qpos.copy()
    if hasattr(raw_env, "_prev_qvel"):
        raw_env._prev_qvel = raw_env._data.qvel.copy()
    raw_env.post_step()
    return resize_render(raw_env.render())


def cube_distance(raw_env: Any, target: np.ndarray) -> float:
    cube = np.asarray(raw_env._data.joint("object_joint_0").qpos[:3]).copy()
    return float(np.linalg.norm(cube - np.asarray(target)))


def rollout_population(
    raw_env: Any,
    init_row: dict,
    goal_row: dict,
    reset_seed: int,
    actions_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = goal_field(goal_row, "block_0_pos")
    endpoints: list[np.ndarray] = []
    distances = np.empty(len(actions_raw), dtype=np.float64)
    successes = np.zeros(len(actions_raw), dtype=bool)
    executed = np.empty(len(actions_raw), dtype=np.int64)

    for i, sequence in enumerate(actions_raw):
        endpoint = restore(raw_env, init_row, goal_row, reset_seed)
        n_executed = 0
        terminated = False
        for action in sequence.reshape(-1, sequence.shape[-1]):
            _, _, terminated, truncated, _ = raw_env.step(action)
            n_executed += 1
            if terminated or truncated:
                break
        endpoint = resize_render(raw_env.render())
        endpoints.append(endpoint)
        distances[i] = cube_distance(raw_env, target)
        successes[i] = bool(terminated) or distances[i] <= 0.04
        executed[i] = n_executed
    return np.stack(endpoints), distances, successes, executed


@torch.inference_mode()
def endpoint_latent_costs(
    model: Any,
    endpoint_images: np.ndarray,
    goal_image: np.ndarray,
    transform: Any,
    batch_size: int,
) -> np.ndarray:
    device = str(next(model.parameters()).device)
    goal = model.encode({"pixels": transform_images(goal_image, transform, device)})["emb"]
    goal = goal[:, -1].float()
    costs: list[np.ndarray] = []
    for start in range(0, len(endpoint_images), batch_size):
        pixels = transform_images(endpoint_images[start : start + batch_size], transform, device)
        emb = model.encode({"pixels": pixels})["emb"][:, -1].float()
        costs.append((emb - goal).square().sum(dim=-1).cpu().numpy())
    return np.concatenate(costs)


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = 0.5 * (start + end - 1)
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def population_metrics(
    learned: np.ndarray,
    true_latent: np.ndarray,
    physical: np.ndarray,
    success: np.ndarray,
    start_distance: float,
) -> dict[str, float]:
    n_top = max(1, int(np.ceil(0.10 * len(learned))))
    learned_order = np.argsort(learned)
    latent_order = np.argsort(true_latent)
    physical_order = np.argsort(physical)
    selected = int(learned_order[0])
    latent_selected = int(latent_order[0])
    physical_best = int(physical_order[0])
    learned_top = set(learned_order[:n_top].tolist())
    latent_top = set(latent_order[:n_top].tolist())
    physical_top = set(physical_order[:n_top].tolist())
    return {
        "spearman_learned_true_latent": spearman(learned, true_latent),
        "spearman_true_latent_physical": spearman(true_latent, physical),
        "spearman_learned_physical": spearman(learned, physical),
        "top10pct_recall_true_latent": len(learned_top & latent_top) / n_top,
        "top10pct_recall_physical": len(learned_top & physical_top) / n_top,
        "false_elite_rate_physical": float(
            np.mean(physical[learned_order[:n_top]] > np.median(physical))
        ),
        "dynamics_selection_regret_latent": float(true_latent[selected] - true_latent.min()),
        "representation_selection_regret_m": float(physical[latent_selected] - physical.min()),
        "end_to_end_selection_regret_m": float(physical[selected] - physical.min()),
        "selected_physical_distance_m": float(physical[selected]),
        "latent_selected_physical_distance_m": float(physical[latent_selected]),
        "oracle_physical_distance_m": float(physical[physical_best]),
        "candidate_headroom_m": float(np.median(physical) - physical[physical_best]),
        "start_physical_distance_m": float(start_distance),
        "oracle_improvement_m": float(start_distance - physical[physical_best]),
        "selected_success": float(success[selected]),
        "latent_selected_success": float(success[latent_selected]),
        "oracle_candidate_success": float(success.any()),
        "representation_success_gap": float(
            int(success.any()) - int(success[latent_selected])
        ),
        "end_to_end_success_gap": float(
            int(success.any()) - int(success[selected])
        ),
    }


def bootstrap_summary(
    snapshot_metrics: list[dict[str, Any]], n_bootstrap: int, seed: int
) -> dict[str, dict[str, dict[str, float]]]:
    rng = np.random.default_rng(seed)
    output: dict[str, dict[str, dict[str, float]]] = {}
    populations = sorted({row["population"] for row in snapshot_metrics})
    for population in populations:
        group = [row for row in snapshot_metrics if row["population"] == population]
        metric_names = [k for k in group[0] if k not in {"snapshot", "population"}]
        output[population] = {}
        for name in metric_names:
            values = np.asarray([row[name] for row in group], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if not len(finite):
                output[population][name] = {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
                continue
            draws = rng.choice(finite, size=(n_bootstrap, len(finite)), replace=True).mean(axis=1)
            output[population][name] = {
                "mean": float(finite.mean()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
            }
    return output


def representation_gate(metrics: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    final = metrics["final"]
    regret_positive = final["representation_selection_regret_m"]["ci_low"] > 0
    success_gap_positive = final["representation_success_gap"]["ci_low"] > 0
    coverage_positive = final["oracle_candidate_success"]["ci_low"] > 0
    strong = regret_positive and success_gap_positive and coverage_positive
    distance_only = regret_positive and coverage_positive and not success_gap_positive
    return {
        "status": (
            "strong_pass"
            if strong
            else "distance_only_support"
            if distance_only
            else "no_pass"
        ),
        "criteria": {
            "representation_regret_ci_low_gt_zero": regret_positive,
            "representation_success_gap_ci_low_gt_zero": success_gap_positive,
            "oracle_candidate_success_ci_low_gt_zero": coverage_positive,
        },
        "population": "final",
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("candidate audit must run in a GPU Slurm allocation")
    if args.topk > args.num_samples:
        raise ValueError("topk cannot exceed num_samples")
    if args.cem_steps < 2:
        raise ValueError("cem_steps must be at least 2 to record distinct initial/final populations")

    import stable_worldmodel as swm
    from sklearn.preprocessing import StandardScaler
    from stable_worldmodel.world.world import _extract_init_goal

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset = swm.data.load_dataset(args.dataset, keys_to_cache=["action"])
    manifest_size = args.manifest_snapshots or args.num_snapshots
    full_manifest = build_manifest(dataset, manifest_size, args.goal_offset, args.seed)
    if args.snapshot_index is not None:
        if not 0 <= args.snapshot_index < len(full_manifest):
            raise ValueError(
                f"snapshot_index {args.snapshot_index} outside manifest of size {len(full_manifest)}"
            )
        manifest = [full_manifest[args.snapshot_index]]
    else:
        manifest = full_manifest
    (args.out_dir / "manifest.json").write_text(
        json.dumps([asdict(row) for row in full_manifest], indent=2, sort_keys=True) + "\n"
    )
    (args.out_dir / "selected_snapshots.json").write_text(
        json.dumps([asdict(row) for row in manifest], indent=2, sort_keys=True) + "\n"
    )

    action_scaler = StandardScaler()
    action_data = dataset.get_col_data("action")
    action_data = action_data[~np.isnan(action_data).any(axis=1)]
    action_scaler.fit(action_data)
    roundtrip = action_scaler.inverse_transform(action_scaler.transform(action_data[:1024]))
    normalizer_error = float(np.max(np.abs(roundtrip - action_data[:1024])))

    model = swm.wm.utils.load_pretrained(args.checkpoint).cuda().eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = make_transform(224)

    world = swm.World(
        "swm/OGBCube-v0",
        num_envs=1,
        max_episode_steps=2 * args.horizon * args.action_block,
        env_type="single",
        ob_type="states",
        multiview=False,
        width=224,
        height=224,
        visualize_info=False,
        terminate_at_goal=True,
        image_shape=(224, 224),
    )
    raw_env = world.envs.envs[0].unwrapped
    raw_action_dim = int(np.prod(world.envs.single_action_space.shape))
    if raw_action_dim * args.action_block <= 0:
        raise RuntimeError("invalid action shape")

    candidate_rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    determinism_rows: list[dict[str, Any]] = []

    for snapshot in manifest:
        init_rows, goal_rows, _ = _extract_init_goal(
            dataset,
            [snapshot.episode],
            [snapshot.start_step],
            args.goal_offset,
        )
        init_row, goal_row = init_rows[0], goal_rows[0]
        target = goal_field(goal_row, "block_0_pos")

        first_pixels = restore(raw_env, init_row, goal_row, snapshot.reset_seed)
        first_qpos = raw_env._data.qpos.copy()
        first_qvel = raw_env._data.qvel.copy()
        first_target = raw_env._data.mocap_pos[raw_env._cube_target_mocap_ids[0]].copy()
        start_distance = cube_distance(raw_env, target)
        second_pixels = restore(raw_env, init_row, goal_row, snapshot.reset_seed)
        determinism = {
            "snapshot": snapshot.order,
            "qpos_restore_max_abs": float(np.max(np.abs(first_qpos - raw_env._data.qpos))),
            "qvel_restore_max_abs": float(np.max(np.abs(first_qvel - raw_env._data.qvel))),
            "pixel_restore_max_abs": float(np.max(np.abs(first_pixels.astype(np.int16) - second_pixels.astype(np.int16)))),
            "target_restore_max_abs": float(np.max(np.abs(first_target - raw_env._data.mocap_pos[raw_env._cube_target_mocap_ids[0]]))),
            "dataset_qpos_max_abs": float(np.max(np.abs(np.asarray(init_row["qpos"]) - raw_env._data.qpos))),
            "dataset_qvel_max_abs": float(np.max(np.abs(np.asarray(init_row["qvel"]) - raw_env._data.qvel))),
        }
        determinism_rows.append(determinism)
        restoration_errors = [
            determinism[key]
            for key in (
                "qpos_restore_max_abs",
                "qvel_restore_max_abs",
                "pixel_restore_max_abs",
                "target_restore_max_abs",
                "dataset_qpos_max_abs",
                "dataset_qvel_max_abs",
            )
        ]
        if max(restoration_errors) > 1e-8:
            raise RuntimeError(f"snapshot restoration failed: {determinism}")

        recorder = PopulationRecorder(final_step=args.cem_steps - 1)
        cost = swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE())
        solver = swm.planning.CEMSolver(
            cost=cost,
            batch_size=1,
            num_samples=args.num_samples,
            var_scale=args.var_scale,
            n_steps=args.cem_steps,
            topk=args.topk,
            device="cuda",
            seed=args.seed + snapshot.order,
            callbacks=[recorder],
        )
        config = swm.PlanConfig(
            horizon=args.horizon,
            receding_horizon=args.horizon,
            action_block=args.action_block,
            history_len=1,
            warm_start=True,
        )
        policy = swm.policy.WorldModelPolicy(
            solver=solver,
            config=config,
            process={"action": action_scaler},
            transform={"pixels": transform, "goal": transform},
        )
        policy.set_env(world.envs)

        raw_info = {
            "pixels": np.asarray(init_row["pixels"])[None, None],
            "goal": np.asarray(goal_row["goal"])[None, None],
            "action": np.full((1, 1, raw_action_dim), np.nan, dtype=np.float32),
        }
        prepared = policy._prepare_info(raw_info)
        with torch.inference_mode():
            solver.solve(prepared)
        if set(recorder.records) != {"initial", "final"}:
            raise RuntimeError(f"missing recorded populations: {recorder.records.keys()}")

        for population, record in recorder.records.items():
            normalized = record["actions_normalized"]
            flat = normalized.reshape(-1, raw_action_dim)
            actions_raw = action_scaler.inverse_transform(flat).reshape(
                args.num_samples, args.horizon, args.action_block, raw_action_dim
            )
            endpoints, physical, successes, executed = rollout_population(
                raw_env,
                init_row,
                goal_row,
                snapshot.reset_seed,
                actions_raw,
            )
            true_latent = endpoint_latent_costs(
                model,
                endpoints,
                np.asarray(goal_row["goal"]),
                transform,
                args.encode_batch,
            )
            learned = np.asarray(record["learned_cost"])
            metrics = population_metrics(
                learned, true_latent, physical, successes, start_distance
            )
            snapshot_rows.append(
                {"snapshot": snapshot.order, "population": population, **metrics}
            )
            for candidate in range(args.num_samples):
                candidate_rows.append(
                    {
                        "snapshot": snapshot.order,
                        "population": population,
                        "candidate": candidate,
                        "learned_cost": float(learned[candidate]),
                        "true_endpoint_latent_cost": float(true_latent[candidate]),
                        "physical_distance_m": float(physical[candidate]),
                        "success": int(successes[candidate]),
                        "executed_steps": int(executed[candidate]),
                    }
                )
            np.savez_compressed(
                args.artifact_dir / f"snapshot_{snapshot.order:03d}_{population}.npz",
                actions_normalized=normalized,
                actions_raw=actions_raw,
                learned_cost=learned,
                true_endpoint_latent_cost=true_latent,
                physical_distance_m=physical,
                success=successes,
                executed_steps=executed,
            )
        print(f"AUDIT_SNAPSHOT_DONE {snapshot.order + 1}/{len(manifest)}")

    with gzip.open(args.out_dir / "candidate_costs.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    with (args.out_dir / "snapshot_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(snapshot_rows[0]))
        writer.writeheader()
        writer.writerows(snapshot_rows)

    metrics_summary = bootstrap_summary(snapshot_rows, args.bootstrap, args.seed + 99)
    summary = {
        "config": {
            "dataset": args.dataset,
            "checkpoint": args.checkpoint,
            "num_snapshots": args.num_snapshots,
            "manifest_snapshots": manifest_size,
            "processed_snapshots": len(manifest),
            "snapshot_index": args.snapshot_index,
            "goal_offset": args.goal_offset,
            "horizon": args.horizon,
            "action_block": args.action_block,
            "num_samples": args.num_samples,
            "cem_steps": args.cem_steps,
            "topk": args.topk,
            "var_scale": args.var_scale,
            "seed": args.seed,
        },
        "action_normalizer_roundtrip_max_abs": normalizer_error,
        "restoration": determinism_rows,
        "metrics": metrics_summary,
        "representation_gate": representation_gate(metrics_summary),
        "scope": "offline matched-candidate audit; simulator was not exposed to the planner",
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    world.close()
    print(json.dumps(summary["metrics"], indent=2, sort_keys=True, allow_nan=True))
    print("OGB_STAGE0_CANDIDATE_AUDIT_DONE")


if __name__ == "__main__":
    main()
