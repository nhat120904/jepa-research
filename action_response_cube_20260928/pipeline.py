#!/usr/bin/env python3
"""CEM-neighborhood Cube supervision with exact same-state branching."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diagnosis" / "scripts"
OFFICIAL = "quentinll/lewm-cube"
DATASET = "ogbench/cube_single_expert.h5"
EVAL_SEED = 42
GOAL_OFFSET = 25
HORIZON = 5
BLOCK = 5
CANDIDATE_RANKS = (0, 3, 7, 14, 22, 29, 50, 90)
MIXED_INITIAL_RANKS = (0, 10, 30, 90)
MIXED_FINAL_RANKS = (0, 3, 10, 29)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dump(path: Path, data: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class GoalProgressScorer(torch.nn.Module):
    """Task-supervised cost, reading only predicted endpoint and goal latents."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(4 * dim),
            torch.nn.Linear(4 * dim, 128),
            torch.nn.GELU(),
            torch.nn.Linear(128, 1),
        )

    def forward(self, endpoint: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        goal = goal.expand_as(endpoint)
        delta = endpoint - goal
        return self.net(torch.cat((endpoint, goal, delta, delta.abs()), dim=-1)).squeeze(-1)


class ScorerObjective(torch.nn.Module):
    def __init__(self, scorer: GoalProgressScorer):
        super().__init__()
        self.scorer = scorer

    def forward(self, info: dict) -> torch.Tensor:
        endpoint = info["predicted_emb"][:, :, -1]
        goal = info["goal_emb"][:, -1][:, None]
        return self.scorer(endpoint, goal)


def sample_manifests(dataset, audit, n: int, seed: int):
    test = audit.build_manifest(dataset, 50, GOAL_OFFSET, EVAL_SEED)
    test_eps = {s.episode for s in test}
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    offsets = np.asarray(dataset.offsets, dtype=np.int64)
    eligible = np.array([i for i, length in enumerate(lengths)
                         if length > GOAL_OFFSET and i not in test_eps], dtype=np.int64)
    if len(eligible) < n:
        raise RuntimeError("not enough held-out episodes for distinct training roots")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(eligible, size=n, replace=False)
    train = []
    for i, ep in enumerate(chosen):
        step = int(rng.integers(0, int(lengths[ep]) - GOAL_OFFSET))
        train.append(audit.Snapshot(
            order=i, episode=int(ep), start_step=step,
            storage_row=int(offsets[ep] + step), reset_seed=seed + 10_000 + i,
        ))
    return train, test


def make_solver(swm, model, objective, world, scaler, transform, seed, recorder=None):
    callbacks = [recorder] if recorder is not None else []
    solver = swm.planning.CEMSolver(
        cost=swm.planning.ShootingCostEvaluator(model, objective),
        batch_size=1, num_samples=300, n_steps=30, topk=30,
        var_scale=1.0, device="cuda", seed=seed, callbacks=callbacks,
    )
    policy = swm.policy.WorldModelPolicy(
        solver=solver,
        config=swm.PlanConfig(horizon=HORIZON, receding_horizon=HORIZON,
                              action_block=BLOCK, history_len=1, warm_start=True),
        process={"action": scaler},
        transform={"pixels": transform, "goal": transform},
    )
    policy.set_env(world.envs)
    return policy


def dataset_rows(dataset, swm, snapshot):
    from stable_worldmodel.world.world import _extract_init_goal
    init, goal, _ = _extract_init_goal(
        dataset, [snapshot.episode], [snapshot.start_step], GOAL_OFFSET
    )
    return init[0], goal[0]


def reset_exact(raw_env, snapshot, init, goal, corrected, audit):
    raw_env.reset(seed=snapshot.reset_seed, options={"variation": []})
    import mujoco
    raw_env._model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    corrected.restore_complete(raw_env, init["qpos"], init["qvel"], goal, audit)
    if not np.array_equal(raw_env._data.qpos, np.asarray(init["qpos"])):
        raise RuntimeError("exact qpos restoration failed")
    if not np.array_equal(raw_env._data.qvel, np.asarray(init["qvel"])):
        raise RuntimeError("exact qvel restoration failed")
    return audit.resize_render(raw_env.render())


def collect(args, swm, audit, corrected, dataset, model, scaler, transform,
            train_manifest, run_dir):
    corrected.load_stage0_transform_images = audit.transform_images
    world, raw_env, _, _ = corrected.make_world(swm, train_manifest[0])
    dim = int(np.prod(world.envs.single_action_space.shape))
    contexts, goals, endpoints, actions, distances, successes = ([] for _ in range(6))
    try:
        for i, snapshot in enumerate(train_manifest):
            init, goal = dataset_rows(dataset, swm, snapshot)
            context_image = reset_exact(raw_env, snapshot, init, goal, corrected, audit)
            target = audit.goal_field(goal, "block_0_pos")
            recorder = audit.PopulationRecorder(final_step=29)
            policy = make_solver(swm, model, swm.planning.GoalMSE(), world, scaler,
                                 transform, args.seed + i, recorder)
            raw_info = {
                "pixels": context_image[None, None],
                "goal": np.asarray(goal["goal"])[None, None],
                "action": np.full((1, 1, dim), np.nan, dtype=np.float32),
            }
            with torch.inference_mode():
                policy.solver.solve(policy._prepare_info(raw_info))
            if set(recorder.records) != {"initial", "final"}:
                raise RuntimeError(f"CEM populations missing at snapshot {i}: {recorder.records.keys()}")
            if args.population == "mixed":
                candidate_chunks = []
                for label, ranks in (("initial", MIXED_INITIAL_RANKS),
                                     ("final", MIXED_FINAL_RANKS)):
                    record = recorder.records[label]
                    order = np.argsort(record["learned_cost"], kind="mergesort")
                    candidate_chunks.append(np.asarray(record["actions_normalized"])[order[list(ranks)]])
                norm = np.concatenate(candidate_chunks, axis=0)
            else:
                final = recorder.records["final"]
                order = np.argsort(final["learned_cost"], kind="mergesort")
                norm = np.asarray(final["actions_normalized"])[order[list(CANDIDATE_RANKS)]]
            raw = scaler.inverse_transform(norm.reshape(-1, dim)).reshape(
                len(CANDIDATE_RANKS), HORIZON * BLOCK, dim
            )
            rendered = []
            physical = []
            passed = []
            for seq in raw:
                corrected.restore_complete(raw_env, init["qpos"], init["qvel"], goal, audit)
                terminated = False
                for action in seq:
                    _, _, terminated, truncated, _ = raw_env.step(action)
                    if terminated or truncated:
                        break
                rendered.append(audit.resize_render(raw_env.render()))
                distance = audit.cube_distance(raw_env, target)
                physical.append(distance)
                passed.append(bool(terminated) or distance <= 0.04)
            if i < 2:
                # Replay one selected sequence exactly; the historical weaker
                # Cube reset failed this requirement.
                first_image = rendered[0]
                first_distance = physical[0]
                corrected.restore_complete(raw_env, init["qpos"], init["qvel"], goal, audit)
                for action in raw[0]:
                    _, _, terminal, truncated, _ = raw_env.step(action)
                    if terminal or truncated:
                        break
                if not np.array_equal(first_image, audit.resize_render(raw_env.render())):
                    raise RuntimeError("branch endpoint pixels did not replay exactly")
                if first_distance != audit.cube_distance(raw_env, target):
                    raise RuntimeError("branch physical outcome did not replay exactly")
            contexts.append(corrected.encode_images(model, context_image, transform, 16)[0])
            goals.append(corrected.encode_images(model, np.asarray(goal["goal"]), transform, 16)[0])
            endpoints.append(corrected.encode_images(model, np.stack(rendered), transform, 16))
            actions.append(norm.astype(np.float32))
            distances.append(physical)
            successes.append(passed)
            if (i + 1) % 16 == 0 or i + 1 == len(train_manifest):
                print(f"COLLECTED {i + 1}/{len(train_manifest)}", flush=True)
    finally:
        world.close()
    path = run_dir / "cem_branches.npz"
    np.savez_compressed(
        path,
        context=np.asarray(contexts, np.float32),
        goal=np.asarray(goals, np.float32),
        endpoint=np.asarray(endpoints, np.float32),
        action=np.asarray(actions, np.float32),
        physical_distance_m=np.asarray(distances, np.float32),
        success=np.asarray(successes, bool),
        episode=np.asarray([s.episode for s in train_manifest], np.int32),
        step=np.asarray([s.start_step for s in train_manifest], np.int32),
    )
    return path


def load_batch(data, indices, device):
    return {key: torch.as_tensor(data[key][indices], device=device) for key in
            ("context", "goal", "endpoint", "action", "physical_distance_m")}


def scorer_error(scorer, batch):
    endpoint = batch["endpoint"]
    goal = batch["goal"][:, None]
    target = batch["physical_distance_m"] / 0.1
    pred = scorer(endpoint, goal)
    return torch.nn.functional.smooth_l1_loss(pred, target)


def train_scorer(args, data, device, run_dir):
    n = len(data["context"])
    split = max(1, int(n * 0.8))
    scorer = GoalProgressScorer(data["context"].shape[-1]).to(device)
    opt = torch.optim.AdamW(scorer.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed + 1100)
    dev = load_batch(data, np.arange(split, n), device)
    best, best_step, best_state = float("inf"), -1, None
    history = []
    for step in range(args.scorer_updates + 1):
        if step % 25 == 0 or step == args.scorer_updates:
            scorer.eval()
            with torch.inference_mode():
                score = float(scorer_error(scorer, dev))
            history.append({"step": step, "dev_smooth_l1": score})
            if score < best:
                best, best_step = score, step
                best_state = copy.deepcopy(scorer.state_dict())
        if step == args.scorer_updates:
            break
        scorer.train()
        batch = load_batch(data, rng.integers(0, split, size=min(16, split)), device)
        opt.zero_grad(set_to_none=True)
        loss = scorer_error(scorer, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
        opt.step()
    scorer.load_state_dict(best_state)
    scorer.eval().requires_grad_(False)
    with torch.inference_mode():
        estimated = scorer(dev["endpoint"], dev["goal"][:, None])
        physical = dev["physical_distance_m"]
        chosen = estimated.argmin(dim=1)
        native = (dev["endpoint"] - dev["goal"][:, None]).square().sum(dim=-1).argmin(dim=1)
        optimum = physical.min(dim=1).values
        selected = physical.gather(1, chosen[:, None]).squeeze(1)
        native_selected = physical.gather(1, native[:, None]).squeeze(1)
        dev_selection = {
            "scorer_true_endpoint_regret_m": float((selected - optimum).mean()),
            "native_l2_true_endpoint_regret_m": float((native_selected - optimum).mean()),
            "scorer_true_endpoint_selected_success_rate": float((selected <= 0.04).float().mean()),
            "native_l2_true_endpoint_selected_success_rate": float((native_selected <= 0.04).float().mean()),
        }
    torch.save(scorer.cpu().state_dict(), run_dir / "scorer.pt")
    scorer.to(device)
    dump(run_dir / "scorer_train.json", {"best_step": best_step,
                                        "best_dev_smooth_l1": best,
                                        "dev_selection": dev_selection,
                                        "history": history})
    print(f"SCORER best={best_step} dev={best:.5f}", flush=True)
    return scorer


def predicted_endpoints(model, batch):
    context = batch["context"]
    k = batch["action"].shape[1]
    info = {
        "pixels": torch.empty(len(context), k, 1, 1, device=context.device),
        "emb": context[:, None, None].expand(-1, k, 1, -1),
    }
    return model.rollout(info, batch["action"])["predicted_emb"][:, :, -1]


def dynamics_losses(model, scorer, batch, rank_weight):
    pred = predicted_endpoints(model, batch)
    prediction = (pred - batch["endpoint"]).square().mean()
    costs = scorer(pred, batch["goal"][:, None])
    best = batch["physical_distance_m"].argmin(dim=1)
    ranking = torch.nn.functional.cross_entropy(-costs / 0.3, best)
    return prediction + rank_weight * ranking, prediction, ranking


def train_dynamics(args, original, scorer, data, arm, device, run_dir):
    model = copy.deepcopy(original).to(device)
    model.encoder.eval().requires_grad_(False)
    model.projector.eval().requires_grad_(False)
    model.predictor.requires_grad_(True)
    model.action_encoder.requires_grad_(True)
    model.pred_proj.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    n = len(data["context"])
    split = max(1, int(n * 0.8))
    rng = np.random.default_rng(args.seed + 2200)
    dev = load_batch(data, np.arange(split, n), device)
    rank_weight = args.rank_weight if arm == "rank" else 0.0
    best, best_step, best_state = float("inf"), -1, None
    history = []
    for step in range(args.updates + 1):
        if step % 20 == 0 or step == args.updates:
            model.eval()
            with torch.inference_mode():
                value, pred, rank = dynamics_losses(model, scorer, dev, rank_weight)
            score = float(value)
            history.append({"step": step, "dev_objective": score,
                            "dev_prediction": float(pred), "dev_ranking": float(rank)})
            if score < best:
                best, best_step = score, step
                best_state = {name: p.detach().cpu().clone()
                              for name, p in model.named_parameters() if p.requires_grad}
        if step == args.updates:
            break
        model.predictor.train()
        model.action_encoder.train()
        # pred_proj contains BatchNorm; preserve released running statistics.
        model.pred_proj.eval()
        batch = load_batch(data, rng.integers(0, split, size=min(16, split)), device)
        opt.zero_grad(set_to_none=True)
        total, _, _ = dynamics_losses(model, scorer, batch, rank_weight)
        if not torch.isfinite(total):
            raise RuntimeError(f"nonfinite loss in {arm} at step {step}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    model.load_state_dict(best_state, strict=False)
    model.eval().requires_grad_(False)
    torch.save(model.cpu().state_dict(), run_dir / f"{arm}.pt")
    dump(run_dir / f"{arm}_train.json", {"best_step": best_step,
                                          "best_dev_objective": best,
                                          "rank_weight": rank_weight,
                                          "history": history})
    print(f"DYNAMICS {arm} best={best_step} dev={best:.5f}", flush=True)
    del model
    torch.cuda.empty_cache()


def evaluate(args, swm, audit, corrected, dataset, original, scorer, scaler,
             transform, eval_manifest, device, run_dir):
    corrected.load_stage0_transform_images = audit.transform_images
    arms = ("original_scorer", "prediction_native", "prediction_scorer", "rank_scorer")
    summary = {}
    for arm in arms:
        model = copy.deepcopy(original).to(device)
        if arm.startswith("prediction"):
            model.load_state_dict(torch.load(run_dir / "prediction.pt", map_location="cpu", weights_only=True))
        elif arm == "rank_scorer":
            model.load_state_dict(torch.load(run_dir / "rank.pt", map_location="cpu", weights_only=True))
        model.eval().requires_grad_(False)
        objective = (swm.planning.GoalMSE() if arm == "prediction_native"
                     else ScorerObjective(scorer))
        world, raw_env, _, _ = corrected.make_world(swm, eval_manifest[0])
        dim = int(np.prod(world.envs.single_action_space.shape))
        rows = []
        try:
            for j, snapshot in enumerate(eval_manifest[:args.eval_roots]):
                path = run_dir / f"eval_{arm}_{j:02d}.json"
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                    continue
                init, goal = dataset_rows(dataset, swm, snapshot)
                frame = reset_exact(raw_env, snapshot, init, goal, corrected, audit)
                target = audit.goal_field(goal, "block_0_pos")
                goal_image = np.asarray(goal["goal"])
                policy = make_solver(swm, model, objective, world, scaler,
                                     transform, EVAL_SEED + j)
                start = time.monotonic()
                passed = False
                steps = 0
                for step in range(50):
                    info = {
                        "pixels": frame[None, None],
                        "goal": goal_image[None, None],
                        "action": np.full((1, 1, dim), np.nan, np.float32),
                        "terminated": np.zeros(1, bool),
                    }
                    with torch.inference_mode():
                        action = policy.get_action(info)[0]
                    _, _, terminated, truncated, _ = raw_env.step(action)
                    steps = step + 1
                    passed = passed or bool(terminated)
                    if passed or truncated:
                        break
                    if (step + 1) % 25 == 0:
                        frame = audit.resize_render(raw_env.render())
                distance = audit.cube_distance(raw_env, target)
                passed = passed or distance <= 0.04
                row = {"arm": arm, "root": j, "episode": snapshot.episode,
                       "start_step": snapshot.start_step, "success": passed,
                       "final_cube_distance_m": distance, "steps": steps,
                       "seconds": time.monotonic() - start}
                dump(path, row)
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            world.close()
        summary[arm] = {"successes": sum(int(x["success"]) for x in rows),
                        "count": len(rows),
                        "mean_final_cube_distance_m": float(np.mean([
                            x["final_cube_distance_m"] for x in rows
                        ])), "results": rows}
        del model
        torch.cuda.empty_cache()
    dump(run_dir / "evaluation.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--collect", type=int, default=160)
    parser.add_argument("--scorer-updates", type=int, default=300)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--eval-roots", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--rank-weight", type=float, default=0.02)
    parser.add_argument("--population", choices=("final", "mixed"), default="final")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("model, physics and analysis must run under sbatch")
    if args.collect < 2 or not 1 <= args.eval_roots <= 50:
        raise ValueError("invalid collect/eval size")
    import stable_worldmodel as swm
    corrected = load_module(DIAG / "76_ogb_true_endpoint_corrected.py", "cube_corrected")
    audit = corrected.load_stage0_module()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, args.run_dir / "pipeline.py")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    dataset = swm.data.load_dataset(DATASET, keys_to_cache=["action"])
    train_manifest, eval_manifest = sample_manifests(dataset, audit, args.collect, args.seed)
    scaler = StandardScaler()
    raw_actions = dataset.get_col_data("action")
    scaler.fit(raw_actions[~np.isnan(raw_actions).any(axis=1)])
    model = swm.wm.utils.load_pretrained(OFFICIAL).to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = audit.make_transform(224)
    checkpoint = Path(os.environ["STABLEWM_HOME"]) / "checkpoints/models--quentinll--lewm-cube/weights.pt"
    dump(args.run_dir / "provenance.json", {
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "dataset": DATASET, "collect": args.collect,
        "train_manifest": [asdict(x) for x in train_manifest],
        "eval_manifest": [asdict(x) for x in eval_manifest],
        "seed": args.seed, "scorer_updates": args.scorer_updates,
        "updates": args.updates, "lr": args.lr, "rank_weight": args.rank_weight,
        "population": args.population,
        "candidate_ranks": (CANDIDATE_RANKS if args.population == "final" else
                            {"initial": MIXED_INITIAL_RANKS, "final": MIXED_FINAL_RANKS}),
        "planner": {"horizon": HORIZON, "action_block": BLOCK,
                    "cem_samples": 300, "cem_iterations": 30, "cem_elites": 30,
                    "goal_offset": GOAL_OFFSET, "eval_budget": 50},
        "supervision": "simulator cube-to-goal distance labels train a scorer and rank loss; no privileged future at deployment",
    })
    path = collect(args, swm, audit, corrected, dataset, model, scaler, transform,
                   train_manifest, args.run_dir)
    with np.load(path) as archive:
        data = {key: archive[key] for key in archive.files}
    scorer = train_scorer(args, data, device, args.run_dir)
    for arm in ("prediction", "rank"):
        train_dynamics(args, model, scorer, data, arm, device, args.run_dir)
    evaluate(args, swm, audit, corrected, dataset, model, scorer, scaler,
             transform, eval_manifest, device, args.run_dir)


if __name__ == "__main__":
    main()
