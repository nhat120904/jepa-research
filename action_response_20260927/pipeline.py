#!/usr/bin/env python3
"""Matched-branch action-response fine-tuning and native PushT control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import gymnasium as gym
import h5py
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as transforms


OFFICIAL = "quentinll/lewm-pusht"
DATA = Path(os.environ.get("STABLEWM_HOME", "")) / "datasets/pusht_expert_train.h5"
HORIZON = 5
BLOCK = 5
HISTORY = 1  # scripts/plan/config/pusht.yaml leaves history_len at default 1
EVAL_SEED = 42


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def image_transform():
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=224),
    ])


def encode_images(model, images: np.ndarray, transform, device):
    # The official policy transposes NHWC to NCHW before ToImage.
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(images), 8):
            x = torch.stack([
                transform(torch.from_numpy(frame.transpose(2, 0, 1)))
                for frame in images[start : start + 8]
            ]).to(device)
            z = model.encode({"pixels": x[:, None]})["emb"][:, 0]
            chunks.append(z.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def evaluation_rows(offsets: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    valid = np.concatenate([
        np.arange(int(off), int(off) + max(0, int(length) - 25), dtype=np.int64)
        for off, length in zip(offsets, lengths)
        if length > 25
    ])
    rng = np.random.default_rng(EVAL_SEED)
    return np.sort(rng.choice(valid, size=50, replace=False))


def raw_actions_to_model(actions: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(actions.reshape(-1, 2)).reshape(*actions.shape[:-1], -1).astype(np.float32)


def make_pair(h5, ep: int, t: int, scaler, rng, env):
    off = int(h5["ep_offset"][ep])
    state0 = np.asarray(h5["state"][off + t - 10], dtype=np.float64)
    goal_state = np.asarray(h5["state"][off + t + 25], dtype=np.float64)
    prefix = np.asarray(h5["action"][off + t - 10 : off + t], dtype=np.float32)
    base = np.asarray(h5["action"][off + t : off + t + 25], dtype=np.float32)
    base_norm = raw_actions_to_model(base, scaler)
    epsilon = float(rng.choice([0.25, 0.5, 0.75]))
    direction = rng.normal(size=base_norm.shape).astype(np.float32)
    direction /= np.sqrt(np.mean(direction**2)) + 1e-8
    pair_norm = np.stack([base_norm + epsilon * direction, base_norm - epsilon * direction])
    pair_raw = np.clip(scaler.inverse_transform(pair_norm.reshape(-1, 2)), -1.0, 1.0)
    pair_raw = pair_raw.reshape(2, 25, 2).astype(np.float32)
    pair_norm = raw_actions_to_model(pair_raw, scaler).reshape(2, 5, 10)

    context = None
    outcomes = []
    branch_states = []
    for branch in range(2):
        env.reset(options={"state": state0, "goal_state": goal_state})
        frames = []
        for k, action in enumerate(prefix):
            env.step(action)
            if k == 9:
                frames.append(env.render())
        current_state = env.unwrapped._get_obs().copy()
        branch_states.append(current_state)
        if context is None:
            context = np.stack(frames)
        endpoints = []
        for k, action in enumerate(pair_raw[branch]):
            env.step(action)
            if (k + 1) % BLOCK == 0:
                endpoints.append(env.render())
        outcomes.append(np.stack(endpoints))
    # Restoring from the same state and replaying the same history must be exact
    # before training on a finite-difference target.
    if not np.allclose(branch_states[0], branch_states[1], atol=1e-5, rtol=0):
        raise RuntimeError(f"non-deterministic branch root: {branch_states}")
    return context[-1:], np.stack(outcomes), np.empty((0, 10), np.float32), pair_norm, epsilon


def collect(args, model, scaler, h5, eval_rows, transform, device):
    rng = np.random.default_rng(args.seed)
    offsets = h5["ep_offset"][:]
    lengths = h5["ep_len"][:]
    eval_eps = set(np.searchsorted(offsets, eval_rows, side="right") - 1)
    eligible = np.array([
        ep for ep, length in enumerate(lengths)
        if length >= 36 and ep not in eval_eps
    ], dtype=np.int64)
    env = gym.make("swm/PushT-v1", render_mode="rgb_array", resolution=224)
    rows = []
    for i in range(args.pairs):
        ep = int(rng.choice(eligible))
        t = int(rng.integers(10, int(lengths[ep]) - 25))
        context, future, past, actions, eps = make_pair(h5, ep, t, scaler, rng, env)
        ctx_z = encode_images(model, context, transform, device)
        fut_z = encode_images(model, future.reshape(-1, *future.shape[-3:]), transform, device)
        rows.append((ctx_z, fut_z.reshape(2, 5, -1), past, actions, ep, t, eps))
        if (i + 1) % 16 == 0:
            print(f"collected {i + 1}/{args.pairs}", flush=True)
    env.close()
    path = args.run_dir / "branches.npz"
    np.savez_compressed(
        path,
        context=np.stack([r[0] for r in rows]).astype(np.float32),
        future=np.stack([r[1] for r in rows]).astype(np.float32),
        history_action=np.stack([r[2] for r in rows]).astype(np.float32),
        action=np.stack([r[3] for r in rows]).astype(np.float32),
        episode=np.array([r[4] for r in rows]),
        step=np.array([r[5] for r in rows]),
        epsilon=np.array([r[6] for r in rows]),
    )
    return path


def model_prediction(model, batch):
    b = batch["context"].shape[0]
    info = {
        "pixels": torch.empty(b, 2, HISTORY, 1, device=batch["context"].device),
        "emb": batch["context"][:, None].expand(-1, 2, -1, -1),
        "action_history": batch["history_action"][:, None].expand(-1, 2, -1, -1),
    }
    pred = model.rollout(info, batch["action"])["predicted_emb"]
    return pred[:, :, HISTORY:]


def losses(model, batch, lam: float, response_target: str):
    pred = model_prediction(model, batch)[:, :, [0, 2, 4]]
    target = batch["future"][:, :, [0, 2, 4]]
    prediction = (pred - target).square().mean()
    response_delta = (pred[:, 0] - pred[:, 1]) - (target[:, 0] - target[:, 1])
    if response_target == "endpoint":
        response_delta = response_delta[:, -1:]
    response = response_delta.square().mean()
    return prediction + lam * response, prediction, response


def train_arm(args, original, data, arm: str, device):
    import copy

    model = copy.deepcopy(original).to(device)
    model.encoder.eval().requires_grad_(False)
    model.projector.eval().requires_grad_(False)
    model.predictor.requires_grad_(True)
    model.action_encoder.requires_grad_(True)
    model.pred_proj.requires_grad_(True)
    train_params = [
        p for part in (model.predictor, model.action_encoder, model.pred_proj)
        for p in part.parameters() if p.requires_grad
    ]
    optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed + 100)
    n = len(data["context"])
    split = max(1, int(n * 0.85))
    lam = 0.0 if arm == "prediction" else 1.0
    log = []
    dev = {k: torch.as_tensor(data[k][split:], device=device) for k in
           ("context", "future", "history_action", "action")}
    best_score = float("inf")
    best_step = -1
    best_state = None

    def check_dev(step, pred=None, response=None):
        nonlocal best_score, best_step, best_state
        model.eval()
        with torch.inference_mode():
            _, dev_pred, dev_response = losses(model, dev, 0.0, args.response_target)
        score = float(dev_pred + lam * dev_response)
        row = {"arm": arm, "step": step,
               "train_prediction": None if pred is None else float(pred),
               "train_response": None if response is None else float(response),
               "dev_prediction": float(dev_pred), "dev_response": float(dev_response),
               "selection_score": score}
        log.append(row)
        print(json.dumps(row), flush=True)
        if score < best_score:
            best_score = score
            best_step = step
            best_state = {name: param.detach().cpu().clone()
                          for name, param in model.named_parameters() if param.requires_grad}

    check_dev(0)
    for step in range(args.updates):
        idx = rng.integers(0, split, size=min(16, split))
        batch = {k: torch.as_tensor(data[k][idx], device=device) for k in
                 ("context", "future", "history_action", "action")}
        model.predictor.train()
        model.action_encoder.train()
        # The checkpoint's pred_proj MLP contains BatchNorm1d. Rollout training
        # repeatedly calls it on a small, branch-correlated batch; updating
        # its running statistics corrupts evaluation even when dev weights
        # improve. Keep its statistics fixed while training its affine/linear
        # parameters through the usual gradients.
        model.pred_proj.eval()
        optimizer.zero_grad(set_to_none=True)
        total, pred, response = losses(model, batch, lam, args.response_target)
        if not torch.isfinite(total):
            raise RuntimeError(f"nonfinite {arm} loss at step {step}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        optimizer.step()
        if (step + 1) % 20 == 0 or step + 1 == args.updates:
            check_dev(step + 1, pred, response)
    assert best_state is not None
    model.load_state_dict(best_state, strict=False)
    out = args.run_dir / f"{arm}.pt"
    torch.save(model.cpu().state_dict(), out)
    write_json(args.run_dir / f"{arm}_train.json", {"lambda": lam,
               "best_step": best_step, "best_score": best_score, "history": log})
    del model
    torch.cuda.empty_cache()
    return out


def evaluate(args, scaler, eval_rows, offsets, device):
    dataset = swm.data.load_dataset(
        "pusht_expert_train.h5", keys_to_cache=["action", "proprio", "state"]
    )
    process = {"action": scaler}
    for key in ("proprio", "state"):
        sc = StandardScaler().fit(dataset.get_col_data(key))
        process[key] = sc
        process[f"goal_{key}"] = sc
    transform = {"pixels": image_transform(), "goal": image_transform()}
    callables = [
        {"method": "_set_state", "args": {"state": {"value": "state"}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
    ]
    summary = {}
    for arm in ("prediction", "response"):
        model = swm.wm.utils.load_pretrained(OFFICIAL)
        model.load_state_dict(torch.load(args.run_dir / f"{arm}.pt", map_location="cpu", weights_only=True), strict=True)
        model = model.to(device).eval().requires_grad_(False)
        model.interpolate_pos_encoding = True
        cost = swm.planning.ShootingCostEvaluator(model, swm.planning.GoalMSE())
        config = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)
        world = swm.World("swm/PushT-v1", num_envs=1, max_episode_steps=100, image_shape=(224, 224))
        rows = []
        try:
            for j, row in enumerate(eval_rows[:args.eval_roots]):
                out = args.run_dir / f"eval_{arm}_{j:02d}.json"
                if out.exists():
                    rows.append(json.loads(out.read_text()))
                    continue
                ep = int(np.searchsorted(offsets, row, side="right") - 1)
                step = int(row - offsets[ep])
                solver = swm.planning.solver.CEMSolver(
                    cost, batch_size=1, num_samples=300, n_steps=30, topk=30,
                    var_scale=1.0, device=device, seed=EVAL_SEED + j
                )
                policy = swm.policy.WorldModelPolicy(
                    solver=solver, config=config, process=process, transform=transform
                )
                world.set_policy(policy)
                start = time.monotonic()
                metrics = world.evaluate(
                    dataset=dataset, episodes_idx=[ep], start_steps=[step],
                    goal_offset=25, eval_budget=50, callables=callables,
                )
                raw_env = world.envs.envs[0].unwrapped
                final_state = raw_env._get_obs()
                goal_state = np.asarray(raw_env.goal_state, dtype=np.float64)
                _, native_distance = raw_env.eval_state(goal_state, final_state)
                position_error = float(np.linalg.norm(goal_state[:4] - final_state[:4]))
                block_shape = raw_env.shapes[int(raw_env.variation_space["block"]["shape"].value)]
                symmetry = raw_env.shape_symmetry_angles.get(block_shape, 2 * np.pi)
                angle_error = abs(goal_state[4] - final_state[4]) % symmetry
                angle_error = float(min(angle_error, symmetry - angle_error))
                item = {"arm": arm, "root": j, "row": int(row), "episode": ep,
                        "step": step, "success": bool(metrics["episode_successes"][0]),
                        "native_final_state_distance": float(native_distance),
                        "position_error": position_error,
                        "symmetry_aware_angle_error_rad": angle_error,
                        "seconds": time.monotonic() - start}
                write_json(out, item)
                rows.append(item)
                print(json.dumps(item), flush=True)
        finally:
            world.close()
        summary[arm] = {"successes": sum(int(x["success"]) for x in rows),
                        "count": len(rows),
                        "mean_native_final_state_distance": float(np.mean([
                            x["native_final_state_distance"] for x in rows
                        ])),
                        "mean_position_error": float(np.mean([
                            x["position_error"] for x in rows
                        ])),
                        "mean_symmetry_aware_angle_error_rad": float(np.mean([
                            x["symmetry_aware_angle_error_rad"] for x in rows
                        ])), "results": rows}
        del model
        torch.cuda.empty_cache()
    write_json(args.run_dir / "evaluation.json", summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=96)
    parser.add_argument("--updates", type=int, default=160)
    parser.add_argument("--eval-roots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=27)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--branch-cache", type=Path)
    parser.add_argument("--response-target", choices=("all3", "endpoint"), default="all3")
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("physics, model loading, and training require an sbatch compute node")
    if args.pairs < 16 or args.eval_roots < 1 or args.eval_roots > 50:
        raise ValueError("pairs must be >=16 and eval-roots between 1 and 50")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(__file__, args.run_dir / "pipeline.py")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    original = swm.wm.utils.load_pretrained(OFFICIAL).to(device).eval().requires_grad_(False)
    original.interpolate_pos_encoding = True
    ckpt_dir = Path(os.environ["STABLEWM_HOME"]) / "checkpoints/models--quentinll--lewm-pusht"
    with h5py.File(DATA, "r", swmr=True) as h5:
        offsets = h5["ep_offset"][:]
        lengths = h5["ep_len"][:]
        eval_rows = evaluation_rows(offsets, lengths)
        scaler = StandardScaler().fit(h5["action"][:])
        write_json(args.run_dir / "provenance.json", {
            "checkpoint": str(ckpt_dir / "weights.pt"),
            "checkpoint_sha256": sha256(ckpt_dir / "weights.pt"),
            "config_sha256": sha256(ckpt_dir / "config.json"),
            "dataset": str(DATA), "data_rows": int(sum(lengths)),
            "action_mean": scaler.mean_.tolist(), "action_scale": scaler.scale_.tolist(),
            "eval_rows": eval_rows.tolist(), "eval_seed": EVAL_SEED,
            "pairs": args.pairs, "updates": args.updates, "seed": args.seed,
            "lr": args.lr, "checkpoint_selection": "lowest held-out arm objective, including step zero",
            "response_target": args.response_target,
            "branch_cache_source": str(args.branch_cache) if args.branch_cache else None,
            "eval_roots": args.eval_roots,
            "planner": {"horizon": 5, "action_block": 5, "receding_horizon": 5,
                        "history_len": 1,
                        "samples": 300, "iterations": 30, "elites": 30},
        })
        if args.branch_cache:
            pair_file = args.run_dir / "branches.npz"
            shutil.copy2(args.branch_cache, pair_file)
            with np.load(pair_file) as cached:
                if len(cached["context"]) != args.pairs:
                    raise ValueError("Branch cache size does not match --pairs")
        else:
            pair_file = collect(args, original, scaler, h5, eval_rows, image_transform(), device)
        print(f"branch cache: {pair_file}", flush=True)
    with np.load(pair_file) as cached:
        data = {k: cached[k] for k in cached.files}
    for arm in ("prediction", "response"):
        train_arm(args, original, data, arm, device)
    evaluate(args, scaler, eval_rows, offsets, device)


if __name__ == "__main__":
    main()
