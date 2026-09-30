#!/usr/bin/env python3
"""Same-state action-response fine-tuning with LeWM's unchanged planner."""

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import pipeline as exp


def losses(model, batch, response_weight):
    predicted = exp.predicted_endpoints(model, batch)
    residual = predicted - batch["endpoint"]
    prediction = residual.square().mean()
    # E_{i,j} ||(pred_i-pred_j)-(true_i-true_j)||², including zero diagonal.
    # Centering computes this exactly without materializing a K x K tensor.
    action_response = 2 * (residual - residual.mean(dim=1, keepdim=True)).square().mean()
    return prediction + response_weight * action_response, prediction, action_response


def train(args, original, data, device):
    model = copy.deepcopy(original).to(device)
    model.encoder.eval().requires_grad_(False)
    model.projector.eval().requires_grad_(False)
    model.predictor.requires_grad_(True)
    model.action_encoder.requires_grad_(True)
    model.pred_proj.requires_grad_(True)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-4)
    split = max(1, int(len(data["context"]) * 0.8))
    dev = exp.load_batch(data, np.arange(split, len(data["context"])), device)
    rng = np.random.default_rng(args.seed + 2200)
    best, best_step, best_state = float("inf"), -1, None
    history = []
    for step in range(args.updates + 1):
        if step % 20 == 0 or step == args.updates:
            model.eval()
            with torch.inference_mode():
                total, prediction, response = losses(model, dev, args.response_weight)
            score = float(total)
            history.append({"step": step, "dev_objective": score,
                            "dev_prediction": float(prediction),
                            "dev_action_response": float(response)})
            if score < best:
                best, best_step = score, step
                best_state = {name: p.detach().cpu().clone() for name, p in
                              model.named_parameters() if p.requires_grad}
        if step == args.updates:
            break
        model.predictor.train()
        model.action_encoder.train()
        model.pred_proj.eval()  # preserve released BatchNorm statistics
        batch = exp.load_batch(data, rng.integers(0, split, size=min(16, split)), device)
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = losses(model, batch, args.response_weight)
        if not torch.isfinite(total):
            raise RuntimeError(f"nonfinite response loss at update {step}")
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
    model.load_state_dict(best_state, strict=False)
    model.eval().requires_grad_(False)
    torch.save(model.cpu().state_dict(), args.out_dir / "response_native.pt")
    exp.dump(args.out_dir / "train.json", {"best_step": best_step,
                                          "best_dev_objective": best,
                                          "response_weight": args.response_weight,
                                          "history": history})
    print(f"ACTION_RESPONSE best={best_step} dev={best:.5f}", flush=True)
    return model.to(device)


def evaluate(args, swm, audit, corrected, dataset, model, scaler, transform, roots):
    world, raw_env, _, _ = corrected.make_world(swm, roots[0])
    dim = int(np.prod(world.envs.single_action_space.shape))
    rows = []
    try:
        for i, snapshot in enumerate(roots[:args.eval_roots]):
            path = args.out_dir / f"eval_response_native_{i:02d}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
                continue
            init, goal = exp.dataset_rows(dataset, swm, snapshot)
            frame = exp.reset_exact(raw_env, snapshot, init, goal, corrected, audit)
            target = audit.goal_field(goal, "block_0_pos")
            goal_image = np.asarray(goal["goal"])
            policy = exp.make_solver(swm, model, swm.planning.GoalMSE(), world,
                                     scaler, transform, exp.EVAL_SEED + i)
            start = time.monotonic()
            passed = False
            steps = 0
            for step in range(50):
                info = {"pixels": frame[None, None], "goal": goal_image[None, None],
                        "action": np.full((1, 1, dim), np.nan, np.float32),
                        "terminated": np.zeros(1, bool)}
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
            row = {"arm": "response_native", "root": i, "episode": snapshot.episode,
                   "start_step": snapshot.start_step, "success": passed,
                   "final_cube_distance_m": distance, "steps": steps,
                   "seconds": time.monotonic() - start}
            exp.dump(path, row)
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        world.close()
    reference = json.loads((args.parent_run / "evaluation.json").read_text())["prediction_native"]
    if len(reference["results"]) < args.eval_roots:
        raise RuntimeError("parent prediction-native evaluation is incomplete")
    reference_rows = reference["results"][:args.eval_roots]
    exp.dump(args.out_dir / "evaluation.json", {
        "prediction_native": {"successes": sum(int(x["success"]) for x in reference_rows),
                              "count": len(reference_rows),
                              "mean_final_cube_distance_m": float(np.mean([
                                  x["final_cube_distance_m"] for x in reference_rows])),
                              "results": reference_rows},
        "response_native": {"successes": sum(int(x["success"]) for x in rows),
                            "count": len(rows),
                            "mean_final_cube_distance_m": float(np.mean([
                                x["final_cube_distance_m"] for x in rows])),
                            "results": rows},
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--eval-roots", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--response-weight", type=float, default=1.0)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("training and physics require sbatch")
    if not 1 <= args.eval_roots <= 50 or args.updates < 0:
        raise ValueError("invalid evaluation roots or updates")
    import stable_worldmodel as swm
    corrected = exp.load_module(exp.DIAG / "76_ogb_true_endpoint_corrected.py", "cube_corrected_response")
    audit = corrected.load_stage0_module()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    with np.load(args.parent_run / "cem_branches.npz") as archive:
        data = {k: archive[k] for k in archive.files}
    dataset = swm.data.load_dataset(exp.DATASET, keys_to_cache=["action"])
    provenance = json.loads((args.parent_run / "provenance.json").read_text())
    roots = [audit.Snapshot(**entry) for entry in provenance["eval_manifest"]]
    action_data = dataset.get_col_data("action")
    scaler = StandardScaler().fit(action_data[~np.isnan(action_data).any(axis=1)])
    original = swm.wm.utils.load_pretrained(exp.OFFICIAL).cuda().eval().requires_grad_(False)
    original.interpolate_pos_encoding = True
    exp.dump(args.out_dir / "provenance.json", {
        "parent_run": str(args.parent_run), "parent_job": 55413,
        "supervision": "same-state endpoint image embeddings, no physical task labels",
        "deployment_cost": "released LeWM GoalMSE",
        "updates": args.updates, "lr": args.lr,
        "response_weight": args.response_weight, "seed": args.seed,
        "eval_roots": args.eval_roots})
    model = train(args, original, data, torch.device("cuda"))
    evaluate(args, swm, audit, corrected, dataset, model, scaler,
             audit.make_transform(224), roots)


if __name__ == "__main__":
    main()
