#!/usr/bin/env python3
"""Broaden same-state action branches, then compare matched response losses."""

import argparse
import copy
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

import pipeline as exp
import response_followup as response


def signal(data):
    endpoints = data["endpoint"]
    centered = endpoints - endpoints.mean(axis=1, keepdims=True)
    physical = data["physical_distance_m"]
    return {
        "mean_endpoint_latent_pair_mse": float(2 * np.mean(centered ** 2)),
        "median_physical_distance_spread_m": float(np.median(
            physical.max(axis=1) - physical.min(axis=1))),
        "banks_with_successful_candidate": int(data["success"].any(axis=1).sum()),
        "banks": int(len(endpoints)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--collect", type=int, default=160)
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--eval-roots", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--response-weight", type=float, default=1.0)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("collection, training and control require sbatch")
    if args.collect < 2 or not 1 <= args.eval_roots <= 50:
        raise ValueError("invalid collection/evaluation size")

    import stable_worldmodel as swm
    corrected = exp.load_module(exp.DIAG / "76_ogb_true_endpoint_corrected.py", "cube_corrected_mixed")
    audit = corrected.load_stage0_module()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = swm.data.load_dataset(exp.DATASET, keys_to_cache=["action"])
    training_roots, eval_roots = exp.sample_manifests(dataset, audit, args.collect, args.seed)
    action_data = dataset.get_col_data("action")
    scaler = StandardScaler().fit(action_data[~np.isnan(action_data).any(axis=1)])
    model = swm.wm.utils.load_pretrained(exp.OFFICIAL).cuda().eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    checkpoint = Path(os.environ["STABLEWM_HOME"]) / "checkpoints/models--quentinll--lewm-cube/weights.pt"
    exp.dump(args.out_dir / "provenance.json", {
        "parent_run": str(args.parent_run), "checkpoint_sha256": exp.file_sha256(checkpoint),
        "dataset": exp.DATASET, "collect": args.collect,
        "train_manifest": [asdict(x) for x in training_roots],
        "eval_manifest": [asdict(x) for x in eval_roots],
        "population": "mixed", "initial_ranks": exp.MIXED_INITIAL_RANKS,
        "final_ranks": exp.MIXED_FINAL_RANKS,
        "planner": {"horizon": exp.HORIZON, "action_block": exp.BLOCK,
                    "cem_samples": 300, "cem_iterations": 30, "cem_elites": 30,
                    "eval_budget": 50},
        "training": {"updates": args.updates, "lr": args.lr,
                     "response_weight": args.response_weight, "seed": args.seed},
        "supervision": "same-state endpoint image embeddings; physical distances are diagnostics only",
    })
    args.population = "mixed"
    cache = exp.collect(args, swm, audit, corrected, dataset, model, scaler,
                        audit.make_transform(224), training_roots, args.out_dir)
    with np.load(cache) as archive:
        data = {key: archive[key] for key in archive.files}
    with np.load(args.parent_run / "cem_branches.npz") as archive:
        parent = {key: archive[key] for key in archive.files}
    exp.dump(args.out_dir / "bank_signal.json", {
        "final_population_parent": signal(parent),
        "mixed_population": signal(data),
    })

    results = {}
    for name, weight in (("prediction_mixed", 0.0),
                         ("response_mixed", args.response_weight)):
        arm_args = copy.copy(args)
        arm_args.response_weight = weight
        arm_args.out_dir = args.out_dir / name
        arm_args.out_dir.mkdir()
        # Same initialization, batches, dropout stream, optimizer and planner.
        torch.manual_seed(args.seed + 777)
        trained = response.train(arm_args, model, data, torch.device("cuda"))
        response.evaluate(arm_args, swm, audit, corrected, dataset, trained,
                          scaler, audit.make_transform(224), eval_roots)
        arm_result = json.loads((arm_args.out_dir / "evaluation.json").read_text())
        results[name] = arm_result["response_native"]
        results[name]["results"] = [dict(row, arm=name) for row in results[name]["results"]]
        del trained
        torch.cuda.empty_cache()
    exp.dump(args.out_dir / "evaluation.json", results)


if __name__ == "__main__":
    main()
