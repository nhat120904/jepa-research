#!/usr/bin/env python3
"""Independent non-contact Cube starts with exact state restoration."""

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


def hard_roots(dataset, audit, corrected, swm, excluded, count, seed):
    import mujoco

    candidates = audit.build_manifest(dataset, 2000, exp.GOAL_OFFSET, seed)
    world, raw_env, _, _ = corrected.make_world(swm, candidates[0])
    joint = mujoco.mj_name2id(raw_env._model, mujoco.mjtObj.mjOBJ_JOINT, "object_joint_0")
    adr = int(raw_env._model.jnt_qposadr[joint])
    rng = np.random.default_rng(seed + 1)
    rng.shuffle(candidates)
    selected = []
    used = set()
    try:
        for snapshot in candidates:
            if snapshot.episode in excluded or snapshot.episode in used:
                continue
            init, goal = exp.dataset_rows(dataset, swm, snapshot)
            exp.reset_exact(raw_env, snapshot, init, goal, corrected, audit)
            target = audit.goal_field(goal, "block_0_pos")
            base_cube = np.asarray(raw_env._data.joint("object_joint_0").qpos[:3]).copy()
            hand = np.asarray(raw_env._data.site_xpos[raw_env._pinch_site_id]).copy()
            if np.linalg.norm(base_cube - target) < 0.05 or np.linalg.norm(hand - base_cube) < 0.02:
                continue
            angle = rng.uniform(0, 2 * np.pi)
            delta = 0.02 * np.array([np.cos(angle), np.sin(angle)])
            qpos = np.asarray(init["qpos"]).copy()
            qpos[adr] = np.clip(qpos[adr] + delta[0], 0.30, 0.55)
            qpos[adr + 1] = np.clip(qpos[adr + 1] + delta[1], -0.30, 0.30)
            if np.linalg.norm(qpos[adr : adr + 2] - np.asarray(init["qpos"])[adr : adr + 2]) < 0.015:
                continue
            corrected.restore_complete(raw_env, qpos, init["qvel"], goal, audit)
            cube = np.asarray(raw_env._data.joint("object_joint_0").qpos[:3]).copy()
            hand = np.asarray(raw_env._data.site_xpos[raw_env._pinch_site_id]).copy()
            start_distance = float(np.linalg.norm(cube - target))
            if start_distance < 0.05 or np.linalg.norm(hand - cube) < 0.02:
                continue
            selected.append({
                "episode": snapshot.episode, "start_step": snapshot.start_step,
                "reset_seed": snapshot.reset_seed, "qpos": qpos.tolist(),
                "start_distance_m": start_distance,
                "cube_perturbation_xy_m":
                    (qpos[adr : adr + 2] - np.asarray(init["qpos"])[adr : adr + 2]).tolist(),
            })
            used.add(snapshot.episode)
            if len(selected) == count:
                break
    finally:
        world.close()
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)}/{count} non-contact hard starts found")
    return selected


def evaluate(args, swm, audit, corrected, dataset, original, scorer, scaler, transform, roots):
    arms = ("original_native", "original_scorer", "prediction_native",
            "prediction_scorer", "rank_scorer")
    report = {}
    first = audit.Snapshot(0, roots[0]["episode"], roots[0]["start_step"], 0,
                           roots[0]["reset_seed"])
    for arm in arms:
        model = copy.deepcopy(original).cuda()
        if arm.startswith("prediction"):
            model.load_state_dict(torch.load(args.parent_run / "prediction.pt", map_location="cpu", weights_only=True))
        elif arm == "rank_scorer":
            model.load_state_dict(torch.load(args.parent_run / "rank.pt", map_location="cpu", weights_only=True))
        model.eval().requires_grad_(False)
        objective = (swm.planning.GoalMSE() if arm.endswith("native")
                     else exp.ScorerObjective(scorer))
        world, raw_env, _, _ = corrected.make_world(swm, first)
        dim = int(np.prod(world.envs.single_action_space.shape))
        rows = []
        try:
            for i, entry in enumerate(roots):
                path = args.out_dir / f"eval_{arm}_{i:02d}.json"
                if path.exists():
                    rows.append(json.loads(path.read_text()))
                    continue
                snapshot = audit.Snapshot(i, entry["episode"], entry["start_step"], 0,
                                          entry["reset_seed"])
                init, goal = exp.dataset_rows(dataset, swm, snapshot)
                raw_env.reset(seed=snapshot.reset_seed, options={"variation": []})
                import mujoco
                raw_env._model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
                corrected.restore_complete(raw_env, np.asarray(entry["qpos"]),
                                           init["qvel"], goal, audit)
                frame = audit.resize_render(raw_env.render())
                target = audit.goal_field(goal, "block_0_pos")
                goal_image = np.asarray(goal["goal"])
                policy = exp.make_solver(swm, model, objective, world, scaler,
                                         transform, exp.EVAL_SEED + i)
                start = time.monotonic()
                passed = False
                steps = 0
                for step in range(50):
                    info = {"pixels": frame[None, None],
                            "goal": goal_image[None, None],
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
                row = {"arm": arm, "root": i, "episode": snapshot.episode,
                       "start_step": snapshot.start_step, "success": passed,
                       "final_cube_distance_m": distance, "steps": steps,
                       "start_distance_m": entry["start_distance_m"],
                       "seconds": time.monotonic() - start}
                exp.dump(path, row)
                rows.append(row)
                print(json.dumps(row), flush=True)
        finally:
            world.close()
        report[arm] = {"successes": sum(int(x["success"]) for x in rows),
                       "count": len(rows),
                       "mean_final_cube_distance_m": float(np.mean([
                           x["final_cube_distance_m"] for x in rows])),
                       "results": rows}
        del model
        torch.cuda.empty_cache()
    exp.dump(args.out_dir / "evaluation.json", report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("physics and model evaluation require sbatch")
    import stable_worldmodel as swm
    corrected = exp.load_module(exp.DIAG / "76_ogb_true_endpoint_corrected.py", "cube_corrected_hard")
    audit = corrected.load_stage0_module()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = swm.data.load_dataset(exp.DATASET, keys_to_cache=["action"])
    provenance = json.loads((args.parent_run / "provenance.json").read_text())
    excluded = {x["episode"] for x in provenance["train_manifest"]}
    path = args.out_dir / "hard_roots.json"
    if path.exists():
        roots = json.loads(path.read_text())
    else:
        roots = hard_roots(data, audit, corrected, swm, excluded, args.roots, args.seed)
        path.write_text(json.dumps(roots, indent=2) + "\n")
    action_data = data.get_col_data("action")
    scaler = StandardScaler().fit(action_data[~np.isnan(action_data).any(axis=1)])
    original = swm.wm.utils.load_pretrained(exp.OFFICIAL).cuda().eval().requires_grad_(False)
    original.interpolate_pos_encoding = True
    scorer_state = torch.load(args.parent_run / "scorer.pt", map_location="cpu", weights_only=True)
    dim = scorer_state["net.0.weight"].numel() // 4
    scorer = exp.GoalProgressScorer(dim).cuda()
    scorer.load_state_dict(scorer_state)
    scorer.eval().requires_grad_(False)
    evaluate(args, swm, audit, corrected, data, original, scorer, scaler,
             audit.make_transform(224), roots)


if __name__ == "__main__":
    main()
