"""Bounded CPU qualification and RGB/action collection. Run only via Slurm."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

from .runtime import require_slurm


def write_json(path, data):
    # Runs write only into a fresh per-job output directory, never source/config.
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def physical_answers(states, goal_a, goal_b, radius):
    import numpy as np
    from .queries import ordered_score

    near_a = np.linalg.norm(states - goal_a, axis=-1) < radius
    near_b = np.linalg.norm(states - goal_b, axis=-1) < radius
    return {"reach_b": bool(near_b.any()),
            "a_then_b": bool(ordered_score(near_a, near_b))}


def sample_task(env, rng):
    import numpy as np

    # Selection depends on initial/goal geometry, NEVER candidate outcome.
    points = []
    for _ in range(10000):
        p = env.sample_position(rng)
        if all(np.linalg.norm(p - old) >= 9 for old in points):
            points.append(p)
        if len(points) == 3:
            return points
    raise RuntimeError("Could not sample separated start/A/B")


def screen(config, run_dir, upstream_root):
    import numpy as np
    from .proposals import smooth_bank, visual_goal_bank
    from .queries import answer, chroma_features
    from .wall_adapter import WallRGB

    rows = []
    prefix_index = 0
    for layout in config["layouts"]:
        env = WallRGB(upstream_root, *layout, seed=config["seed"])
        for _ in range(config["prefixes_per_layout"]):
            prefix_seed = config["seed"] + prefix_index * 103
            task_rng = np.random.default_rng(prefix_seed)
            start, goal_a, goal_b = sample_task(env, task_rng)
            initial_rgb = env.reset(start)
            snapshot = env.snapshot()
            anchors = np.stack([env.goal_image(goal_a), env.goal_image(goal_b)])
            af, bf = chroma_features(anchors)
            proposal_rng = np.random.default_rng(prefix_seed + 1)
            if config.get("proposal", "smooth_bank") == "visual_goal_bank":
                bank = visual_goal_bank(proposal_rng, initial_rgb, anchors[0], anchors[1],
                                        config["candidates"], config["horizon"],
                                        config["knots"], config["action_max_norm"])
            elif config.get("proposal", "smooth_bank") == "smooth_bank":
                bank = smooth_bank(proposal_rng, config["candidates"], config["horizon"],
                                   config["knots"], config["action_max_norm"])
            else:
                raise ValueError(f"Unknown proposal: {config['proposal']}")
            image_scores = {key: [] for key in ("reach_b", "a_then_b", "reach_a", "b_then_a")}
            outcomes = {key: [] for key in ("reach_b", "a_then_b")}
            saved_images, saved_states = [], []
            save_bank = prefix_index < config["save_example_banks"]
            first_rollout = None
            for candidate in bank:
                images, states = env.rollout(snapshot, candidate)
                if first_rollout is None:
                    first_rollout = (images, states)
                elif not np.array_equal(images[0], initial_rgb):
                    raise AssertionError("Branch initial observation changed")
                query = answer(chroma_features(images[1:]), af, bf)
                truth = physical_answers(states[1:], goal_a, goal_b, config["eval_radius"])
                for key in image_scores:
                    image_scores[key].append(float(query[key]))
                for key in outcomes:
                    outcomes[key].append(truth[key])
                if save_bank:
                    saved_images.append(images)
                    saved_states.append(states)
            replay_images, replay_states = env.rollout(snapshot, bank[0])
            if not (np.array_equal(replay_images, first_rollout[0]) and
                    np.array_equal(replay_states, first_rollout[1])):
                raise AssertionError("Branch replay mismatch")
            row = {"prefix_id": prefix_index, "seed": prefix_seed, "layout": layout,
                   "start": start.tolist(), "goal_a": goal_a.tolist(),
                   "goal_b": goal_b.tolist(), "exact_replay": True, "tasks": {}}
            sequential_index = int(np.argmax(image_scores["reach_a"]))
            for task in outcomes:
                score = np.asarray(image_scores[task])
                truth = np.asarray(outcomes[task], dtype=bool)
                selected = int(np.argmax(score))
                true_best_score = float(score[truth].max()) if truth.any() else None
                row["tasks"][task] = {
                    "default_hold_success": bool(truth[0]),
                    "random_bank_success": float(truth.mean()),
                    "oracle_bank_success": bool(truth.any()),
                    "image_selected_index": selected,
                    "image_selected_success": bool(truth[selected]),
                    "sequential_surrogate_success": bool(truth[sequential_index]),
                    "physical_selection_regret": int(truth.any()) - int(truth[selected]),
                    "score_std": float(score.std()), "score_max": float(score.max()),
                    "best_successful_image_score": true_best_score,
                    "success_count": int(truth.sum()), "candidates": len(bank),
                }
            # Native-order asymmetry, not a method win or a new query operator claim.
            row["mean_order_score_difference"] = float(np.mean(np.abs(
                np.asarray(image_scores["a_then_b"]) - np.asarray(image_scores["b_then_a"]))))
            rows.append(row)
            # Write progress incrementally so timeout/failure preserves completed prefixes.
            with (run_dir / "prefixes.jsonl").open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            if save_bank:
                # State arrays are evaluation-only and stored separately from RGB/actions.
                np.savez_compressed(run_dir / f"bank_{prefix_index:03d}_observations.npz",
                                    rgb=np.stack(saved_images), actions=bank, anchors=anchors)
                np.savez_compressed(run_dir / f"bank_{prefix_index:03d}_privileged_eval.npz",
                                    states=np.stack(saved_states), goal_a=goal_a, goal_b=goal_b)
            prefix_index += 1
            print(f"prefix {prefix_index}/{len(config['layouts']) * config['prefixes_per_layout']} "
                  f"reach={row['tasks']['reach_b']['success_count']}/{len(bank)} "
                  f"order={row['tasks']['a_then_b']['success_count']}/{len(bank)}", flush=True)
    summary = {}
    for task in ("reach_b", "a_then_b"):
        records = [r["tasks"][task] for r in rows]
        summary[task] = {key: float(np.mean([r[key] for r in records])) for key in (
            "random_bank_success", "oracle_bank_success", "image_selected_success",
            "sequential_surrogate_success", "physical_selection_regret", "score_std")}
        summary[task]["informative_prefixes"] = sum(0 < r["success_count"] < r["candidates"]
                                                     for r in records)
    write_json(run_dir / "screen_summary.json", {"prefixes": len(rows), "tasks": summary,
               "status": "DEVELOPMENT_SCREEN_NOT_METHOD_RESULT",
               "kernel": "fixed_spatial_chroma_cosine", "learned_models": False})
    return summary


def collect(config, run_dir, upstream_root):
    import numpy as np
    from .proposals import smooth_bank
    from .wall_adapter import WallRGB

    dataset_dir = run_dir / "dataset"
    dataset_dir.mkdir()
    records = []
    # Explicit disjoint seed ranges; selection fixed BEFORE observing trajectories.
    for split, offset in (("train", 100000), ("val", 200000), ("test", 300000)):
        split_dir = dataset_dir / split
        split_dir.mkdir()
        for index in range(config[f"{split}_episodes"]):
            seed = config["seed"] + offset + index
            rng = np.random.default_rng(seed)
            env = WallRGB(upstream_root, *config["dataset_layout"], seed=seed)
            env.reset(env.sample_position(rng))
            # Entire action sequence sampled before rollout: no feedback-policy leakage.
            actions = smooth_bank(rng, 2, config["episode_steps"],
                                  1 + config["episode_steps"] // 8,
                                  config["action_max_norm"])[1]
            images, _ = env.rollout(env.snapshot(), actions)
            relative_path = f"{split}/episode_{index:04d}.npz"
            path = dataset_dir / relative_path
            np.savez_compressed(path, rgb=images, actions=actions)
            records.append({"id": f"{split}_{index:04d}", "split": split, "seed": seed,
                            "path": relative_path, "layout": config["dataset_layout"],
                            "steps": len(actions), "frames": len(images),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            if (index + 1) % 8 == 0:
                print(f"collect {split}: {index+1}/{config[f'{split}_episodes']}", flush=True)
        write_json(dataset_dir / "manifest.json", {"schema_version": 1, "episodes": records,
            "inputs": ["rgb", "actions"], "contains_privileged_training_state": False,
            "temporal_contract": "rgb[t] -- actions[t] --> rgb[t+1]",
            "query_targets": "computed later from native-cadence RGB; train anchors only"})
    return len(records)


def main():
    require_slurm()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if (args.run_dir / "provenance.json").exists():
        raise FileExistsError("Refusing to overwrite a previously started run")
    config = json.loads(args.config.read_text())
    import torch
    torch.set_num_threads(1)  # Small 65x65 operations; avoid thread-launch overhead.
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "numpy", "gym", "omegaconf", "scipy")}
    write_json(args.run_dir / "provenance.json", {
        "job_id": os.environ["SLURM_JOB_ID"], "python": platform.python_version(),
        "config": config, "versions": versions,
        "upstream_snapshot": args.upstream_root, "protocol": "v1"})
    start = time.monotonic()
    summary = screen(config, args.run_dir, args.upstream_root)
    episodes = 0 if args.screen_only else collect(config, args.run_dir, args.upstream_root)
    write_json(args.run_dir / "completion.json", {
        "status": "COMPLETED", "elapsed_seconds": time.monotonic() - start,
        "episodes": episodes, "screen": summary,
        "next": "Inspect proposal/query support before GPU encoding; no WM trained."})
    print("COMPLETED: CPU preflight/data only; no learned WM result.", flush=True)


if __name__ == "__main__":
    main()
