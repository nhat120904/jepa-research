"""Collect proposal-matched RGB branches with shared prospective queries."""
import argparse
import hashlib
import json
from pathlib import Path

from .runtime import require_slurm


def main():
    require_slurm()
    import numpy as np
    import torch
    from .preflight import sample_task
    from .proposals import visual_goal_bank
    from .wall_adapter import WallRGB
    from .queries import chroma_features, answer

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    records, statistics = [], []
    # Fixed before collection; no filtering by success or score. Separate from old data.
    candidate_ids = [1, 2, 3, 17, 53, 89, 100, 120]
    for split, count, offset, horizons in (("train", 24, 400000, [32, 48]),
                                           ("val", 12, 500000, [32, 48, 64])):
        (args.output / split).mkdir()
        for prefix in range(count):
            seed = 20260922 + offset + 103 * prefix
            layout = [32, [20, 30, 40][prefix % 3]]
            env = WallRGB(args.upstream, *layout, seed=seed)
            start, a, b = sample_task(env, np.random.default_rng(seed))
            initial = env.reset(start)
            # Four identical observations correspond to an actual stationary prefix.
            history = [initial] + [env.step(np.zeros(2, np.float32)) for _ in range(3)]
            snapshot = env.snapshot()
            anchors = np.stack([env.goal_image(a), env.goal_image(b)])
            fa, fb = chroma_features(anchors)
            for horizon in horizons:
                bank = visual_goal_bank(np.random.default_rng(seed + horizon), history[-1],
                                        anchors[0], anchors[1], 128, horizon, 7, 1.8)
                labels = []
                for candidate_id in candidate_ids:
                    rgb, _ = env.rollout(snapshot, bank[candidate_id])
                    record_id = f"{split}_p{prefix:03d}_h{horizon}_c{candidate_id:03d}"
                    relative = f"{split}/{record_id}.npz"
                    path = args.output / relative
                    np.savez_compressed(path, rgb=rgb, actions=bank[candidate_id],
                                        anchors=anchors, history_rgb=np.stack(history))
                    labels.append(float(answer(chroma_features(rgb[1:]), fa, fb)["a_then_b"]))
                    records.append({"id": record_id, "split": split, "path": relative,
                                    "prefix_id": f"{split}_{prefix:03d}", "layout": layout,
                                    "horizon": horizon, "candidate_id": candidate_id,
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                statistics.append({"prefix_id": f"{split}_{prefix:03d}", "horizon": horizon,
                                   "order_min": min(labels), "order_max": max(labels),
                                   "order_spread": max(labels) - min(labels)})
            print(f"collected {split} prefix {prefix+1}/{count}", flush=True)
    manifest = {"schema_version": 2, "kind": "prospective_branches", "episodes": records,
                "split_unit": "prefix including all candidate/horizon branches",
                "test_split_read": False, "candidate_ids": candidate_ids,
                "goals_chosen_before_candidates": True,
                "history": "four actual observations at stationary reset",
                "training_arrays": ["rgb", "actions", "anchors", "history_rgb"],
                "statistics": statistics}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"COMPLETED {len(records)} branches", flush=True)


if __name__ == "__main__":
    main()
