"""Episode-split window sampler and train-anchor query generator."""
import json
from pathlib import Path

import numpy as np
import torch

from .queries import chroma_features, ordered_score


class EpisodeStore:
    def __init__(self, feature_root, dataset_root, split):
        if split not in ("train", "val"):
            raise ValueError("Development store only permits train or val")
        feature_root, dataset_root = Path(feature_root), Path(dataset_root)
        manifest = json.loads((feature_root / "manifest.json").read_text())
        if manifest.get("test_split_read") is not False:
            raise RuntimeError("Feature manifest does not attest sealed test split")
        records = [r for r in manifest["records"] if r["split"] == split]
        self.episodes = []
        for record in records:
            with np.load(feature_root / record["path"]) as encoded:
                features = encoded["features"].astype(np.float32)
                actions = encoded["actions"].astype(np.float32)
            source = dataset_root / record["source"]
            with np.load(source) as raw:
                rgb = raw["rgb"]
            if len(features) != len(rgb) or len(actions) + 1 != len(rgb):
                raise RuntimeError(f"Temporal mismatch: {record['id']}")
            self.episodes.append({"id": record["id"], "features": features,
                                  "global": features.mean(1), "actions": actions,
                                  "chroma": chroma_features(rgb).astype(np.float32)})
        if not self.episodes:
            raise RuntimeError(f"No episodes for {split}")
        self.patch_count, self.patch_dim = self.episodes[0]["features"].shape[-2:]


class WindowSampler:
    OPS = ("reach", "occupancy", "endpoint", "ordered")

    def __init__(self, store, train_anchor_store, history=4, queries=8, seed=0):
        self.store, self.history, self.queries = store, history, queries
        self.rng = np.random.default_rng(seed)
        # Anchor pool is always TRAIN frames, including for validation random negatives.
        self.anchor_global = np.concatenate([e["global"] for e in train_anchor_store.episodes])
        self.anchor_chroma = np.concatenate([e["chroma"] for e in train_anchor_store.episodes])
        self.query_dim = 2 * store.patch_dim + len(self.OPS)

    def _query(self, episode, begin, horizon):
        future_global = episode["global"][begin + 1:begin + horizon + 1]
        future_chroma = episode["chroma"][begin + 1:begin + horizon + 1]
        vectors, targets, types = [], [], []
        for query_index in range(self.queries):
            op = query_index % len(self.OPS)
            # Balanced positive-looking and unrelated anchors; target remains continuous.
            use_future = (query_index // len(self.OPS)) % 2 == 0
            if op == 3 and use_future:
                i = int(self.rng.integers(0, horizon - 1))
                j = int(self.rng.integers(i + 1, horizon))
                anchor_a_g, anchor_b_g = future_global[i], future_global[j]
                anchor_a_c, anchor_b_c = future_chroma[i], future_chroma[j]
            elif op == 3:
                indices = self.rng.integers(0, len(self.anchor_global), 2)
                anchor_a_g, anchor_b_g = self.anchor_global[indices]
                anchor_a_c, anchor_b_c = self.anchor_chroma[indices]
            else:
                if use_future:
                    index = int(self.rng.integers(horizon))
                    anchor_g, anchor_c = future_global[index], future_chroma[index]
                else:
                    index = int(self.rng.integers(len(self.anchor_global)))
                    anchor_g, anchor_c = self.anchor_global[index], self.anchor_chroma[index]
                anchor_a_g = anchor_g if op == 1 else np.zeros_like(anchor_g)
                anchor_b_g = anchor_g if op in (0, 2) else np.zeros_like(anchor_g)
                anchor_a_c = anchor_c if op == 1 else np.zeros_like(anchor_c)
                anchor_b_c = anchor_c if op in (0, 2) else np.zeros_like(anchor_c)
            onehot = np.zeros(len(self.OPS), dtype=np.float32)
            onehot[op] = 1
            vectors.append(np.concatenate([anchor_a_g, anchor_b_g, onehot]))
            sim_a = np.clip(future_chroma @ anchor_a_c, 0, 1)
            sim_b = np.clip(future_chroma @ anchor_b_c, 0, 1)
            if op == 0:
                target = sim_b.max()
            elif op == 1:
                target = sim_a.mean()
            elif op == 2:
                target = sim_b[-1]
            else:
                target = ordered_score(sim_a, sim_b)
            targets.append(target)
            types.append(op)
        return np.stack(vectors), np.asarray(targets, np.float32), np.asarray(types)

    def batch(self, batch_size, horizon):
        histories, actions, futures, queries, answers, types = [], [], [], [], [], []
        for _ in range(batch_size):
            episode = self.store.episodes[int(self.rng.integers(len(self.store.episodes)))]
            max_begin = len(episode["actions"]) - horizon
            begin = int(self.rng.integers(self.history - 1, max_begin + 1))
            histories.append(episode["features"][begin - self.history + 1:begin + 1])
            actions.append(episode["actions"][begin:begin + horizon])
            futures.append(episode["features"][begin + 1:begin + horizon + 1])
            q, y, t = self._query(episode, begin, horizon)
            queries.append(q); answers.append(y); types.append(t)
        to_tensor = lambda x: torch.from_numpy(np.stack(x))
        return tuple(map(to_tensor, (histories, actions, futures, queries, answers, types)))
