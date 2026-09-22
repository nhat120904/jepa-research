"""All candidates at a prefix share prospectively chosen queries; prefix split."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .queries import chroma_features, similarity, ordered_score

QUERY_NAMES = ["reach_a", "reach_b", "occupancy_a", "occupancy_b",
               "endpoint_a", "endpoint_b", "a_then_b", "b_then_a"]


def make_queries(anchors, projection):
    # Same frozen random projection per patch; preserve all 16 spatial locations.
    a, b = (anchors.float() @ projection).flatten(1)
    specs = [(a,a,0), (b,b,0), (a,a,1), (b,b,1),
             (a,a,2), (b,b,2), (a,b,3), (b,a,3)]
    return torch.stack([torch.cat([x, y, torch.eye(4)[op]]) for x,y,op in specs])


def targets(rgb, anchors):
    fa, fb = chroma_features(anchors)
    features = chroma_features(rgb[1:])  # exclude current frame
    a, b = similarity(features, fa), similarity(features, fb)
    return np.asarray([a.max(), b.max(), a.mean(), b.mean(), a[-1], b[-1],
                       ordered_score(a,b), ordered_score(b,a)], np.float32)


class BranchStore:
    def __init__(self, feature_root, projection_dim=32, seed=20260922):
        feature_root = Path(feature_root)
        manifest = json.loads((feature_root / "manifest.json").read_text())
        if manifest.get("dataset_kind") != "prospective_branches":
            raise ValueError("Requires prospective branch dataset")
        self.patches, self.dim = manifest["feature_shape"]
        generator = torch.Generator().manual_seed(seed)
        self.projection = torch.randn(self.dim, projection_dim, generator=generator)
        self.projection /= projection_dim ** .5
        self.query_dim = 2 * self.patches * projection_dim + 4
        self.rows = []
        groups = {}
        for record in manifest["records"]:
            if record["split"] not in ("train", "val"):
                raise ValueError("Sealed test must not be loaded")
            path = feature_root / record["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"Feature hash mismatch: {path}")
            with np.load(path) as f:
                history = torch.from_numpy(f["history_features"].copy())
                future = torch.from_numpy(f["features"][1:].copy())
                actions = torch.from_numpy(f["actions"].copy())
                query = make_queries(torch.from_numpy(f["anchor_features"].copy()), self.projection)
            raw_path = Path(manifest["dataset_root"]) / record["source"]
            if hashlib.sha256(raw_path.read_bytes()).hexdigest() != record["source_sha256"]:
                raise ValueError(f"Source RGB hash mismatch: {raw_path}")
            with np.load(raw_path) as raw:
                target = torch.from_numpy(targets(raw["rgb"], raw["anchors"]))
            row = {**record, "history": history, "future": future, "actions": actions,
                   "queries": query, "answers": target}
            key = record["prefix_id"]
            if key in groups:
                old = groups[key]
                if not (torch.equal(query, old["queries"]) and
                        torch.equal(history, old["history"]) and old["split"] == row["split"]):
                    raise AssertionError("Candidates must share exact queries/history/split")
            groups[key] = row
            self.rows.append(row)
        self.train_horizons = [32, 48]
        self.train = [i for i,r in enumerate(self.rows) if r["split"] == "train"]
        self.val = [i for i,r in enumerate(self.rows) if r["split"] == "val"]
        assert self.train and self.val
        assert set(self.rows[i]["horizon"] for i in self.train) == set(self.train_horizons)

    def batch(self, indices, device):
        return {key: torch.stack([self.rows[i][key] for i in indices]).to(device).float()
                for key in ("history", "future", "actions", "queries", "answers")}

    def sample(self, rng, batch_size, device):
        horizon = int(rng.choice(self.train_horizons))
        pool = [i for i in self.train if self.rows[i]["horizon"] == horizon]
        return self.batch(rng.choice(pool, batch_size, replace=True), device)
