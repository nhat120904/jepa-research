"""Episode-feature schema and window dataset for matched Stage-C arms."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


REQUIRED_EPISODE_KEYS = {
    "visual_features",
    "proprio",
    "actions",
    "eventual_success",
}


class StageCDataError(RuntimeError):
    pass


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1:
        raise StageCDataError(f"Unsupported manifest schema: {manifest.get('schema_version')}")
    if not manifest.get("episodes"):
        raise StageCDataError("Manifest contains no episode features")
    source_splits = {}
    path_splits = {}
    seen = set()
    for entry in manifest["episodes"]:
        uid = entry["episode_uid"]
        if uid in seen:
            raise StageCDataError(f"Duplicate episode_uid: {uid}")
        seen.add(uid)
        source = entry.get("source_episode_uid", uid)
        if source in source_splits and source_splits[source] != entry["split"]:
            raise StageCDataError(f"Source episode crosses splits: {source}")
        source_splits[source] = entry["split"]
        resolved = str((path.parent / entry["path"]).resolve())
        if resolved in path_splits and path_splits[resolved] != entry["split"]:
            raise StageCDataError(f"Feature file crosses splits: {resolved}")
        path_splits[resolved] = entry["split"]
    return manifest


class SegmentWindowDataset(Dataset):
    """Lazily constructs causal history/action/future windows from encoded episodes."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        history_steps: int,
        segment_steps: int,
        stride: int = 1,
        cache_episodes: int = 2,
        progress_dim: int = 7,
        training: bool = False,
        forbidden_training_source: str = "stage_b_validation_branches",
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.manifest = load_manifest(self.manifest_path)
        self.history_steps = int(history_steps)
        self.segment_steps = int(segment_steps)
        self.cache_episodes = int(cache_episodes)
        self.progress_dim = int(progress_dim)
        self.cache: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()
        self.entries = []
        self.windows: list[tuple[int, int]] = []
        candidate_groups = {}
        for entry in self.manifest["episodes"]:
            if entry["split"] != split:
                continue
            if training and (
                not entry.get("allow_training", False)
                or entry.get("source") == forbidden_training_source
            ):
                raise StageCDataError(
                    f"Forbidden training entry {entry.get('episode_uid')} from {entry.get('source')}"
                )
            entry_index = len(self.entries)
            self.entries.append(entry)
            first = self.history_steps - 1
            stop = int(entry["num_frames"]) - self.segment_steps
            is_candidate = int(entry.get("candidate_group_id", -1)) >= 0
            if split == "candidate_eval" and not is_candidate:
                raise StageCDataError("candidate_eval contains an ungrouped episode")
            if is_candidate:
                required = {"source_episode_uid", "prefix_uid", "decision_step",
                            "candidate_index", "candidate_count"}
                if required - set(entry):
                    raise StageCDataError(f"Candidate {entry['episode_uid']} missing {required - set(entry)}")
                t = int(entry["decision_step"])
                if not first <= t < stop:
                    raise StageCDataError(f"Invalid causal decision window for {entry['episode_uid']}: {t}")
                self.windows.append((entry_index, t))
                key = (int(entry["task_id"]), int(entry["candidate_group_id"]))
                candidate_groups.setdefault(key, []).append(entry_index)
            else:
                self.windows.extend((entry_index, t) for t in range(first, stop, stride))
        if not self.windows:
            raise StageCDataError(f"No valid {split} windows in {self.manifest_path}")
        # Run on the compute node when constructing the dataset, before any training.
        for key, indices in candidate_groups.items():
            entries = [self.entries[i] for i in indices]
            reference = entries[0]
            count = int(reference["candidate_count"])
            if count < 2 or sorted(int(e["candidate_index"]) for e in entries) != list(range(count)):
                raise StageCDataError(f"Duplicate/missing candidates in {key}")
            for field in ("source_episode_uid", "prefix_uid", "decision_step", "candidate_count"):
                if any(e[field] != reference[field] for e in entries):
                    raise StageCDataError(f"Candidate group {key} mixes {field}")
            t = int(reference["decision_step"])
            history_slice = slice(t - self.history_steps + 1, t + 1)
            expected = self._episode(indices[0])
            for i in indices[1:]:
                actual = self._episode(i)
                for field in ("visual_features", "proprio"):
                    if not torch.equal(expected[field][history_slice], actual[field][history_slice]):
                        raise StageCDataError(f"Candidates in {key} do not share pre-action {field}")

    def __len__(self) -> int:
        return len(self.windows)

    def _episode(self, entry_index: int) -> dict[str, torch.Tensor]:
        if entry_index in self.cache:
            episode = self.cache.pop(entry_index)
            self.cache[entry_index] = episode
            return episode
        entry = self.entries[entry_index]
        path = self.root / entry["path"]
        episode = torch.load(path, map_location="cpu", weights_only=False)
        missing = REQUIRED_EPISODE_KEYS - set(episode)
        if missing:
            raise StageCDataError(f"{path} is missing keys: {sorted(missing)}")
        frames = int(entry["num_frames"])
        for key in REQUIRED_EPISODE_KEYS:
            if len(episode[key]) != frames:
                raise StageCDataError(f"{path}:{key} has {len(episode[key])}, expected {frames}")
        self.cache[entry_index] = episode
        while len(self.cache) > self.cache_episodes:
            self.cache.popitem(last=False)
        return episode

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        entry_index, t = self.windows[index]
        entry = self.entries[entry_index]
        episode = self._episode(entry_index)
        h0 = t - self.history_steps + 1
        f1 = t + self.segment_steps + 1
        progress = episode.get("progress")
        if progress is None:
            progress_future = torch.zeros(self.segment_steps, self.progress_dim)
            has_progress = torch.tensor(False)
        else:
            progress_future = progress[t + 1 : f1].float()
            has_progress = torch.tensor(True)
        return {
            "visual_history": episode["visual_features"][h0 : t + 1].float(),
            "proprio_history": episode["proprio"][h0 : t + 1].float(),
            "actions": episode["actions"][t : t + self.segment_steps].float(),
            "visual_future": episode["visual_features"][t + 1 : f1].float(),
            "proprio_future": episode["proprio"][t + 1 : f1].float(),
            "progress_future": progress_future,
            "has_progress": has_progress,
            "eventual_success": episode["eventual_success"][t].float(),
            "task_id": torch.tensor(int(entry["task_id"]), dtype=torch.long),
            "episode_id": torch.tensor(int(entry["episode_id"]), dtype=torch.long),
            "candidate_group_id": torch.tensor(
                int(entry.get("candidate_group_id", -1)), dtype=torch.long
            ),
            "candidate_index": torch.tensor(
                int(entry.get("candidate_index", -1)), dtype=torch.long
            ),
            "candidate_count": torch.tensor(int(entry.get("candidate_count", 0)), dtype=torch.long),
        }


def class_balance(dataset: SegmentWindowDataset) -> dict[str, int]:
    positives = 0
    negatives = 0
    for entry_index, t in dataset.windows:
        label = bool(dataset._episode(entry_index)["eventual_success"][t].item())
        positives += int(label)
        negatives += int(not label)
    return {"positive": positives, "negative": negatives}


def require_binary_training_labels(dataset: SegmentWindowDataset) -> dict[str, int]:
    balance = class_balance(dataset)
    if min(balance.values()) == 0:
        raise StageCDataError(
            "Continuation-value training requires both success and failure windows; "
            f"observed {balance}. Collect a separate training-rollout split."
        )
    return balance
