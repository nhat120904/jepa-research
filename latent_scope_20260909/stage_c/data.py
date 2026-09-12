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
            self.windows.extend((entry_index, t) for t in range(first, stop, stride))
        if not self.windows:
            raise StageCDataError(f"No valid {split} windows in {self.manifest_path}")

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
