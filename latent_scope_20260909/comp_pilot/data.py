"""GPU-side feature store, window sampling and precomputed segment labels."""

from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from .labels import MAP_CELLS, EpisodeLabels, split_is_hard
from .models import STRIDE

HISTORY_TOKENS = 96
EVAL_PARTS = {"64=32+32": [32, 32], "64=24+40": [24, 40], "128=64+64": [64, 64], "128=32+32+64": [32, 32, 64]}
DIRECT_LENGTHS = (32, 64, 128)
TRAIN_LENGTHS = (32, 64)


class FeatureStore:
    def __init__(self, feature_root: Path, label_root: Path, audit_path: Path, device: torch.device, max_episodes: int | None = None) -> None:
        manifest = json.loads((feature_root / "manifest.json").read_text())["episodes"]
        if max_episodes:
            keep = {s: [m for m in manifest if m["split"] == s][: max(2, max_episodes // 3)] for s in ("train", "val", "test")}
            manifest = sorted(sum(keep.values(), []), key=lambda m: m["episode"])
        audit = json.loads(audit_path.read_text())
        self.label_agree = {r["episode"]: bool(r.get("agree", False)) for r in audit["labels"]["per_episode"]}
        self.episodes = [m["episode"] for m in manifest]
        self.split = {m["episode"]: m["split"] for m in manifest}
        self.lengths = {m["episode"]: m["frames"] for m in manifest}
        self.feature_root = feature_root
        self.manifest = manifest
        self.label_root = label_root
        self.device = device

    def load(self) -> None:
        """Load features to the device (call after LabelTable, whose worker pool forks)."""
        visual, proprio, actions, offsets, total = [], [], [], {}, 0
        for m in self.manifest:
            payload = torch.load(self.feature_root / m["path"], map_location="cpu", weights_only=True)
            offsets[m["episode"]] = total
            total += len(payload["visual"])
            visual.append(payload["visual"])
            proprio.append(payload["proprio"])
            actions.append(payload["actions"])
        self.offsets = offsets
        self.visual = torch.cat(visual).to(self.device)
        proprio_t, actions_t = torch.cat(proprio), torch.cat(actions)
        train_rows = torch.cat([torch.arange(offsets[e], offsets[e] + self.lengths[e]) for e in self.episodes if self.split[e] == "train"])
        self.proprio_stats = (proprio_t[train_rows].mean(0), proprio_t[train_rows].std(0).clamp_min(1e-3))
        self.action_stats = (actions_t[train_rows].mean(0), actions_t[train_rows].std(0).clamp_min(1e-3))
        self.proprio = ((proprio_t - self.proprio_stats[0]) / self.proprio_stats[1]).to(self.device)
        self.actions = ((actions_t - self.action_stats[0]) / self.action_stats[1]).to(self.device)

    def split_episodes(self, split: str) -> list[int]:
        return [e for e in self.episodes if self.split[e] == split]

    def batch(self, eps: list[int], starts: list[int], length: int) -> dict:
        dev = self.device
        off = torch.tensor([self.offsets[e] for e in eps], device=dev)[:, None]
        t = torch.tensor(starts, device=dev)[:, None]
        k = torch.arange(HISTORY_TOKENS, device=dev)[None]
        hist_local = t - STRIDE * (HISTORY_TOKENS - 1 - k)
        mask = hist_local >= 0
        hist = off + hist_local.clamp_min(0)
        fut = off + t + 1 + torch.arange(length, device=dev)[None]
        act = off + t + torch.arange(length, device=dev)[None]
        cur = (off + t)[:, 0]
        return {
            "hist_visual": self.visual[hist],
            "hist_proprio": self.proprio[hist],
            "hist_actions": self.actions[hist],
            "hist_mask": mask,
            "cur_visual": self.visual[cur],
            "cur_proprio": self.proprio[cur],
            "fut_visual": self.visual[fut],
            "fut_proprio": self.proprio[fut],
            "actions": self.actions[act],
        }


def _label_worker(task):
    label_root, episode, windows, hard_specs = task
    labels = EpisodeLabels(Path(label_root) / f"episode_{episode:06d}.npz")
    out = {}
    for a, b in windows:
        w = labels.window(a, b)
        out[(a, b)] = (w["count_increment"], w["count_empty"], w["area_empty"], w["has_contact"], np.packbits(w["map"].astype(np.uint8)))
    hard = {(t, name): split_is_hard(labels, t, parts) for t, name, parts in hard_specs}
    return episode, out, hard


class LabelTable:
    """Precomputed targets for frames (a, b] of every window the pilot uses."""

    def __init__(self, store: FeatureStore, workers: int) -> None:
        tasks = []
        for e in store.episodes:
            n = store.lengths[e]
            windows, hard_specs = set(), []
            if store.split[e] in ("train", "val"):
                for length in TRAIN_LENGTHS:
                    m = length // 2
                    for t in range(0, n - length, STRIDE):
                        windows |= {(t, t + length), (t, t + m), (t + m, t + length)}
            if store.split[e] in ("val", "test"):
                for t in eval_starts(n):
                    for length in DIRECT_LENGTHS:
                        windows.add((t, t + length))
                    for name, parts in EVAL_PARTS.items():
                        cut = t
                        for p in parts:
                            windows.add((cut, cut + p))
                            cut += p
                        hard_specs.append((t, name, parts))
            tasks.append((str(store.label_root), e, sorted(windows), hard_specs))
        self.table, self.hard = {}, {}
        with mp.get_context("fork").Pool(workers) as pool:
            for episode, out, hard in pool.imap_unordered(_label_worker, tasks):
                for key, value in out.items():
                    self.table[(episode, *key)] = value
                for key, value in hard.items():
                    self.hard[(episode, *key)] = value
        self.device = store.device

    def get(self, eps: list[int], windows: list[tuple[int, int]]) -> dict:
        rows = [self.table[(e, a, b)] for e, (a, b) in zip(eps, windows)]
        scal = torch.tensor([[r[0], r[1], r[2], r[3]] for r in rows], dtype=torch.float32, device=self.device)
        maps = np.unpackbits(np.stack([r[4] for r in rows]), axis=1)[:, : MAP_CELLS * MAP_CELLS]
        return {
            "count_increment": scal[:, 0],
            "count_empty": scal[:, 1],
            "area_empty": scal[:, 2],
            "has_contact": scal[:, 3],
            "map": torch.from_numpy(maps.astype(np.float32)).to(self.device),
        }


def eval_starts(n: int) -> list[int]:
    return list(range(0, n - 128, 16))


def train_starts(store: FeatureStore, split: str, length: int, stride: int) -> list[tuple[int, int]]:
    return [(e, t) for e in store.split_episodes(split) for t in range(0, store.lengths[e] - length, stride)]
