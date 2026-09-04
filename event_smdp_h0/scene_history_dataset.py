"""Sequence dataset construction for the Scene event-history factorial.

The H1 collector already walked canonical milestone paths and, from every root
snapshot, executed all seven skills.  That gives two kinds of labelled event
states at no extra simulation cost:

``canonical`` roots
    ``o_0, a_0, o_1, ... , o_k`` where every ``a`` is the canonical skill.  This
    is the only subset the original single-frame observer was trained on.
``endpoint`` deviations
    the same prefix followed by one arbitrary skill.  These cover event states
    that no canonical path ever visits, which is exactly where the deployed
    observer failed.

Both subsets are emitted as padded ``(observation, previous skill)`` sequences
so the frame and history arms consume byte-identical data.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from event_smdp_h0.scene_event_history import NO_SKILL, canonical_skill_paths


def _root_view(
    data: dict[str, np.ndarray], mask: np.ndarray, root_index: int
) -> dict[str, Any]:
    rows = np.flatnonzero(mask & (data["root_index"] == root_index))
    if rows.size == 0:
        raise RuntimeError(f"missing root {root_index}")
    return {"rows": rows, "first": int(rows[0])}


def build_sequences(
    data: dict[str, np.ndarray], view: str, *, max_steps: int | None = None
) -> dict[str, np.ndarray]:
    """Expand concatenated shard rows into padded prefix sequences."""

    before_key, after_key = f"before_{view}", f"after_{view}"
    feature_dim = int(data[before_key].shape[1])
    groups = np.unique(
        np.stack([data["task_id"], data["reset_seed"], data["path_id"]], axis=1), axis=0
    )
    limit = max_steps or (
        max(len(path) for task in (4, 5) for path in canonical_skill_paths(task)) + 2
    )

    features: list[np.ndarray] = []
    skills: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    for task_id, reset_seed, path_id in groups:
        task_id, reset_seed, path_id = int(task_id), int(reset_seed), int(path_id)
        mask = (
            (data["task_id"] == task_id)
            & (data["reset_seed"] == reset_seed)
            & (data["path_id"] == path_id)
        )
        path = canonical_skill_paths(task_id)[path_id]
        prefix_features: list[np.ndarray] = []
        prefix_skills: list[int] = []
        for root_index in range(len(path) + 1):
            root = _root_view(data, mask, root_index)
            first = root["first"]
            prefix_features.append(data[before_key][first].astype(np.float32))
            prefix_skills.append(NO_SKILL if root_index == 0 else int(path[root_index - 1]))
            common = {
                "task_id": task_id,
                "reset_seed": reset_seed,
                "path_id": path_id,
                "root_index": root_index,
                "goal": data["goal"][first].astype(np.float32),
            }
            features.append(np.stack(prefix_features))
            skills.append(np.asarray(prefix_skills, dtype=np.int64))
            records.append(
                {
                    **common,
                    "cube": int(data["before_cube_stage"][first]),
                    "window": int(data["before_window_stage"][first]),
                    "stable": float(data["before_stable_count"][first] >= 3),
                    "terminal_skill": -1,
                    "is_endpoint": 0,
                }
            )
            canonical_next = int(path[root_index]) if root_index < len(path) else None
            for row in root["rows"]:
                skill = int(data["skill"][row])
                if skill == canonical_next:
                    # Identical to the next canonical root sample.
                    continue
                features.append(
                    np.stack(
                        prefix_features + [data[after_key][row].astype(np.float32)]
                    )
                )
                skills.append(np.asarray(prefix_skills + [skill], dtype=np.int64))
                records.append(
                    {
                        **common,
                        "cube": int(data["after_cube_stage"][row]),
                        "window": int(data["after_window_stage"][row]),
                        "stable": float(data["after_stable_count"][row] >= 3),
                        "terminal_skill": skill,
                        "is_endpoint": 1,
                    }
                )

    lengths = np.asarray([len(item) for item in features], dtype=np.int64)
    if int(lengths.max()) > limit:
        raise RuntimeError(f"sequence longer than the {limit}-step budget")
    steps = int(lengths.max())
    padded = np.zeros((len(features), steps, feature_dim), dtype=np.float32)
    padded_skills = np.full((len(features), steps), NO_SKILL, dtype=np.int64)
    for index, (feature, skill) in enumerate(zip(features, skills)):
        padded[index, : len(feature)] = feature
        padded_skills[index, : len(skill)] = skill

    def column(key: str, dtype: Any) -> np.ndarray:
        return np.asarray([record[key] for record in records], dtype=dtype)

    return {
        "feature": padded,
        "prev_skill": padded_skills,
        "length": lengths,
        "goal": np.stack([record["goal"] for record in records]),
        "task_id": column("task_id", np.int64),
        "reset_seed": column("reset_seed", np.int64),
        "path_id": column("path_id", np.int64),
        "root_index": column("root_index", np.int64),
        "terminal_skill": column("terminal_skill", np.int64),
        "is_endpoint": column("is_endpoint", np.int64),
        "cube": column("cube", np.int64),
        "window": column("window", np.int64),
        "stable": column("stable", np.float32),
    }


def restrict(dataset: dict[str, np.ndarray], coverage: str) -> dict[str, np.ndarray]:
    """``canonical`` keeps only milestone roots; ``full`` keeps every state."""

    if coverage == "full":
        return dataset
    if coverage != "canonical":
        raise ValueError(f"unknown coverage: {coverage}")
    keep = dataset["is_endpoint"] == 0
    return {key: value[keep] for key, value in dataset.items()}
