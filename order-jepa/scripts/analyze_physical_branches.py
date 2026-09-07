#!/usr/bin/env python3
"""Cheap physical-only qualification of collected swapped-order branches."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from order_jepa.core import clustered_bootstrap_mean  # noqa: E402


def angle_distance(a: float, b: float) -> float:
    return float(abs((a - b + np.pi) % (2 * np.pi) - np.pi))


def clustered_metric(values: np.ndarray, episodes: np.ndarray, seed: int) -> dict:
    mean, lo, hi = clustered_bootstrap_mean(
        values.astype(np.float64), episodes, n_resamples=5000, seed=seed
    )
    return {"mean": mean, "ci95": [lo, hi], "n": int(values.size)}


def stratum_summary(
    mask: np.ndarray, obj: np.ndarray, episodes: np.ndarray, seed: int
) -> dict:
    values = obj[mask]
    stratum_episodes = episodes[mask]
    if values.size == 0:
        return {"pairs": 0}
    return {
        "pairs": int(values.size),
        "fraction": float(values.size / obj.size),
        "effect_mean": clustered_metric(values, stratum_episodes, seed),
        "effect_median": float(np.median(values)),
        "fraction_above_0.25": clustered_metric(
            values >= 0.25, stratum_episodes, seed + 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branches-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("physical branch analysis must run on a compute node")

    rows = []
    reset_rows = []
    for meta_path in sorted(args.branches_dir.glob("anchor_*.json")):
        meta = json.loads(meta_path.read_text())
        reset_rows.append(
            (
                float(meta["reset_physics_max_abs"]),
                float(meta["reset_pixel_max_abs"]),
                float(meta["repeat_endpoint_max_abs"]),
            )
        )
        arrays = np.load(meta_path.with_suffix(".npz"), allow_pickle=False)
        for pair in meta["order_pairs"]:
            ij, ji = int(pair["ij"]), int(pair["ji"])
            state_ij, state_ji = arrays["candidate_state"][ij], arrays["candidate_state"][ji]
            agent_effect = float(np.linalg.norm(state_ij[:2] - state_ji[:2]))
            object_position_effect = float(np.linalg.norm(state_ij[2:4] - state_ji[2:4]))
            object_angle_effect = angle_distance(float(state_ij[4]), float(state_ji[4]))
            rows.append(
                {
                    "anchor_id": int(meta["anchor_id"]),
                    "episode": int(meta["episode"]),
                    "pair_id": int(pair["pair_id"]),
                    "agent_effect_px": agent_effect,
                    "object_position_effect_px": object_position_effect,
                    "object_angle_effect_rad": object_angle_effect,
                    "object_effect_normalized": object_position_effect / 20.0
                    + object_angle_effect / (np.pi / 9.0),
                    "contact_steps_ij": int(np.sum(arrays["candidate_contacts"][ij] > 0)),
                    "contact_steps_ji": int(np.sum(arrays["candidate_contacts"][ji] > 0)),
                }
            )
    if not rows:
        raise ValueError("no order pairs found")
    obj = np.asarray([row["object_effect_normalized"] for row in rows])
    pos = np.asarray([row["object_position_effect_px"] for row in rows])
    agent = np.asarray([row["agent_effect_px"] for row in rows])
    contact = np.asarray(
        [row["contact_steps_ij"] + row["contact_steps_ji"] > 0 for row in rows]
    )
    contact_ij = np.asarray([row["contact_steps_ij"] > 0 for row in rows])
    contact_ji = np.asarray([row["contact_steps_ji"] > 0 for row in rows])
    episodes = np.asarray([row["episode"] for row in rows], dtype=np.int64)
    above = obj >= 0.25
    no_contact = ~(contact_ij | contact_ji)
    both_contact = contact_ij & contact_ji
    asymmetric_contact = contact_ij ^ contact_ji
    reset_np = np.asarray(reset_rows, dtype=np.float64)
    summary = {
        "schema": "order-jepa-physical-qualification-v2",
        "pairs": len(rows),
        "anchors": len(set(row["anchor_id"] for row in rows)),
        "episodes": int(np.unique(episodes).size),
        "reset": {
            "physics_max_abs": float(np.max(reset_np[:, 0])),
            "pixel_max_abs": float(np.max(reset_np[:, 1])),
            "repeat_endpoint_max_abs": float(np.max(reset_np[:, 2])),
        },
        "contact_pair_fraction": clustered_metric(contact, episodes, 20260907),
        "object_effect_normalized": {
            **clustered_metric(obj, episodes, 20260908),
            "median": float(np.median(obj)),
            "max": float(np.max(obj)),
            "fraction_above_0.25": clustered_metric(above, episodes, 20260909),
        },
        "object_position_effect_px": {
            "mean": float(np.mean(pos)),
            "median": float(np.median(pos)),
            "max": float(np.max(pos)),
        },
        "agent_effect_px": {
            "mean": float(np.mean(agent)),
            "median": float(np.median(agent)),
            "max": float(np.max(agent)),
        },
        "object_effect_by_contact": {
            "no_contact": stratum_summary(no_contact, obj, episodes, 20260910),
            "both_orders_contact": stratum_summary(
                both_contact, obj, episodes, 20260912
            ),
            "one_order_contacts": stratum_summary(
                asymmetric_contact, obj, episodes, 20260914
            ),
        },
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
