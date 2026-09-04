#!/usr/bin/env python3
"""Minimal independent recomputation of the ablation headline numbers.

Deliberately shares no code with analyze_scene_event_ablation.py.  Reads the raw
per-reset shards and recomputes arm rates, the paired contrasts and the
reproduction check from scratch, so the reported conclusion does not rest on a
single analysis implementation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, required=True)
    parser.add_argument("--factorial-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("verification must run inside a Slurm compute job")

    def read(root: Path) -> dict:
        table = {}
        for shard in sorted(root.glob("*/result.json")):
            payload = json.loads(shard.read_text())
            key = (payload["task_id"], payload["reset_seed"])
            for row in payload["results"]:
                table[(key, row["arm"], row["observer_seed"])] = (
                    bool(row["success"]),
                    tuple(row["deployed_skills"]),
                )
        return table

    abl = read(args.ablation_root)
    fac = read(args.factorial_root)
    resets = sorted({k[0] for k in abl})
    arms = sorted({k[1] for k in abl})

    def per_reset(arm: str) -> np.ndarray:
        out = []
        for reset in resets:
            cells = [abl[(reset, arm, s)][0] for s in (None, 0, 1, 2) if (reset, arm, s) in abl]
            out.append(float(np.mean([float(c) for c in cells])))
        return np.asarray(out)

    def ci(values: np.ndarray, seed: int) -> tuple[float, float, float]:
        rng = np.random.default_rng(seed)
        draws = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
        return float(values.mean()), float(np.percentile(draws, 2.5)), float(
            np.percentile(draws, 97.5)
        )

    shared = [k for k in abl if k in fac]
    mismatch = [k for k in shared if abl[k] != fac[k]]

    rates = {arm: ci(per_reset(arm), 7) for arm in arms}
    pairs = [
        ("history_full", "action_only_full"),
        ("history_full", "obs_history_full"),
        ("history_full", "openloop_transition"),
        ("action_only_full", "openloop_transition"),
        ("obs_history_full", "frame_full"),
        ("history_full", "frame_full"),
        ("history_full", "oracle_event"),
    ]
    contrasts = {
        f"{a}-{b}": ci(per_reset(a) - per_reset(b), 11) for a, b in pairs
    }

    result = {
        "num_resets": len(resets),
        "shared_cells_vs_factorial": len(shared),
        "reproduction_mismatches": len(mismatch),
        "rates_percent": {
            arm: [round(100 * v, 4) for v in vals] for arm, vals in rates.items()
        },
        "contrasts_points": {
            name: [round(100 * v, 4) for v in vals] for name, vals in contrasts.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
