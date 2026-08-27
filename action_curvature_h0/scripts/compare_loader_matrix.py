#!/usr/bin/env python3
"""Decide whether a loader knob preserves the official recipe's computation.

Two criteria, in order:

1. **Batch stream.** The sha256 of every batch the loader yields must match A1's
   exactly. If the streams differ the configuration feeds the model different
   data and is rejected immediately, whatever the weights do.
2. **Weights, against a measured floor.** A1 and A2 are the same configuration
   run twice, so their weight difference *is* the reproducibility floor of this
   stack (CUDA nondeterminism and anything else irreducible). A candidate is
   accepted only if its difference from A1 is no larger than that floor.

Without step 2's floor, "the weights differ" is not a valid criterion, because
the recipe may not reproduce itself bitwise in the first place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

BASELINE = "A1"
REPEAT = "A2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_hashes(work: Path, name: str) -> list[str]:
    path = work / f"{name}_hashes.json"
    return json.loads(path.read_text())["hashes"] if path.exists() else []


def state_dict(path: Path) -> dict[str, torch.Tensor]:
    blob = torch.load(path, map_location="cpu")
    if isinstance(blob, dict):
        for key in ("model", "state_dict"):
            if key in blob:
                return blob[key]
    return blob


def weight_delta(a: dict[str, torch.Tensor],
                 b: dict[str, torch.Tensor]) -> dict[str, Any]:
    if set(a) != set(b):
        return {"comparable": False, "reason": "different tensor keys"}
    worst, worst_key, n_diff = 0.0, None, 0
    for k in sorted(a):
        x, y = a[k].float(), b[k].float()
        if x.shape != y.shape:
            return {"comparable": False, "reason": f"shape mismatch at {k}"}
        d = (x - y).abs().max().item()
        s = x.abs().max().item()
        rel = d / s if s > 0 else d
        if d > 0:
            n_diff += 1
        if rel > worst:
            worst, worst_key = rel, k
    return {"comparable": True, "n_tensors": len(a), "n_differing": n_diff,
            "worst_relative": worst, "worst_key": worst_key}


def main() -> None:
    args = parse_args()
    names = sorted(p.name for p in args.checkpoints.iterdir() if p.is_dir())

    def ckpt(name: str) -> Path | None:
        found = sorted((args.checkpoints / name).glob("weights_epoch_*.pt"))
        return found[-1] if found else None

    base_hashes = load_hashes(args.work, BASELINE)
    base_ckpt = ckpt(BASELINE)
    if not base_hashes or base_ckpt is None:
        raise RuntimeError("baseline A1 is missing hashes or a checkpoint")
    base_sd = state_dict(base_ckpt)

    report: dict[str, Any] = {
        "baseline": BASELINE, "n_batches_hashed": len(base_hashes),
        "runs": {}, "floor": None,
    }

    for name in names:
        h = load_hashes(args.work, name)
        c = ckpt(name)
        entry: dict[str, Any] = {
            "n_hashes": len(h),
            "batch_stream_identical": bool(h and h == base_hashes),
            "first_divergent_batch": next(
                (i for i, (x, y) in enumerate(zip(h, base_hashes)) if x != y),
                None) if h else None,
        }
        if c is not None and name != BASELINE:
            entry["weights_vs_A1"] = weight_delta(base_sd, state_dict(c))
        report["runs"][name] = entry

    floor = report["runs"].get(REPEAT, {}).get("weights_vs_A1", {})
    report["floor"] = floor.get("worst_relative")
    report["recipe_reproduces_bitwise"] = (report["floor"] == 0.0)
    if not report["runs"].get(REPEAT, {}).get("batch_stream_identical"):
        report["verdict"] = ("INVALID: the recipe does not even reproduce its "
                             "own batch stream; no knob can be judged")
        print(json.dumps(report, indent=2))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        return

    accepted = []
    for name, entry in report["runs"].items():
        if name in (BASELINE, REPEAT):
            continue
        w = entry.get("weights_vs_A1", {})
        ok = (entry["batch_stream_identical"] and w.get("comparable")
              and report["floor"] is not None
              and w.get("worst_relative", float("inf")) <= report["floor"])
        entry["accepted"] = bool(ok)
        entry["reason"] = (
            "batch stream differs" if not entry["batch_stream_identical"]
            else "weight difference exceeds the A1-vs-A2 floor"
            if not ok else "equivalent to the recipe")
        if ok:
            accepted.append(name)
    report["accepted"] = accepted
    report["verdict"] = (f"ACCEPTED: {accepted}" if accepted
                         else "NO KNOB ACCEPTED: run the recipe unchanged")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
