#!/usr/bin/env python3
"""Apply the locked lambda-selection rule to the dev-eval shards.

Rule, in order, each lambda compared against its PAIRED lambda=0 at the same
seed (twelfth amendment F.2, dev protocol G):

  1. must lower angular curvature
  2. reject if base rollout loss is >10% worse
  3. reject if action sensitivity falls >20%
  4. among survivors take the lowest false-valley rate; ties to smaller lambda

Everything is read from the fixed diagnostic manifest, never from training logs,
because each training log point sits at a different sampled sigma and direction
and is not comparable across arms.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", type=Path, required=True)
    p.add_argument("--train-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-physical-spread-m", type=float, default=1e-4)
    p.add_argument("--max-base-loss-degradation", type=float, default=0.10)
    p.add_argument("--max-sensitivity-drop", type=float, default=0.20)
    return p.parse_args()


def arm_metrics(arm_dir: Path, min_spread: float) -> dict[str, Any] | None:
    rows: list[dict[str, Any]] = []
    for path in sorted(arm_dir.glob("snapshot_*/records.json")):
        rows.extend(json.loads(path.read_text()))
    keep = [r for r in rows if r.get("valid_unclipped")
            and r.get("ordinal_physical_cost_spread", 0.0) >= min_spread]
    if not keep:
        return None
    ang = [r["k_model_self"] * np.sqrt(max(r["model_angular_fraction"], 0.0))
           for r in keep]
    return {
        "n_records": len(rows), "n_usable": len(keep),
        "angular_curvature": float(np.median(ang)),
        "false_valley": float(np.mean([bool(r["ordinal_false_valley"]) for r in keep])),
        "action_sensitivity": float(np.median([r["s_model"] for r in keep])),
    }


def main() -> None:
    args = parse_args()
    arms: dict[str, dict[str, Any]] = {}
    for arm_dir in sorted(args.eval_root.iterdir()):
        if not arm_dir.is_dir():
            continue
        m = arm_metrics(arm_dir, args.min_physical_spread_m)
        if m is None:
            continue
        log = args.train_root / arm_dir.name / "train_log.json"
        if log.exists():
            hist = json.loads(log.read_text())["history"]
            # Final-window mean, not the last point, which is one noisy batch.
            tail = hist[-5:] if len(hist) >= 5 else hist
            m["base_loss"] = float(np.mean([h["base_loss"] for h in tail]))
        arms[arm_dir.name] = m

    paired: list[dict[str, Any]] = []
    for name, m in arms.items():
        if name == "original" or not name.startswith("lam"):
            continue
        lam_str, seed_str = name.split("_seed")
        lam = float(lam_str[3:])
        if lam == 0.0:
            continue
        ctrl = arms.get(f"lam0_seed{seed_str}")
        if ctrl is None:
            continue
        d = {
            "arm": name, "lambda": lam, "seed": int(seed_str),
            "angular_curvature": m["angular_curvature"],
            "angular_curvature_ctrl": ctrl["angular_curvature"],
            "angular_delta": m["angular_curvature"] - ctrl["angular_curvature"],
            "false_valley": m["false_valley"],
            "false_valley_ctrl": ctrl["false_valley"],
            "sensitivity_ratio": m["action_sensitivity"] / max(ctrl["action_sensitivity"], 1e-12),
        }
        if "base_loss" in m and "base_loss" in ctrl:
            d["base_loss_ratio"] = m["base_loss"] / max(ctrl["base_loss"], 1e-12)
        paired.append(d)

    by_lambda: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for d in paired:
        by_lambda[d["lambda"]].append(d)

    verdicts = []
    for lam in sorted(by_lambda):
        rows = by_lambda[lam]
        ang_ok = all(r["angular_delta"] < 0 for r in rows)
        base_ok = all(r.get("base_loss_ratio", 1.0)
                      <= 1.0 + args.max_base_loss_degradation for r in rows)
        sens_ok = all(r["sensitivity_ratio"]
                      >= 1.0 - args.max_sensitivity_drop for r in rows)
        verdicts.append({
            "lambda": lam, "n_seeds": len(rows),
            "mean_angular_delta": float(np.mean([r["angular_delta"] for r in rows])),
            "mean_false_valley": float(np.mean([r["false_valley"] for r in rows])),
            "mean_false_valley_ctrl": float(np.mean([r["false_valley_ctrl"] for r in rows])),
            "mean_base_loss_ratio": float(np.mean([r.get("base_loss_ratio", float("nan"))
                                                   for r in rows])),
            "mean_sensitivity_ratio": float(np.mean([r["sensitivity_ratio"] for r in rows])),
            "gate1_lowers_angular": ang_ok,
            "gate2_base_loss_ok": base_ok,
            "gate3_sensitivity_ok": sens_ok,
            "survives": bool(ang_ok and base_ok and sens_ok),
        })

    survivors = [v for v in verdicts if v["survives"]]
    if survivors:
        best = min(survivors, key=lambda v: (v["mean_false_valley"], v["lambda"]))
        chosen: Any = best["lambda"]
        decision = "LAMBDA_SELECTED"
    else:
        chosen, decision = None, "NO_LAMBDA_SURVIVES_GATES"

    report = {"arms": arms, "paired": paired, "by_lambda": verdicts,
              "chosen_lambda": chosen, "decision": decision,
              "thresholds": {
                  "max_base_loss_degradation": args.max_base_loss_degradation,
                  "max_sensitivity_drop": args.max_sensitivity_drop,
                  "min_physical_spread_m": args.min_physical_spread_m}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"by_lambda": verdicts, "chosen_lambda": chosen,
                      "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
