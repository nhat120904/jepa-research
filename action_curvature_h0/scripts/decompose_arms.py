#!/usr/bin/env python3
"""Why did cosine AS raise k_angular?  Decomposed from the existing dev shards.

No GPU: every term is recoverable in closed form from what the diagnostic
already stores.  With  span = ||v+ + v-||  and  ||D2|| = ||v+ - v-||,

    ||v+||^2 + ||v-||^2   = (span^2 + ||D2||^2) / 2
    ||v+|| ||v-|| cos     = (span^2 - ||D2||^2) / 4
    (||v+|| - ||v-||)^2   = ||D2||^2 * f_radial

which determines ||v+|| ||v-|| and hence cos.  Self-consistency is asserted
against the stored angular fraction via  2||v+||||v-||(1-cos) = ||D2||^2 f_ang.

Reading, per the locked order:
  - if (1-cos) does not fall with lambda, the AS gradient cannot steer its own
    objective;
  - if (1-cos) falls while k_angular rises, the training objective and the
    planner-relevant metric are different quantities, and the amendment writes
    itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-physical-spread-m", type=float, default=1e-4)
    return p.parse_args()


def terms(r: dict[str, Any]) -> dict[str, float] | None:
    d2 = float(r["d2_model_norm"])
    k = float(r["k_model_self"])
    if not np.isfinite(d2) or not np.isfinite(k) or k <= 0 or d2 <= 0:
        return None
    span = d2 / k
    f_r = float(np.clip(r["model_radial_fraction"], 0.0, 1.0))
    f_a = float(np.clip(r["model_angular_fraction"], 0.0, 1.0))

    sum_sq = (span**2 + d2**2) / 2.0                 # ||v+||^2 + ||v-||^2
    prod = (sum_sq - d2**2 * f_r) / 2.0              # ||v+|| ||v-||
    if prod <= 0:
        return None
    cos = ((span**2 - d2**2) / 4.0) / prod
    one_minus_cos = 1.0 - cos
    # self-consistency with the stored fraction
    resid = abs(2.0 * prod * one_minus_cos - d2**2 * f_a) / max(d2**2, 1e-30)
    return {"one_minus_cos": one_minus_cos, "norm_product": prod, "span": span,
            "d2": d2, "k_angular": k * np.sqrt(f_a), "consistency_resid": resid}


def main() -> None:
    args = parse_args()
    arms: dict[str, dict[str, float]] = {}
    worst_resid = 0.0
    for arm_dir in sorted(args.eval_root.iterdir()):
        if not arm_dir.is_dir():
            continue
        vals: list[dict[str, float]] = []
        for path in arm_dir.glob("snapshot_*/records.json"):
            for r in json.loads(path.read_text()):
                if (r.get("valid_unclipped")
                        and r.get("ordinal_physical_cost_spread", 0.0) >= args.min_physical_spread_m):
                    t = terms(r)
                    if t:
                        vals.append(t)
                        worst_resid = max(worst_resid, t["consistency_resid"])
        if vals:
            arms[arm_dir.name] = {k: float(np.median([v[k] for v in vals]))
                                  for k in vals[0]} | {"n": len(vals)}

    rows = []
    for name, m in arms.items():
        if not name.startswith("lam") or name.startswith("lam0_seed"):
            continue
        lam = float(name.split("_seed")[0][3:])
        seed = name.split("_seed")[1]
        ctrl = arms.get(f"lam0_seed{seed}")
        if ctrl is None or lam == 0.0:
            continue
        rows.append({
            "lambda": lam, "seed": int(seed),
            "d_one_minus_cos": m["one_minus_cos"] - ctrl["one_minus_cos"],
            "ratio_one_minus_cos": m["one_minus_cos"] / max(ctrl["one_minus_cos"], 1e-30),
            "ratio_norm_product": m["norm_product"] / max(ctrl["norm_product"], 1e-30),
            "ratio_span": m["span"] / max(ctrl["span"], 1e-30),
            "ratio_k_angular": m["k_angular"] / max(ctrl["k_angular"], 1e-30),
        })

    by_lam: dict[float, list[dict[str, Any]]] = {}
    for r in rows:
        by_lam.setdefault(r["lambda"], []).append(r)
    summary = [{
        "lambda": lam,
        "mean_ratio_one_minus_cos": float(np.mean([r["ratio_one_minus_cos"] for r in v])),
        "mean_ratio_norm_product": float(np.mean([r["ratio_norm_product"] for r in v])),
        "mean_ratio_span": float(np.mean([r["ratio_span"] for r in v])),
        "mean_ratio_k_angular": float(np.mean([r["ratio_k_angular"] for r in v])),
        "n_seeds": len(v),
    } for lam, v in sorted(by_lam.items())]

    cos_falls = all(s["mean_ratio_one_minus_cos"] < 1.0 for s in summary)
    ang_rises = all(s["mean_ratio_k_angular"] > 1.0 for s in summary)
    if cos_falls and ang_rises:
        verdict = "OBJECTIVE_MISMATCH_ANGLE_FALLS_BUT_NORMALIZED_CURVATURE_RISES"
    elif not cos_falls:
        verdict = "AS_GRADIENT_CANNOT_STEER_ITS_OWN_OBJECTIVE"
    else:
        verdict = "MIXED_SEE_PER_LAMBDA"

    report = {"per_arm": arms, "paired": rows, "by_lambda": summary,
              "max_consistency_residual": worst_resid, "verdict": verdict}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"by_lambda": summary,
                      "max_consistency_residual": worst_resid,
                      "verdict": verdict}, indent=2))


if __name__ == "__main__":
    main()
