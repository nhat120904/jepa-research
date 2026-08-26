#!/usr/bin/env python3
"""Gates B and C for the physical-space bridge, plus the metre-space primary.

Gate B (global):  median(||D2 e||) / median(||D2 s_true||) < threshold
Gate C (per-snapshot): at least `min_fraction` of NON-DEGENERATE snapshots
                       satisfy the same ratio of medians.

Ratio OF MEDIANS, never median of ratios: a handful of records with a
near-zero denominator would otherwise dominate a median of per-record ratios.

Degeneracy is decided before any gate is read: a record counts only if the
object actually moved (`object_span_m > --min-span`), and a snapshot counts
only if it retains at least `--min-records` such records.
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
    p.add_argument("--shard-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target", default="object", choices=["object", "effector"])
    p.add_argument("--threshold", type=float, default=0.25)
    p.add_argument("--min-fraction", type=float, default=0.75)
    p.add_argument("--min-span", type=float, default=1e-6,
                   help="metres; below this the object did not move and the "
                        "record carries no curvature signal")
    p.add_argument("--min-records", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tgt = args.target

    by_snapshot: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    total = kept = 0
    for path in sorted(args.shard_root.glob("*/snapshot_*/records.json")):
        source = path.parent.parent.name
        records = json.loads(path.read_text())
        summary = json.loads((path.parent / "summary.json").read_text())
        order = int(summary["snapshot"]["order"])
        for r in records:
            total += 1
            if not r.get("valid_unclipped"):
                continue
            if f"d2e_{tgt}_m" not in r:
                continue
            if float(r.get("object_span_m", 0.0)) <= args.min_span:
                continue
            by_snapshot[(source, order)].append(r)
            kept += 1

    if not by_snapshot:
        raise RuntimeError("no non-degenerate records with bridge fields found")

    def ratio_of_medians(rows: list[dict[str, Any]]) -> tuple[float, float, float]:
        e = float(np.median([r[f"d2e_{tgt}_m"] for r in rows]))
        s = float(np.median([r[f"d2_true_state_{tgt}_m"] for r in rows]))
        return e, s, (e / s if s > 0 else float("inf"))

    non_degenerate = {k: v for k, v in by_snapshot.items()
                      if len(v) >= args.min_records}
    per_snapshot = {}
    passing = 0
    for key, rows in sorted(non_degenerate.items()):
        e, s, ratio = ratio_of_medians(rows)
        ok = bool(ratio < args.threshold)
        passing += int(ok)
        per_snapshot[f"{key[0]}/snapshot_{key[1]:03d}"] = {
            "n_records": len(rows), "median_d2e_m": e,
            "median_d2_true_m": s, "ratio": ratio, "pass": ok,
        }

    pooled = [r for rows in non_degenerate.values() for r in rows]
    e_all, s_all, ratio_all = ratio_of_medians(pooled)
    gate_b = bool(ratio_all < args.threshold)
    frac = passing / len(non_degenerate) if non_degenerate else 0.0
    gate_c = bool(frac >= args.min_fraction)

    mism = float(np.median([r[f"d2_mismatch_{tgt}_m"] for r in pooled]))
    model = float(np.median([r[f"d2_model_decoded_{tgt}_m"] for r in pooled]))
    centre_err = float(np.median([r[f"bridge_centre_err_{tgt}_m"] for r in pooled]))

    report = {
        "target": tgt,
        "threshold": args.threshold,
        "min_fraction": args.min_fraction,
        "degeneracy": {"min_span_m": args.min_span,
                       "min_records_per_snapshot": args.min_records},
        "counts": {"records_seen": total, "records_kept": kept,
                   "snapshots_with_records": len(by_snapshot),
                   "snapshots_non_degenerate": len(non_degenerate)},
        "gate_b": {"median_d2e_m": e_all, "median_d2_true_m": s_all,
                   "ratio": ratio_all, "pass": gate_b},
        "gate_c": {"snapshots_passing": passing,
                   "snapshots_considered": len(non_degenerate),
                   "fraction": frac, "pass": gate_c},
        "per_snapshot": per_snapshot,
        "primary_if_gates_pass": {
            "median_d2_mismatch_m": mism,
            "median_d2_model_decoded_m": model,
            "median_d2_true_m": s_all,
            "median_bridge_centre_err_m": centre_err,
        },
        "verdict": ("BRIDGE_USABLE" if (gate_b and gate_c)
                    else "BRIDGE_FAILS_DO_NOT_SUBSTITUTE_NONLINEAR_PROBE"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("counts", "gate_b", "gate_c", "primary_if_gates_pass",
                       "verdict")}, indent=2))


if __name__ == "__main__":
    main()
