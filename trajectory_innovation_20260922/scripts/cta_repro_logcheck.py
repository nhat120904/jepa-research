"""Compare preserved proposal-reproduction decision logs on a Slurm CPU node."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def main(args):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this array analysis through sbatch")
    report = {"job": os.environ["SLURM_JOB_ID"], "source": str(args.source), "root": args.root}
    for mode in ("old", "canonical"):
        with np.load(args.source / f"{mode}_1.npz") as left, np.load(args.source / f"{mode}_2.npz") as right:
            result = {}
            for key in left.files:
                a, b = left[key][left["root"] == args.root], right[key][right["root"] == args.root]
                row = {"shapes": [list(a.shape), list(b.shape)], "dtype": str(a.dtype)}
                if a.shape == b.shape:
                    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
                    row.update(bitwise_equal=bool(np.array_equal(a, b)),
                               nan_equal=bool(np.array_equal(a, b, equal_nan=True)),
                               max_abs=float(np.nanmax(delta)),
                               unequal=int(np.count_nonzero(a != b)),
                               close_1e12=bool(np.allclose(a, b, rtol=0, atol=1e-12, equal_nan=True)))
                    indices = np.argwhere(a != b)[:8]
                    row["examples"] = [{"index": i.tolist(), "left": float(a[tuple(i)]),
                                        "right": float(b[tuple(i)])} for i in indices]
                result[key] = row
            report[mode] = result
    source_report = json.loads((args.source / "reproduction.json").read_text())
    checks = ("canonical_initial", "canonical_nested_8_16", "canonical_nested_1_8")
    report["proposal_prefix_checks"] = {key: source_report[key] for key in checks}
    report["verification_contract"] = {
        "exact_log_fields": [key for key in report["canonical"] if key != "cov"],
        "coverage_atol": 1e-12,
        "coverage_rtol": 0,
        "scope": "initial K=1/8/16 prefixes; root 2205 P0 logged decisions alone versus with root 2206",
        "raw_gpu_job": "56693 FAILED on a strict bitwise coverage assertion; not relabelled",
    }
    report["bounded_contract_pass"] = (
        all(value["bitwise_equal"] for value in report["proposal_prefix_checks"].values())
        and all(value["bitwise_equal"] for key, value in report["canonical"].items() if key != "cov")
        and report["canonical"]["cov"]["close_1e12"]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=int, default=2205)
    main(parser.parse_args())
