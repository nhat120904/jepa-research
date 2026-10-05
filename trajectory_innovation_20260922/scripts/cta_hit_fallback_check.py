"""Check step-zero fallback equivalence from completed pilot outputs only.

Compute-node CPU work. No checkpoints or models are loaded. Checkpoint config
records establish the declared step-zero relationship; observed cross-scores,
trajectories and outcomes test that relationship. This is not an independent
attestation that the stored tensor weights are identical.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cta_hit_aggregate import load_runs
from ti_wm.contract import require_compute


PREFIX = {"cta": "CTA", "endpoint": "END", "direct": "DIR"}
EXACT_FIELDS = ("root", "decision", "t", "chosen", "geom", "native_hit")
EPISODE_EXACT = ("root", "success", "steps", "terminated", "truncated_by_max_decisions", "reset_excluded")
EPISODE_FLOAT = ("max_coverage", "score")
ATOL = 1e-12


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(f"{value.dtype}:{value.shape}\n".encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def array_check(left, right, tolerance=False):
    left, right = np.asarray(left), np.asarray(right)
    same_shape = left.shape == right.shape
    hashes = {"left_sha256": array_sha256(left), "right_sha256": array_sha256(right)}
    bitwise = same_shape and left.dtype == right.dtype and hashes["left_sha256"] == hashes["right_sha256"]
    difference = float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64)))) if same_shape and left.size else None
    passed = (same_shape and bool(np.allclose(left, right, atol=ATOL, rtol=0))) if tolerance else bitwise
    return {"pass": passed, "bitwise_equal": bitwise, "same_shape": same_shape,
            "max_absolute_difference": difference, "atol": ATOL if tolerance else 0.,
            "rtol": 0., **hashes}


def episode_check(left, right, roots):
    exact = {key: all(left[r][key] == right[r][key] for r in roots) for key in EPISODE_EXACT}
    floating = {key: array_check([left[r][key] for r in roots], [right[r][key] for r in roots], tolerance=True)
                for key in EPISODE_FLOAT}
    failures = [r for r in roots if any(left[r][key] != right[r][key] for key in EPISODE_EXACT)
                or any(abs(left[r][key] - right[r][key]) > ATOL for key in EPISODE_FLOAT)]
    return {"pass": all(exact.values()) and all(value["pass"] for value in floating.values()),
            "exact_fields": exact, "floating_fields": floating, "failed_roots": failures, "roots": len(roots)}


def relationships(experiment):
    """Find BASE aliases among the conventional CTRL and HIT checkpoints."""
    checkpoints, specs = experiment["checkpoints"], experiment["checkpoint_specs"]
    base = checkpoints.get("BASE")
    out = []
    for label, suffix in (("CTRL", "control"), ("HIT", "hit")):
        if label not in checkpoints:
            continue
        continuation = checkpoints[label]["config"].get("continuation", {})
        selected = continuation.get("selected_steps", {})
        for kind, prefix in PREFIX.items():
            alias = f"{prefix}_{label}"
            if alias not in specs:
                continue
            key = f"{kind}_{suffix}"
            if key not in selected:
                raise ValueError(f"Missing selected-step metadata {label}/{key}")
            step = selected[key]
            if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                raise ValueError(f"Invalid selected step {label}/{key}: {step!r}")
            if step != 0:
                continue
            baseline = f"{prefix}_BASE"
            out.append({"alias": alias, "baseline": baseline, "family": kind, "checkpoint": label,
                        "selected_step": 0, "fallback_reused_base": True,
                        "base_sha256_matches": bool(base and continuation.get("base_sha256") == base["sha256"]),
                        "source_reader_frozen_declared": continuation.get("source_reader_frozen") is True,
                        "alias_spec_matches": tuple(specs[alias]) == (label, kind),
                        "baseline_spec_matches": baseline in specs and tuple(specs[baseline]) == ("BASE", kind),
                        "alias_whole_checkpoint_sha256": checkpoints[label]["sha256"],
                        "baseline_whole_checkpoint_sha256": base["sha256"] if base else None})
    return out


def check(closed_runs, out):
    require_compute()
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    roots, episodes, logs, shards, experiment = load_runs(closed_runs)
    aliases = relationships(experiment)
    report = {"schema": "cta_hit_fallback_check_v1", "job": os.environ["SLURM_JOB_ID"],
              "source_shards": [value["path"] for value in shards], "roots": roots,
              "arms": experiment["arms"], "relationships": [],
              "claim_scope": "Declared step-zero provenance plus observed score/trajectory/outcome equivalence; no checkpoint tensors loaded",
              "weight_equality_independently_attested": False,
              "contract": {"cross_scores": "dtype/shape/content SHA256 equality on every acting arm's states",
                           "trajectory_exact": list(EXACT_FIELDS), "coverage_atol": ATOL, "rtol": 0.,
                           "episode_exact": list(EPISODE_EXACT), "episode_atol": list(EPISODE_FLOAT)},
              "notes": ["Whole checkpoint hashes may differ because configs and other model states differ.",
                        "A step-zero fallback is the original deployed model, not a learned improvement.",
                        "Equivalence on this development pilot is not a universal hardware/backend guarantee."]}
    for declared in aliases:
        left, right = declared["alias"], declared["baseline"]
        checks = {key: declared[key] for key in ("base_sha256_matches", "source_reader_frozen_declared",
                                                "alias_spec_matches", "baseline_spec_matches")}
        checks["acting_arms_present"] = left in episodes and right in episodes
        checks["crossscore_columns_present"] = left in experiment["log_scorers"] and right in experiment["log_scorers"]
        result = {**declared, "provenance": checks}
        if all(checks.values()):
            result["cross_scores"] = {arm: array_check(log[f"score_{left}"], log[f"score_{right}"])
                                      for arm, log in logs.items()}
            left_order = np.lexsort((logs[left]["decision"], logs[left]["root"]))
            right_order = np.lexsort((logs[right]["decision"], logs[right]["root"]))
            result["trajectory"] = {key: array_check(logs[left][key][left_order], logs[right][key][right_order])
                                    for key in EXACT_FIELDS}
            result["trajectory"]["cov"] = array_check(logs[left]["cov"][left_order], logs[right]["cov"][right_order],
                                                        tolerance=True)
            result["episodes"] = episode_check(episodes[left], episodes[right], roots)
            passed = (all(value["pass"] for value in result["cross_scores"].values())
                      and all(value["pass"] for value in result["trajectory"].values())
                      and result["episodes"]["pass"])
        else:
            passed = False
        result["status"] = "PASS" if passed else "FAIL"
        report["relationships"].append(result)
    report["status"] = ("NOT_APPLICABLE" if not aliases else "PASS"
                        if all(value["status"] == "PASS" for value in report["relationships"]) else "FAIL")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(out), "status": report["status"],
                      "aliases": {item["alias"]: item["status"] for item in report["relationships"]}}), flush=True)
    if report["status"] == "FAIL":
        raise RuntimeError(f"Step-zero fallback equivalence failed; inspect {out}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed-run", type=Path, nargs="+", required=True, help="Completed shard result directories")
    parser.add_argument("--out", type=Path, required=True, help="A distinct output JSON file")
    args = parser.parse_args()
    check(args.closed_run, args.out)
