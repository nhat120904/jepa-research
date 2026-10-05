"""Bounded GPU investigation of proposal batch-size dependence; no training or sealed roots.

The production canonical adapter is opt-in and does not change historical runs or checkpoints.
"""
import argparse
import functools
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ti_wm.contract import candidate_seed, require_compute
from ti_wm.cta_batch import K, run_arm
from ti_wm.cta_runtime import keep_steps, run_segment
from ti_wm.pusht_runtime import CLONERS, PolicyRunner, done, reset_branch
from ti_wm.pusht_canonical import CanonicalPolicyRunner


TI = Path("/mnt/data/nhatnc129/jepa/trajectory_innovation")


def diff(a, b):
    return {"bitwise_equal": bool(np.array_equal(a, b)), "max_abs": float(np.abs(a - b).max())}


def main(args):
    require_compute()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    smoke = json.loads((TI / "gate_smoke_53803/smoke.json").read_text())
    runner = PolicyRunner(TI / "prepare_53776/checkpoint", "cuda", n_exec=15)
    canonical = CanonicalPolicyRunner(TI / "prepare_53776/checkpoint", "cuda", n_exec=15)
    state = reset_branch(2205)
    seed = [candidate_seed(2205, 0, k) for k in range(K)]
    report = {"job": os.environ["SLURM_JOB_ID"], "scope": "development root 2205, native K=8, L=15",
              "backend": {"cudnn_deterministic": torch.backends.cudnn.deterministic,
                          "cudnn_benchmark": torch.backends.cudnn.benchmark,
                          "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "cudnn_tf32": torch.backends.cudnn.allow_tf32}, "initial_banks": {}}
    save = lambda: (args.out / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    first = runner.draw([state.hist] * K, seed)
    for copies in (1, 4, 25, 100):
        bank = runner.draw([state.hist] * (K * copies), seed * copies)[:K]
        report["initial_banks"][str(copies)] = diff(first, bank)
        save()
    cfirst = canonical.draw([state.hist] * K, seed)
    cother = canonical.draw([state.hist] * (K * 4), seed * 4)[:K]
    report["canonical_initial"] = diff(cfirst, cother)
    report["canonical_vs_old"] = diff(cfirst, first)
    nested = canonical.bank(state.hist, [candidate_seed(2205, 0, k) for k in range(16)])
    singleton = canonical.bank(state.hist, seed[:1])
    report["canonical_nested_8_16"] = diff(cfirst, nested[:K])
    report["canonical_nested_1_8"] = diff(singleton, cfirst[:1])
    state.env.close()
    save()

    segment = functools.partial(run_segment, keep=keep_steps(15))
    cloner = CLONERS[smoke["clone_method"]]
    report["episodes"] = {}
    trajectories = {}
    # P0 for one target root, alone versus accompanied by another root. No learned scorer is involved.
    for name, policy in (("old", runner), ("canonical", canonical)):
        for roots in ([2205], [2205, 2206]):
            eps, arrays, timing = run_arm("P0", roots, policy, cloner, None, [], reset_branch, done, segment)
            tag = f"{name}_{len(roots)}"
            report["episodes"][tag] = eps
            trajectories[tag] = {k: v[arrays["root"] == 2205] for k, v in arrays.items()}
            np.savez(args.out / f"{tag}.npz", **arrays)
            save()
            print(tag, eps, flush=True)
    report["canonical_root2205_episode_equal"] = report["episodes"]["canonical_1"][0] == report["episodes"]["canonical_2"][0]
    report["old_root2205_episode_equal"] = report["episodes"]["old_1"][0] == report["episodes"]["old_2"][0]
    left, right = trajectories["canonical_1"], trajectories["canonical_2"]
    report["canonical_log_fields"] = {k: diff(v, right[k]) for k, v in left.items()}
    report["canonical_log_contract"] = {
        "exact_fields": [k for k in left if k != "cov"],
        # Native polygon coverage can differ by a float64 ULP with identical geometry.
        # The archived 56693 logs were separately diagnosed by CPU job 56705.
        "coverage_atol": 1e-12,
        "coverage_rtol": 0,
    }
    report["canonical_logged_decisions_equal"] = (
        all(np.array_equal(v, right[k]) for k, v in left.items() if k != "cov")
        and bool(np.allclose(left["cov"], right["cov"], atol=1e-12, rtol=0))
    )
    checks = (report["canonical_initial"], report["canonical_nested_8_16"], report["canonical_nested_1_8"])
    save()
    if not all(c["bitwise_equal"] for c in checks) or not report["canonical_logged_decisions_equal"]:
        raise RuntimeError("Canonical proposal contract failed")
    report["status"] = "DONE"
    save()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args())
