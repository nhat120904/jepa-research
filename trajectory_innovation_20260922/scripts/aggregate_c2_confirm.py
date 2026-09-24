"""Gate C2-confirm aggregation and pre-registered verdicts (docs/GATE_C2_CONFIRM_PROTOCOL.md). CPU job."""

import argparse
import json
from pathlib import Path

import numpy as np

from ti_wm.contract import require_compute
from ti_wm.gates import mcnemar_exact, paired_diff, verdict_confirm

ARMS = ("P0", "PROG8_r0", "PROG8_r1")
R1_MIN_SPEARMAN = 0.70


def load(run, roots, arms):
    records = {}
    for path in sorted(Path(run).glob("roots_*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            assert r["root"] not in records
            records[r["root"]] = r
    missing = sorted(set(roots) - set(records))
    assert not missing, f"missing roots {missing}"
    return {a: np.array([records[r][a]["success"] for r in roots], dtype=float) for a in arms}


def main(run, r1, c2_run, out, first, count):
    require_compute()
    s = load(run, list(range(first, first + count)), ARMS)
    r1_report = json.loads((r1 / "train_report.json").read_text())
    r1_rho = r1_report["val"]["full"]["spearman"]
    primary = paired_diff(s["PROG8_r0"], s["P0"])
    secondary = paired_diff(s["PROG8_r1"], s["P0"])
    c2 = load(c2_run, list(range(1200, 1300)), ("P0", "PROG8"))
    summary = {
        "roots": [first, first + count - 1],
        "success_rate": {a: float(v.mean()) for a, v in s.items()},
        "primary_prog8_r0_minus_p0": primary,
        "primary_mcnemar": mcnemar_exact(s["PROG8_r0"], s["P0"]),
        "verdict": verdict_confirm(primary),
        "r1_val_spearman": r1_rho,
        "r1_training": "OK" if r1_rho >= R1_MIN_SPEARMAN else "r1 training failed",
        "seed_robustness_prog8_r1_minus_p0": secondary,
        "seed_robustness_mcnemar": mcnemar_exact(s["PROG8_r1"], s["P0"]),
        "seed_robustness": "ROBUST" if verdict_confirm(secondary) == "CONFIRMED" else "NOT_ROBUST",
        "pooled_r0_c2_plus_confirm_not_a_test": paired_diff(np.concatenate([c2["PROG8"], s["PROG8_r0"]]),
                                                            np.concatenate([c2["P0"], s["P0"]])),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("run", "r1", "c2_run", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--first", type=int, default=1300)
    parser.add_argument("--count", type=int, default=200)
    a = parser.parse_args()
    main(a.run, a.r1, a.c2_run, a.out, a.first, a.count)
