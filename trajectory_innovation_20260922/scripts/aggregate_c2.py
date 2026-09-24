"""Gate C2 aggregation and pre-registered verdicts (docs/GATE_C2_PROGRESS_READER_PROTOCOL.md). CPU job."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ti_wm.contract import require_compute, select_candidate
from ti_wm.gates import cluster_ratio, mcnemar_exact, paired_diff, retention, verdict_b, verdict_c

ARMS = ("P0", "PHYS8", "PROG8")


def within_bank(records, a, b):
    rhos = []
    for r in records:
        for d in r["P0"]["decisions"]:
            x, y = np.asarray(d[a], float), np.asarray(d[b], float)
            if np.ptp(x) > 0 and np.ptp(y) > 0:
                rhos.append(spearmanr(x, y).statistic)
    return {"mean": float(np.mean(rhos)) if rhos else float("nan"), "n_decisions": len(rhos)}


def main(run, trained, out, first, count):
    require_compute()
    records = {}
    for path in sorted(Path(run).glob("roots_*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            assert r["root"] not in records
            records[r["root"]] = r
    roots = list(range(first, first + count))
    missing = sorted(set(roots) - set(records))
    assert not missing, f"missing roots {missing}"
    records = [records[r] for r in roots]
    success = {a: np.array([r[a]["success"] for r in records], dtype=float) for a in ARMS}

    b_diff = paired_diff(success["PHYS8"], success["P0"])
    b_verdict = verdict_b(b_diff)
    ret = retention(success["PROG8"], success["PHYS8"], success["P0"])
    c2 = verdict_c("PASS" if b_verdict == "PASS" else "NOT_REPLICATED", ret)

    prog_gap, best_gap = [], []
    for r in records:
        pg = bg = 0.0
        for d in r["P0"]["decisions"]:
            phys = np.asarray(d["phys"])
            pg += phys[select_candidate(d["prog"])] - phys[0]
            bg += phys.max() - phys[0]
        prog_gap.append(pg)
        best_gap.append(bg)

    summary = {
        "roots": [first, first + count - 1],
        "reader_training": json.loads((trained / "train_report.json").read_text()),
        "success_rate": {a: float(v.mean()) for a, v in success.items()},
        "B_replication": {"phys8_minus_p0": b_diff, "mcnemar": mcnemar_exact(success["PHYS8"], success["P0"]),
                          "verdict": b_verdict},
        "C2": {"prog8_minus_p0": paired_diff(success["PROG8"], success["P0"]),
               "prog8_minus_phys8": paired_diff(success["PROG8"], success["PHYS8"]),
               "mcnemar_prog8_p0": mcnemar_exact(success["PROG8"], success["P0"]),
               "retention": ret, "verdict": c2 if b_verdict == "PASS" else "NOT_INTERPRETED_B_NOT_REPLICATED",
               "offline_retained_gap": cluster_ratio(prog_gap, best_gap),
               "within_bank_spearman_prog_phys": within_bank(records, "prog", "phys")},
        "diagnostic_gate_c_failure": {
            f"spearman_{score}_{target}": within_bank(records, score, target)
            for score in ("vis", "prog") for target in ("agent_disp", "block_disp", "block_rot", "phys")},
        "seconds_per_root": float(np.mean([r["seconds"] for r in records])),
    }
    summary["reader_training"].pop("error", None)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, default=1200)
    parser.add_argument("--count", type=int, default=100)
    a = parser.parse_args()
    main(a.run, a.trained, a.out, a.first, a.count)
