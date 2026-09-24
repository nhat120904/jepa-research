"""Week-2 aggregation and pre-registered verdict (docs/W2_CLOSED_LOOP_PROTOCOL.md). CPU job."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ti_wm.contract import require_compute, select_candidate
from ti_wm.gates import mcnemar_exact, paired_diff, retention

ARMS = ("P0", "PHYS8", "RANK8")


def main(runs, out, first, count):
    require_compute()
    records = {}
    for path in sorted(p for run in runs for p in Path(run).glob("roots_*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            assert r["root"] not in records
            records[r["root"]] = r
    roots = list(range(first, first + count))
    missing = sorted(set(roots) - set(records))
    assert not missing, f"missing roots {missing}"
    recs = [records[r] for r in roots]
    assert len({r["reader_sha256"] for r in recs}) == 1
    s = {a: np.array([r[a]["success"] for r in recs], dtype=float) for a in ARMS}
    primary = paired_diff(s["RANK8"], s["P0"])
    phys = paired_diff(s["PHYS8"], s["P0"])
    ret = retention(s["RANK8"], s["PHYS8"], s["P0"])
    rhos, agree = [], []
    for r in recs:
        for d in r["RANK8"]["decisions"]:
            x, y = np.asarray(d["rank"], float), np.asarray(d["phys"], float)
            if np.ptp(y) > 0:
                agree.append(select_candidate(list(x)) == select_candidate(list(y)))
                if np.ptp(x) > 0:
                    rhos.append(spearmanr(x, y).statistic)
    summary = {
        "roots": [first, first + count - 1],
        "success_rate": {a: float(v.mean()) for a, v in s.items()},
        "primary_rank8_minus_p0": primary,
        "mcnemar_rank8_p0": mcnemar_exact(s["RANK8"], s["P0"]),
        "verdict": "PASS" if primary["lo"] > 0 else "FAIL",
        "phys8_minus_p0": phys,
        "mcnemar_phys8_p0": mcnemar_exact(s["PHYS8"], s["P0"]),
        "rank8_minus_phys8": paired_diff(s["RANK8"], s["PHYS8"]),
        "retention": ret,
        "strong": bool(primary["lo"] > 0 and phys["lo"] > 0 and ret["ratio"] >= 0.80),
        "within_bank_spearman_rank_cov8": {"mean": float(np.mean(rhos)), "n": len(rhos)},
        "choice_agreement_with_phys8": {"mean": float(np.mean(agree)), "n": len(agree)},
        "preflight": recs[0]["preflight"],
        "seconds_per_root": float(np.mean([r["seconds"] for r in recs])),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, default=1600)
    parser.add_argument("--count", type=int, default=400)
    a = parser.parse_args()
    main(a.run, a.out, a.first, a.count)
