"""CTA closed-loop aggregation (docs/CTA_E2E_PROTOCOL.md). CPU job.

Paired by root: every arm vs P0, CTA8 vs DIRECT8 and vs FULL8, retention of CTA8 against FULL8 (and PHYS8 if run).
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ti_wm.contract import require_compute, select_candidate
from ti_wm.gates import mcnemar_exact, paired_diff, retention
from ti_wm.wb import Logger


def decision_stats(recs, arm):
    rho, agree, gain, secs = [], [], [], []
    for r in recs:
        for d in r[arm]["decisions"]:
            if "phys" not in d or "score" not in d:
                continue
            x, y = np.asarray(d["score"], float), np.asarray(d["phys"], float)
            gain.append(y[d["chosen"]] - y[0])
            if "score_seconds" in d:
                secs.append(d["score_seconds"])
            if np.ptp(y) > 0:
                agree.append(select_candidate(list(x)) == select_candidate(list(y)))
                rho.append(spearmanr(x, y).statistic if np.ptp(x) > 0 else 0.0)
    mean = lambda v: float(np.mean(v)) if v else float("nan")
    return {"within_bank_spearman_vs_cov8": mean(rho), "choice_agreement_with_phys": mean(agree),
            "mean_cov8_gain_of_choice": mean(gain), "score_seconds_per_decision": mean(secs), "decisions": len(gain)}


def main(runs, out, first, count):
    require_compute()
    records = {}
    for path in sorted(p for run in runs for p in Path(run).glob("roots_*.jsonl")):
        for line in path.read_text().splitlines():
            r = json.loads(line)
            assert r["root"] not in records, r["root"]
            records[r["root"]] = r
    roots = list(range(first, first + count))
    missing = sorted(set(roots) - set(records))
    assert not missing, f"missing roots {missing}"
    recs = [records[r] for r in roots]
    assert len({r["checkpoint_sha256"] for r in recs}) == 1
    arms = recs[0]["arms"]
    s = {a: np.array([r[a]["success"] for r in recs], dtype=float) for a in arms}
    summary = {"roots": [first, first + count - 1], "arms": arms, "checkpoint_sha256": recs[0]["checkpoint_sha256"],
               "success_rate": {a: float(v.mean()) for a, v in s.items()}, "diff": {}, "mcnemar": {}}
    pairs = [(a, "P0") for a in arms if a != "P0"] + [(a, b) for a, b in (("CTA8", "DIRECT8"), ("CTA8", "FULL8"),
                                                                         ("CTA8", "CODE8"), ("CODE8", "FULL8"))
                                                   if a in s and b in s]
    for a, b in pairs:
        summary["diff"][f"{a}-{b}"] = paired_diff(s[a], s[b])
        summary["mcnemar"][f"{a}-{b}"] = mcnemar_exact(s[a], s[b])
    for ref in ("FULL8", "PHYS8"):
        if ref in s and "CTA8" in s and "P0" in s:
            summary[f"retention_cta8_vs_{ref.lower()}"] = retention(s["CTA8"], s[ref], s["P0"])
    summary["decisions"] = {a: decision_stats(recs, a) for a in arms if a != "P0"}
    summary["seconds_per_root"] = float(np.mean([r["seconds"] for r in recs]))
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))
    log = Logger(out, "aggregate", {"runs": [str(r) for r in runs], "first": first, "count": count},
                 name=f"agg_{os.environ.get('SLURM_JOB_ID')}_{first}")
    log.summary(summary)
    log.table("closed_loop/diffs", ("contrast", "mean", "lo", "hi", "n", "mcnemar_p"),
              [(k, v["mean"], v["lo"], v["hi"], v["n"], summary["mcnemar"][k]["p"]) for k, v in summary["diff"].items()])
    log.table("closed_loop/arms", ("arm", "success", "spearman_vs_cov8", "agree_phys", "cov8_gain", "score_seconds"),
              [(a, summary["success_rate"][a], *[summary["decisions"].get(a, {}).get(k, float("nan")) for k in
                ("within_bank_spearman_vs_cov8", "choice_agreement_with_phys", "mean_cov8_gain_of_choice",
                 "score_seconds_per_decision")]) for a in arms])
    log.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--first", type=int, required=True)
    p.add_argument("--count", type=int, required=True)
    a = p.parse_args()
    main(a.run, a.out, a.first, a.count)
