"""Aggregate gate A-C shards and apply the pre-registered verdict rules. CPU job."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ti_wm.arms import K_SELECT, hrep_estimates
from ti_wm.contract import require_compute, select_candidate
from ti_wm.gates import (
    cluster_ratio, mcnemar_exact, paired_diff, retention, runtime_blocker, verdict_a, verdict_b, verdict_c,
)

ARMS = ("OFFICIAL", "P0", "PHYS8", "VIS8", "MEDOID8")


def load(runs, roots):
    records = {}
    for run in runs:
        for path in sorted(Path(run).glob("roots_*.jsonl")):
            for line in path.read_text().splitlines():
                r = json.loads(line)
                assert r["root"] not in records, f"duplicate root {r['root']}"
                records[r["root"]] = r
    missing = sorted(set(roots) - set(records))
    assert not missing, f"missing roots: {missing[:10]} ({len(missing)})"
    return [records[r] for r in roots]


def offline_p0(records):
    """Myopic headroom (K=8/16/32) and visual alignment on P0's own decisions."""
    per_root = {"gain8": [], "gain16": [], "gain32": [], "vis_gap": [], "best_gap": []}
    rhos, strict = [], []
    for r in records:
        g = {8: 0.0, 16: 0.0, 32: 0.0}
        vis_gap = best_gap = 0.0
        for d in r["P0"]["decisions"]:
            phys, vis = np.asarray(d["phys"]), np.asarray(d["vis"])
            for k in g:
                g[k] += phys[:k].max() - phys[0]
            strict.append(phys[:K_SELECT].max() > phys[0])
            vis_gap += phys[select_candidate(vis[:K_SELECT].tolist())] - phys[0]
            best_gap += phys[:K_SELECT].max() - phys[0]
            p8, v8 = phys[:K_SELECT], vis[:K_SELECT]
            if np.ptp(p8) > 0 and np.ptp(v8) > 0:
                rhos.append(spearmanr(v8, p8).statistic)
        n = max(len(r["P0"]["decisions"]), 1)
        for k in g:
            per_root[f"gain{k}"].append(g[k] / n)
        per_root["vis_gap"].append(vis_gap)
        per_root["best_gap"].append(best_gap)
    zeros = np.zeros(len(records))
    return {
        **{f"myopic_coverage_gain_k{k}": paired_diff(per_root[f"gain{k}"], zeros) for k in (8, 16, 32)},
        "decisions_with_strictly_better_candidate_k8": float(np.mean(strict)),
        "offline_retained_gap": cluster_ratio(per_root["vis_gap"], per_root["best_gap"]),
        "within_bank_spearman_vis_phys": {"mean": float(np.mean(rhos)) if rhos else float("nan"),
                                          "n_decisions": len(rhos)},
    }


def main(runs, prep, out, first, count):
    require_compute()
    roots = list(range(first, first + count))
    records = load(runs, roots)
    success = {a: np.array([r[a]["success"] for r in records], dtype=float) for a in ARMS}
    coverage = {a: float(np.mean([r[a]["max_coverage"] for r in records])) for a in ARMS}

    published = {e["seed"]: e["success"] for e in json.loads((prep / "checkpoint" / "eval_info.json").read_text())["per_episode"]}
    pub = np.array([published.get(r, np.nan) for r in roots], dtype=float)
    has_pub = ~np.isnan(pub)

    a = verdict_a(success["OFFICIAL"])
    p0_vs_official = paired_diff(success["P0"], success["OFFICIAL"])
    a["published_same_seeds"] = {"n": int(has_pub.sum()), "rate": float(pub[has_pub].mean()) if has_pub.any() else None,
                                 "official_minus_published": paired_diff(success["OFFICIAL"][has_pub], pub[has_pub]) if has_pub.any() else None,
                                 "mcnemar": mcnemar_exact(success["OFFICIAL"][has_pub], pub[has_pub]) if has_pub.any() else None}
    a["p0_minus_official"] = p0_vs_official
    a["runtime_blocker"] = runtime_blocker(p0_vs_official)

    b_diff = paired_diff(success["PHYS8"], success["P0"])
    b = {"phys8_minus_p0": b_diff, "mcnemar": mcnemar_exact(success["PHYS8"], success["P0"]), "verdict": verdict_b(b_diff)}

    ret = retention(success["VIS8"], success["PHYS8"], success["P0"])
    c = {"vis8_minus_p0": paired_diff(success["VIS8"], success["P0"]), "retention": ret,
         "mcnemar_vis8_p0": mcnemar_exact(success["VIS8"], success["P0"]),
         "verdict": verdict_c(b["verdict"], ret), **offline_p0(records)}

    hrep = [hrep_estimates(r["hrep"]["success"]) for r in records]
    zeros = np.zeros(len(hrep))
    diagnostics = {
        "hrep_split": paired_diff([h["split"] for h in hrep], zeros),
        "hrep_naive_biased": paired_diff([h["naive"] for h in hrep], zeros),
        "hrep_default_continuation_success": float(np.mean([h["default"] for h in hrep])),
        "hrep_anchor_t_mean": float(np.mean([r["hrep"]["anchor_t"] for r in records])),
        "medoid8_minus_p0": paired_diff(success["MEDOID8"], success["P0"]),
        "success_rate": {k: float(v.mean()) for k, v in success.items()},
        "mean_max_coverage": coverage,
        "seconds_per_root": float(np.mean([r["seconds"] for r in records])),
        "clone_methods": sorted({r["clone_method"] for r in records}),
    }
    summary = {"roots": [first, first + count - 1], "gate_A": a, "gate_B": b, "gate_C": c, "diagnostics": diagnostics}
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--first", type=int, default=1000)
    parser.add_argument("--count", type=int, default=100)
    x = parser.parse_args()
    main(x.runs, x.prep, x.out, x.first, x.count)
