"""Paired ranking comparison of scorers on the logged closed-loop banks (scripts/cta_diag_onpolicy.py --mode closed).

Every logged scorer scored the same simulated bank at every decision, so scorer differences are paired by decision.
For each scorer: retained gap = sum_d [y(chosen) - y(0)] / sum_d [max y - y(0)] and within-bank Spearman (decisions
with a label spread; a scorer tying the bank counts 0). Differences between scorers are bootstrapped with roots as
clusters, the same resample for both scorers. State sets: P0's visited states (primary: independent of every learned
scorer) and all arms' states pooled (a root and its decisions from every arm form one cluster). CPU only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

RESAMPLES = 4000


def load(closed_run):
    """Decision logs per arm, concatenated over shards; shards must be DONE with identical arms and hashes."""
    logs, reports = {}, []
    for shard in sorted(Path(closed_run).glob("shard_*")):
        rep = json.loads((shard / "closed_report.json").read_text())
        if rep["status"] != "DONE":
            raise ValueError(f"incomplete shard {shard}")
        if reports and (rep["arms"] != reports[0]["arms"] or rep["hashes"] != reports[0]["hashes"]):
            raise ValueError(f"inconsistent shard {shard}")
        reports.append(rep)
        for arm in rep["arms"]:
            with np.load(shard / f"log_{arm}.npz") as z:
                for k in z.files:
                    logs.setdefault(arm, {}).setdefault(k, []).append(z[k])
    return {a: {k: np.concatenate(v) for k, v in d.items()} for a, d in logs.items()}, reports[0]


def spearman_rows(s, y):
    """Row-wise Spearman of (N, K) arrays; 0 where the scores are constant."""
    rs, ry = rankdata(s, axis=1), rankdata(y, axis=1)
    rs, ry = rs - rs.mean(1, keepdims=True), ry - ry.mean(1, keepdims=True)
    den = np.sqrt((rs ** 2).sum(1) * (ry ** 2).sum(1))
    return np.where(den > 0, (rs * ry).sum(1) / np.where(den > 0, den, 1), 0.0)


def per_root(log, scorers, label):
    """Per-root sums: retained-gap numerator per scorer, shared denominator, Spearman sum per scorer and count."""
    y, roots = log[label], log["root"]
    uniq, inv = np.unique(roots, return_inverse=True)
    rows = np.arange(len(y))
    spread = np.ptp(y, axis=1) > 0
    den = np.bincount(inv, y.max(1) - y[:, 0], len(uniq))
    cnt = np.bincount(inv, spread.astype(float), len(uniq))
    num, rho = {}, {}
    for n in scorers:
        s = log[f"score_{n}"]
        chosen = np.argmax(s, axis=1)            # first best; ties -> candidate 0 as ti_wm.contract.select_candidate
        num[n] = np.bincount(inv, y[rows, chosen] - y[:, 0], len(uniq))
        r = np.zeros(len(y))
        if spread.any():
            r[spread] = np.where(np.ptp(s[spread], axis=1) > 0, spearman_rows(s[spread], y[spread]), 0.0)
        rho[n] = np.bincount(inv, r, len(uniq))
    return uniq, num, den, rho, cnt


def compare(uniq, num, den, rho, cnt, pairs, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), (RESAMPLES, len(uniq)))
    bd, bc = den[idx].sum(1), cnt[idx].sum(1)
    gap = {n: (num[n].sum() / den.sum(), num[n][idx].sum(1) / bd) for n in num}
    sp = {n: (rho[n].sum() / cnt.sum(), rho[n][idx].sum(1) / bc) for n in rho}
    ci = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    out = {"n_roots": int(len(uniq)), "n_decisions_with_spread": int(cnt.sum()), "scorers": {}, "pairs": {}}
    for n in num:
        out["scorers"][n] = {"retained_gap": [float(gap[n][0])] + ci(gap[n][1]),
                             "spearman": [float(sp[n][0])] + ci(sp[n][1])}
    for a, b in pairs:
        if a in num and b in num:
            d, r = gap[a][1] - gap[b][1], sp[a][1] - sp[b][1]
            out["pairs"][f"{a}-{b}"] = {"retained_gap": [float(gap[a][0] - gap[b][0])] + ci(d),
                                        "p_gap_le_0": float(np.mean(d <= 0)),
                                        "spearman": [float(sp[a][0] - sp[b][0])] + ci(r)}
    return out


def main(a):
    logs, rep = load(a.closed_run)
    scorers = rep["log_scorers"]
    pairs = [tuple(p.split("-")) for p in a.pairs.split(",")]
    result = {"closed_run": str(a.closed_run), "scorers": scorers, "hashes": rep["hashes"], "sets": {}}
    if a.reference_run:
        # a rerun must revisit the reference run's P0 states exactly; shared scorers must reproduce its scores
        ref, _ = load(a.reference_run)
        mine, theirs = logs["P0"], ref["P0"]
        for k in ("root", "decision", "t", "chosen"):
            if not np.array_equal(mine[k], theirs[k]):
                raise ValueError(f"P0 states differ from the reference run ({k})")
        if not np.allclose(mine["geom"], theirs["geom"]):
            raise ValueError("P0 bank labels differ from the reference run")
        check = {}
        for n in scorers:
            if f"score_{n}" in theirs:
                x, y = mine[f"score_{n}"], theirs[f"score_{n}"]
                check[n] = {"max_abs_diff": float(np.abs(x - y).max()),
                            "same_choice": float(np.mean(np.argmax(x, 1) == np.argmax(y, 1)))}
        result["reference_check"] = check
        print({"reference_check": check}, flush=True)
        if logs.keys() - {"P0"}:
            raise ValueError("--reference-run expects a P0-only rerun")
    for label in ("geom", "cov"):
        # primary: P0's states
        result["sets"][f"P0_states/{label}"] = compare(*per_root(logs["P0"], scorers, label), pairs)
        if len(rep["arms"]) == 1:
            continue
        # pooled: all arms' states, clusters = roots (sums over arms of the same root)
        parts = [per_root(logs[arm], scorers, label) for arm in rep["arms"]]
        roots = np.unique(np.concatenate([p[0] for p in parts]))
        pos = lambda u: np.searchsorted(roots, u)
        num = {n: np.zeros(len(roots)) for n in scorers}
        rho = {n: np.zeros(len(roots)) for n in scorers}
        den, cnt = np.zeros(len(roots)), np.zeros(len(roots))
        for u, nu, de, rh, cn in parts:
            j = pos(u)
            den[j] += de
            cnt[j] += cn
            for n in scorers:
                num[n][j] += nu[n]
                rho[n][j] += rh[n]
        result["sets"][f"all_states/{label}"] = compare(roots, num, den, rho, cnt, pairs)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "paired_ladder.json").write_text(json.dumps(result, indent=1))
    lines = []
    for key, r in result["sets"].items():
        lines += [f"## {key}  (roots {r['n_roots']}, decisions with spread {r['n_decisions_with_spread']})", "",
                  "| scorer | retained gap [95% CI] | within-bank Spearman [95% CI] |", "|---|---|---|"]
        for n, v in r["scorers"].items():
            g, s = v["retained_gap"], v["spearman"]
            lines.append(f"| {n} | {g[0]:.3f} [{g[1]:.3f}, {g[2]:.3f}] | {s[0]:.3f} [{s[1]:.3f}, {s[2]:.3f}] |")
        lines += ["", "| pair | Δ retained gap [95% CI] | P(Δ≤0) | Δ Spearman [95% CI] |", "|---|---|---|---|"]
        for k, v in r["pairs"].items():
            g, s = v["retained_gap"], v["spearman"]
            lines.append(f"| {k} | {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}] | {v['p_gap_le_0']:.3f} | "
                         f"{s[0]:+.3f} [{s[1]:+.3f}, {s[2]:+.3f}] |")
        lines.append("")
    (a.out / "paired_ladder.md").write_text("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--closed-run", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reference-run", type=Path, default=None, help="closed run whose P0 states this run must repeat")
    p.add_argument("--pairs", default="CTAV2-ENDV2,CTAV2-DIRV2,CTAV2-DINOWM,CTAV2-CTA4,ENDV2-DIRV2,ENDV2-DINOWM,"
                                      "DIRV2-DINOWM,FULLV2-CTAV2,FULLV2-ENDV2,FULLV2-DIRV2,CODEV2-CTAV2")
    main(p.parse_args())
