"""Compute rho_init / rho_final / recall / R_sel from CEM candidate dumps.

scripts/53 requires the stateprobe decode columns, which the `straight` arms do not
populate (they run without a probe). The quantities the gate actually turns on --
how well the proxy cost orders candidates by true task progress, at the first CEM
iteration and at the last -- need only `proxy_cost` and `true_shaped_cost`.

Baseline to beat (diagnosis/docs/CURRENT_STATUS.md:71, DINO):
    push l2  rho_init 0.25  rho_final -0.08   R_sel init 2.98 cm
    pick l2  rho_init 0.02  rho_final -0.09   R_sel init 3.74 cm
The gate is rho_final: straightening passes only if it stops being CI-clean negative.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def population_stats(g: pd.DataFrame, topk_frac: float = 0.1):
    proxy = g["proxy_cost"].to_numpy(float)
    true = g["true_shaped_cost"].to_numpy(float)
    obj = g["obj_goal_dist"].to_numpy(float)
    n = len(g)
    if n < 5 or not np.isfinite(proxy).all() or not np.isfinite(true).all():
        return None
    if np.std(proxy) == 0 or np.std(true) == 0:
        return None
    rho = spearmanr(proxy, true).statistic
    k = max(1, int(round(topk_frac * n)))
    proxy_elite = set(np.argsort(proxy)[:k])
    true_elite = set(np.argsort(true)[:k])
    recall = len(proxy_elite & true_elite) / k
    # selection regret in object-goal distance: what the proxy picked vs the best available
    r_sel = (obj[int(np.argmin(proxy))] - obj.min()) * 100.0
    return {"rho": rho, "recall": recall, "r_sel_cm": r_sel, "n": n}


def clustered_ci(vals, groups, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    vals, groups = np.asarray(vals, float), np.asarray(groups)
    ok = np.isfinite(vals)
    vals, groups = vals[ok], groups[ok]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    uniq = np.unique(groups)
    by = {u: vals[groups == u] for u in uniq}
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        boots.append(np.concatenate([by[u] for u in pick]).mean())
    return float(vals.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--out", default="results/hys_rho_audit.md")
    args = ap.parse_args()

    paths = []
    for p in args.candidates:
        paths.extend(sorted(glob.glob(p)))

    rows = []
    for path in paths:
        tag = Path(path).name.replace("cem_preselection_", "").replace("_candidates.csv.gz", "")
        df = pd.read_csv(path)
        for (task, seed, replan), g in df.groupby(["task", "seed", "replan"], sort=False):
            iters = sorted(g["iter"].unique())
            if len(iters) < 2:
                continue
            for label, it in (("init", iters[0]), ("final", iters[-1])):
                st = population_stats(g[g["iter"] == it])
                if st:
                    rows.append({"tag": tag, "task": task, "seed": seed,
                                 "replan": replan, "phase": label, **st})
    d = pd.DataFrame(rows)
    if d.empty:
        raise SystemExit("no populations parsed")

    lines = ["# HyS-JEPA rho audit — does straightening survive CEM search?", "",
             "Populations grouped by (task, seed, replan); CIs are bootstrap clustered on "
             "episode seed. `rho` = Spearman(proxy cost, true shaped cost) over the full "
             "candidate population. **The gate is rho_final.**", "",
             "| arm | phase | rho | 95% CI | recall@10% | R_sel (cm) | n pops |",
             "|---|---|---|---|---|---|---|"]
    summary = {}
    for tag in sorted(d["tag"].unique()):
        for phase in ("init", "final"):
            s = d[(d["tag"] == tag) & (d["phase"] == phase)]
            if s.empty:
                continue
            m, lo, hi = clustered_ci(s["rho"], s["seed"])
            rm, _, _ = clustered_ci(s["recall"], s["seed"])
            sm, _, _ = clustered_ci(s["r_sel_cm"], s["seed"])
            summary[(tag, phase)] = (m, lo, hi)
            flag = " **CI-clean neg**" if hi < 0 else (" **CI-clean pos**" if lo > 0 else "")
            lines.append(f"| {tag} | {phase} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}]{flag} | "
                         f"{rm:.3f} | {sm:.2f} | {len(s)} |")

    lines += ["", "## Gate verdict", ""]
    for tag in sorted(d["tag"].unique()):
        if ("final" not in [p for (t, p) in summary if t == tag]):
            continue
        m, lo, hi = summary[(tag, "final")]
        verdict = "FAIL (still CI-clean negative)" if hi < 0 else (
            "PASS (CI-clean positive)" if lo > 0 else "inconclusive (CI spans zero)")
        lines.append(f"- **{tag}**: rho_final {m:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    d.to_csv(str(out).replace(".md", "_populations.csv"), index=False)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
