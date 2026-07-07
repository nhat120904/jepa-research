"""E0 analysis — cost-overoptimization (Goodhart) curves and the 2-panel figure.

Consumes the sweep outputs (scripts/41): per-episode `overopt_episodes_*.csv` and
per-CEM-iteration `overopt_curves_*.csv`. Produces:

  * results/overopt_analysis.md — across-budget outcome table (with episode-clustered
    bootstrap CIs on obj_goal_dist + success_end) and the within-search Goodhart
    summary (proxy vs true vs decode-error, first→last CEM iter).
  * results/figures/figure_overopt.pdf/.png — two panels:
      (A) across-budget: true obj→goal distance vs CEM iterations, one line per
          (task, cost); the "does more search help?" panel.
      (B) within-search Goodhart: at the largest budget, proxy_min / true_sp /
          decode_err vs CEM iteration for push×stateprobe (the hacking cell), with
          reach×l2 overlaid as the honest-cost control.

Bootstrap is episode-clustered (resample whole episodes) — the paper's CI convention
(metrics/bootstrap.py). Runs on CPU; no GPU/model needed.

    python scripts/42_analyze_overopt.py --tags mwpush mwreach
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _boot_ci(values_by_episode, stat=np.mean, n_boot=2000, seed=0, alpha=0.05):
    """Episode-clustered bootstrap CI for a per-episode statistic. `values_by_episode`
    is a list of arrays (one per episode); we resample episodes with replacement and
    recompute `stat` over the pooled values."""
    rng = np.random.default_rng(seed)
    eps = [np.asarray(v, dtype=float) for v in values_by_episode if len(v)]
    if not eps:
        return (float("nan"), float("nan"), float("nan"))
    n = len(eps)
    point = stat(np.concatenate(eps))
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = stat(np.concatenate([eps[i] for i in idx]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def across_budget(ep: pd.DataFrame) -> pd.DataFrame:
    """Per (task, cost, iterations, samples): success_end fraction and median
    obj_goal_dist, each with an episode-clustered bootstrap CI."""
    rows = []
    for (task, cost, it, n), g in ep.groupby(["task", "cost", "iterations", "samples"]):
        # each episode contributes one row here (one episode = one seed), so
        # episode-clustered == row-clustered; wrap each value as a 1-elt array.
        succ = [np.array([v]) for v in g.success_end.values]
        obj = [np.array([v]) for v in g.obj_goal_dist.values]
        s_pt, s_lo, s_hi = _boot_ci(succ, stat=np.mean)
        o_pt, o_lo, o_hi = _boot_ci(obj, stat=np.median)
        rows.append(dict(task=task, cost=cost, iterations=int(it), samples=int(n),
                         n_ep=len(g),
                         success_end=f"{int(g.success_end.sum())}/{len(g)}",
                         success_frac=s_pt, success_lo=s_lo, success_hi=s_hi,
                         obj_med=o_pt, obj_lo=o_lo, obj_hi=o_hi,
                         ee_med=float(g.ee_dist.median())))
    return pd.DataFrame(rows).sort_values(["task", "cost", "samples", "iterations"])


def within_search(cur: pd.DataFrame, iters_budget: int | None = None) -> pd.DataFrame:
    """Per (task, cost, CEM-iter) mean of proxy_min / true_sp / true_obj / decode_err
    at a fixed total budget (default: the largest iterations present), averaged over
    episodes and replans."""
    out = []
    for (task, cost), g0 in cur.groupby(["task", "cost"]):
        budget = iters_budget or int(g0.iterations.max())
        g = g0[g0.iterations == budget]
        if not len(g):
            continue
        agg = g.groupby("iter").agg(
            proxy_min=("proxy_min", "mean"),
            true_sp=("true_sp_min_believed", "mean"),
            true_obj=("true_obj_min_believed", "mean"),
            decode=("decode_err_med_cm", "mean")).reset_index()
        agg.insert(0, "cost", cost); agg.insert(0, "task", task); agg["budget"] = budget
        out.append(agg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def make_figure(ab: pd.DataFrame, ws: pd.DataFrame, outpath: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: true obj→goal vs iterations (samples fixed at the modal value).
    n_mode = int(ab["samples"].mode().iloc[0])
    a = ab[ab.samples == n_mode]
    for (task, cost), g in a.groupby(["task", "cost"]):
        g = g.sort_values("iterations")
        # reach obj is trivially ~0 (no object) — use ee for reach so the panel is
        # meaningful for both task families.
        if task.startswith("mw-reach"):
            y = g.ee_med.values; ylab = "true dist to goal (obj / ee)"
        else:
            y = g.obj_med.values; ylab = "true dist to goal (obj / ee)"
        axA.plot(g.iterations, y, marker="o", label=f"{task}×{cost}")
        axA.fill_between(g.iterations, g.obj_lo if not task.startswith("mw-reach") else y,
                         g.obj_hi if not task.startswith("mw-reach") else y, alpha=0.12)
    axA.set_xlabel("CEM iterations (optimization pressure)")
    axA.set_ylabel(ylab)
    axA.set_title(f"(A) Does more search help? (n={n_mode} samples)")
    axA.legend(fontsize=7); axA.grid(alpha=0.3)

    # Panel B: within-search Goodhart for the hacking cell + honest control.
    def _plot_cell(task, cost, ax, twin, c_proxy, c_true, c_dec, label):
        g = ws[(ws.task == task) & (ws.cost == cost)].sort_values("iter")
        if not len(g):
            return
        # normalise proxy/true to their iter-0 value so both fit one axis
        p0, t0 = g.proxy_min.iloc[0], g.true_sp.iloc[0]
        ax.plot(g["iter"], g.proxy_min / p0, color=c_proxy, ls="-",
                label=f"{label}: proxy (planner)")
        ax.plot(g["iter"], g.true_sp / t0, color=c_true, ls="--",
                label=f"{label}: TRUE cost")
        twin.plot(g["iter"], g.decode, color=c_dec, ls=":",
                  label=f"{label}: decode err (cm)")

    axBt = axB.twinx()
    _plot_cell("mw-push", "stateprobe", axB, axBt, "#c1121f", "#780000", "#e07a00",
               "push×stateprobe (hack)")
    _plot_cell("mw-reach", "l2", axB, axBt, "#4361ee", "#3a0ca3", "#4895ef",
               "reach×l2 (honest)")
    axB.set_xlabel("CEM iteration within one search")
    axB.set_ylabel("proxy / true cost (normalized to iter 0)")
    axBt.set_ylabel("object decode error (cm)")
    axB.set_title("(B) Proxy improves, TRUE stalls, pocket deepens")
    h1, l1 = axB.get_legend_handles_labels()
    h2, l2 = axBt.get_legend_handles_labels()
    axB.legend(h1 + h2, l1 + l2, fontsize=6, loc="center right")
    axB.grid(alpha=0.3)

    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {outpath} (+ .png)", flush=True)


def _dose(arm: str) -> int:
    """CEM iteration dose from an E1 arm name; gcidm → -1 (pure amortized)."""
    return -1 if arm == "gcidm" else int(arm.rsplit("it", 1)[1])


def e1_tables(ep1: pd.DataFrame, sv: pd.DataFrame):
    """E1 outcome-by-arm and per-replan seed-vs-chosen corruption dose-response."""
    out = {"outcomes": [], "corruption": []}
    for arm, g in sorted(ep1.groupby("arm"), key=lambda kv: _dose(kv[0])):
        succ = [np.array([v]) for v in g.success_end.values]
        obj = [np.array([v]) for v in g.obj_goal_dist.values]
        s_pt, s_lo, s_hi = _boot_ci(succ, stat=np.mean)
        o_pt, o_lo, o_hi = _boot_ci(obj, stat=np.median)
        out["outcomes"].append(dict(
            arm=arm, dose=_dose(arm), n_ep=len(g),
            success_end=f"{int(g.success_end.sum())}/{len(g)}",
            success_frac=s_pt, obj_med=o_pt, obj_lo=o_lo, obj_hi=o_hi,
            ee_med=float(g.ee_dist.median())))
    if len(sv):
        for arm, g in sorted(sv.groupby("arm"), key=lambda kv: _dose(kv[0])):
            dtrue = g.chosen_true_sp - g.seed_true_sp        # +ve = search made TRUE worse
            # episode-clustered CI on the corruption rate (cluster on seed).
            by_ep = [gg.corruption.values for _, gg in g.groupby("seed")]
            c_pt, c_lo, c_hi = _boot_ci(by_ep, stat=np.mean)
            out["corruption"].append(dict(
                arm=arm, dose=_dose(arm), n_replan=len(g),
                beats_seed_proxy=float((g.chosen_proxy < g.seed_proxy).mean()),
                corruption=c_pt, corr_lo=c_lo, corr_hi=c_hi,
                dtrue_mean=float(dtrue.mean())))
    return out


def make_e1_figure(e1: dict, outpath: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    oc = pd.DataFrame(e1["outcomes"]).sort_values("dose")
    co = pd.DataFrame(e1["corruption"]).sort_values("dose") if e1["corruption"] else pd.DataFrame()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.2))

    # Panel A: true obj→goal by search dose (gcidm plotted at x=-2 as "no search").
    x = oc.dose.replace(-1, -2).values
    axA.errorbar(x, oc.obj_med, yerr=[oc.obj_med - oc.obj_lo, oc.obj_hi - oc.obj_med],
                 marker="o", capsize=3, color="#c1121f", label="median true obj→goal")
    axA.axhline(0.05, ls=":", color="grey", label="push success radius (5 cm)")
    axA.set_xticks(x)
    axA.set_xticklabels(["gcidm\n(no search)"] + [str(int(d)) for d in oc.dose[1:]])
    axA.set_xlabel("search dose (CEM iterations)")
    axA.set_ylabel("true object→goal distance (m)")
    axA.set_title("(A) Neither amortize nor search crosses contact")
    axA.legend(fontsize=7); axA.grid(alpha=0.3)

    # Panel B: corruption rate + mean Δtrue vs search dose (seeded arms only).
    if len(co):
        c = co[co.dose >= 0]
        axB.errorbar(c.dose, c.corruption,
                     yerr=[c.corruption - c.corr_lo, c.corr_hi - c.corruption],
                     marker="s", capsize=3, color="#780000", label="corruption rate")
        axB.set_xlabel("search dose (CEM iterations)")
        axB.set_ylabel("corruption rate\n(search: proxy↓ AND true↑)")
        axB.set_title("(B) Corruption rises with search dose")
        axB.set_ylim(bottom=0)
        axBt = axB.twinx()
        axBt.plot(c.dose, c.dtrue_mean, marker="^", ls="--", color="#4361ee",
                  label="mean Δtrue (search−seed)")
        axBt.axhline(0, ls=":", color="grey")
        axBt.set_ylabel("mean Δtrue_sp (search − seed)\n<0 = search helps on average")
        h1, l1 = axB.get_legend_handles_labels()
        h2, l2 = axBt.get_legend_handles_labels()
        axB.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
        axB.grid(alpha=0.3)
    fig.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    fig.savefig(outpath.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {outpath} (+ .png)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["mwpush", "mwreach", "mwpush_nsamp"],
                    help="output tags of the sweep CSVs to pool")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out-md", default="results/overopt_analysis.md")
    ap.add_argument("--out-fig", default="results/figures/figure_overopt.pdf")
    ap.add_argument("--e1-episodes", default="results/e1_episodes.csv",
                    help="E1 outcome CSV (scripts/43); analysed if present")
    ap.add_argument("--e1-seedvs", default="results/e1_seed_vs_chosen.csv")
    ap.add_argument("--out-fig-e1", default="results/figures/figure_e1.pdf")
    args = ap.parse_args()

    rd = Path(args.results_dir)
    eps, curs = [], []
    for tag in args.tags:
        ep_p, cu_p = rd / f"overopt_episodes_{tag}.csv", rd / f"overopt_curves_{tag}.csv"
        if ep_p.exists():
            eps.append(pd.read_csv(ep_p))
        if cu_p.exists():
            curs.append(pd.read_csv(cu_p))
    if not eps:
        raise SystemExit(f"no overopt_episodes_*.csv found for tags {args.tags} in {rd}")
    ep = pd.concat(eps, ignore_index=True)
    cur = pd.concat(curs, ignore_index=True) if curs else pd.DataFrame()

    ab = across_budget(ep)
    ws = within_search(cur) if len(cur) else pd.DataFrame()

    lines = ["# Cost-overoptimization (E0) analysis", "",
             "Latent oracle (perfect dynamics, scripts/30) — any proxy/true divergence is the",
             "*cost under optimization*, not predictor error. See",
             "`docs/plans/2026-07-06-overoptimization-sweep-design.md`.", "",
             "## Across-budget outcomes (does more search help?)", "",
             "Episode-clustered bootstrap 95% CI. `obj_med` = median true object→goal distance;",
             "for reach read `ee_med` (no object).", "",
             "| task | cost | iters | n_samp | n_ep | success_end | success frac [CI] | obj_med [CI] | ee_med |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in ab.iterrows():
        lines.append(
            f"| {r.task} | {r.cost} | {r.iterations} | {r.samples} | {r.n_ep} | "
            f"{r.success_end} | {r.success_frac:.2f} [{r.success_lo:.2f},{r.success_hi:.2f}] | "
            f"{r.obj_med:.3f} [{r.obj_lo:.3f},{r.obj_hi:.3f}] | {r.ee_med:.3f} |")

    if len(ws):
        lines += ["", "## Within-search Goodhart (first → last CEM iter, at max budget)", "",
                  "Proxy = the cost the planner minimizes; TRUE = the real state cost of the",
                  "candidate it most believes in; decode = object-probe error on the committed",
                  "elites (pocket depth).", "",
                  "| task | cost | budget | proxy Δ% | true_sp Δ% | decode first→last (cm) |",
                  "|---|---|---|---|---|---|"]
        for (task, cost), g in ws.groupby(["task", "cost"]):
            g = g.sort_values("iter")
            f, l = g.iloc[0], g.iloc[-1]
            dp = 100 * (l.proxy_min - f.proxy_min) / f.proxy_min
            dt = 100 * (l.true_sp - f.true_sp) / f.true_sp
            lines.append(f"| {task} | {cost} | {int(f.budget)} | {dp:+.0f}% | {dt:+.0f}% | "
                         f"{f.decode:.1f}→{l.decode:.1f} ({100*(l.decode-f.decode)/f.decode:+.0f}%) |")
        lines += ["", "**Reading:** an *exploitable* cost shows proxy Δ ≪ 0 (planner \"improves\"),",
                  "true_sp Δ ≈ 0 (reality stalls), decode ↑ (search converges onto readout-error",
                  "pockets). An *honest* cost (reach×l2) shows proxy ↓ AND the across-budget",
                  "success climbing with iterations. A *no-signal* cost (push×l2) shows proxy ↓",
                  "but true perfectly flat at every budget."]

    # ---- E1: amortized control + search-dose corruption (optional) ----
    e1 = None
    e1_ep_p, e1_sv_p = Path(args.e1_episodes), Path(args.e1_seedvs)
    if e1_ep_p.exists():
        ep1 = pd.read_csv(e1_ep_p)
        sv = pd.read_csv(e1_sv_p) if e1_sv_p.exists() else pd.DataFrame()
        e1 = e1_tables(ep1, sv)
        lines += ["", "## E1 — amortized control + search-dose response (mw-push)", "",
                  "Removing the search adversary (`gcidm`, pure amortized) and re-adding it in",
                  "doses. Episode-clustered bootstrap 95% CI.", "",
                  "| arm (dose) | n_ep | success_end | obj_med [CI] | ee_med |",
                  "|---|---|---|---|---|"]
        for r in e1["outcomes"]:
            lines.append(f"| {r['arm']} | {r['n_ep']} | {r['success_end']} | "
                         f"{r['obj_med']:.3f} [{r['obj_lo']:.3f},{r['obj_hi']:.3f}] | {r['ee_med']:.3f} |")
        if e1["corruption"]:
            lines += ["", "**Seed-vs-chosen** (per replan): does search corrupt the amortized seed?",
                      "`corruption` = search lowered proxy AND raised true cost; `Δtrue_mean` < 0",
                      "means search helped the true state on average.", "",
                      "| arm (dose) | n_replan | beats_seed(proxy) | corruption [CI] | Δtrue_mean |",
                      "|---|---|---|---|---|"]
            for r in e1["corruption"]:
                lines.append(f"| {r['arm']} | {r['n_replan']} | {r['beats_seed_proxy']:.2f} | "
                             f"{r['corruption']:.2f} [{r['corr_lo']:.2f},{r['corr_hi']:.2f}] | "
                             f"{r['dtrue_mean']:+.4f} |")
        lines += ["", "**Reading:** all arms end 0/8 with obj_med ≈ 0.25 m — neither amortizing away",
                  "the search NOR any search dose crosses the contact wall; the object never moves.",
                  "Corruption is real and rises monotonically with dose (0→7%), confirming the",
                  "adversary at the plan level, but is too small to dominate — search still helps",
                  "the *hand* (Δtrue < 0) while neither seed nor search moves the *object*. E1 is a",
                  "null for \"remove the adversary fixes contact\": the frozen representation, not",
                  "the optimizer alone, is the residual wall. Grounding is necessary, not sufficient."]

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_md}", flush=True)
    print("\n".join(lines))

    if len(ws) and len(ab):
        try:
            make_figure(ab, ws, Path(args.out_fig))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] E0 figure skipped: {e}", flush=True)
    if e1 is not None and e1["outcomes"]:
        try:
            make_e1_figure(e1, Path(args.out_fig_e1))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] E1 figure skipped: {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
