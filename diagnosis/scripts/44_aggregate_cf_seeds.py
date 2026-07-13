"""Phase-H hardening — aggregate the counterfactual-predictor A/B across training
seeds and models into seed-mean ± CI, closing the single-seed Table in the paper.

The seed sweep (`slurm_phaseH_cf_seedsweep.sh`, SLURM array) trains
`predictor_cf_<model>_s{1,2,3}.pt` and the main `_lora` run, then scores each with
the DROID Action-Score planning probe (`scripts/08`), writing
`results/droid_planning_cf_<model>_{frozen,lora,s1,s2,s3}.csv` (per-regime rows;
`action_error`, `action_score`, `cra_eff`, `n_planned`). This script pools those
per model and reports, across the CF seeds, mean ± sd and whether every seed beats
the frozen baseline — the robustness claim the paper needs.

Pooling weights each regime×horizon cell by `n_planned` (equivalent to a
per-transition mean over the 157 planned transitions), matching the paper table's
`pooled` row. Frozen is a deterministic single eval, so it has no seed spread.

    python scripts/44_aggregate_cf_seeds.py --model dino_wm_droid \
        --seeds lora s1 s2 s3 --out results/cf_seed_summary.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = [("action_error", "AE", "lower"), ("action_score", "AS", "higher"),
           ("cra_eff", "CRA", "higher")]


def pooled(df: pd.DataFrame) -> dict:
    w = df["n_planned"].values
    return {abbr: float(np.average(df[col], weights=w)) for col, abbr, _ in METRICS}


def beats(cf: float, fz: float, direction: str) -> bool:
    return cf < fz if direction == "lower" else cf > fz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dino_wm_droid")
    ap.add_argument("--seeds", nargs="+", default=["lora", "s1", "s2", "s3"],
                    help="CF-run tags to pool as seeds (the main run is tagged 'lora')")
    ap.add_argument("--frozen-tag", default="frozen",
                    help="frozen-run tag (use heldout_frozen for the corrected Phase-H rerun)")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--extra-models", nargs="*", default=[],
                    help="additional models to report single-seed (e.g. jepa_wm_droid "
                         "with --extra-tags); no seed CI, just frozen vs each tag")
    ap.add_argument("--extra-tags", nargs="*", default=["lora"],
                    help="tags to report for --extra-models")
    ap.add_argument("--out", default="results/cf_seed_summary.md")
    args = ap.parse_args()

    rd = Path(args.results_dir)

    def load(model, tag):
        p = rd / f"droid_planning_cf_{model}_{tag}.csv"
        if not p.exists():
            return None
        return pooled(pd.read_csv(p))

    lines = ["# Phase-H hardening: counterfactual predictor A/B, seed sweep", "",
             f"Model **{args.model}**, CF seeds `{args.seeds}` vs frozen baseline.",
             "Pooled across reported DROID cells (weighted by `n_planned`).", ""]

    fz = load(args.model, args.frozen_tag)
    if fz is None:
        raise SystemExit(f"missing frozen baseline for {args.model}")
    seed_vals = {abbr: [] for _, abbr, _ in METRICS}
    per_seed = {}
    for s in args.seeds:
        p = load(args.model, s)
        if p is None:
            print(f"[warn] missing CF seed {s} for {args.model}", flush=True)
            continue
        per_seed[s] = p
        for _, abbr, _ in METRICS:
            seed_vals[abbr].append(p[abbr])

    lines += ["| metric | frozen | CF seed-mean ± sd | range | all seeds beat frozen? |",
              "|---|---|---|---|---|"]
    summary = {}
    for col, abbr, direction in METRICS:
        v = np.array(seed_vals[abbr])
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        allbeat = all(beats(x, fz[abbr], direction) for x in v)
        summary[abbr] = (fz[abbr], v.mean(), sd, v.min(), v.max(), allbeat)
        arrow = "↓" if direction == "lower" else "↑"
        lines.append(f"| {abbr} ({arrow}) | {fz[abbr]:.3f} | {v.mean():.3f} ± {sd:.3f} "
                     f"(n={len(v)}) | [{v.min():.3f}, {v.max():.3f}] | "
                     f"{'**yes**' if allbeat else 'NO'} |")

    lines += ["", "Per-seed pooled values:", "",
              "| seed | " + " | ".join(a for _, a, _ in METRICS) + " |",
              "|---|" + "---|" * len(METRICS)]
    for s, p in per_seed.items():
        lines.append(f"| {s} | " + " | ".join(f"{p[a]:.3f}" for _, a, _ in METRICS) + " |")

    # Second model(s), single-seed A/B (todo: second-model transfer).
    for m in args.extra_models:
        efz = load(m, "frozen")
        if efz is None:
            continue
        lines += ["", f"## Second model: {m} (single-seed A/B)", "",
                  "| tag | " + " | ".join(f"{a} {'↓' if d=='lower' else '↑'}"
                                          for _, a, d in METRICS) + " | all better? |",
                  "|---|" + "---|" * (len(METRICS) + 1)]
        lines.append(f"| frozen | " + " | ".join(f"{efz[a]:.3f}" for _, a, _ in METRICS) + " | — |")
        for t in args.extra_tags:
            ep = load(m, t)
            if ep is None:
                continue
            allb = all(beats(ep[a], efz[a], d) for _, a, d in METRICS)
            lines.append(f"| {t} | " + " | ".join(f"{ep[a]:.3f}" for _, a, _ in METRICS)
                         + f" | {'**yes**' if allb else 'partial'} |")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
