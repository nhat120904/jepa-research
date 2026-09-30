"""Paired closed-loop comparison of OGBench arms across evaluation runs (scripts/ogbench/ogb_cta_eval.py outputs).

Episodes are paired by (task, episode): every run uses the same frozen policy and the same seeded banks, so P0 is
identical across runs and arms from different checkpoints are paired on the same initial states. An arm is named
<label>:<ARM> with label given per run (--run label=dir). Reports success, per-task success, McNemar exact p and a
bootstrap CI (resampling episodes) for each requested contrast.
Usage: ogb_compare.py --run v1=<dir> --run v2=<dir> --pairs v2:CTA-v1:P0,v2:CTA-v2:DIRECT [--out file.json]
"""
import argparse
import json
from math import comb
from pathlib import Path

import numpy as np


def load(runs):
    arms = {}
    for spec in runs:
        label, path = spec.split("=", 1)
        for line in open(Path(path) / "episodes.jsonl"):
            e = json.loads(line)
            arms.setdefault(f"{label}:{e['arm']}", {})[(e["task"], e["ep"])] = bool(e["success"])
    return arms


def mcnemar(a, b):
    only_a, only_b = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = only_a + only_b
    p = min(1.0, 2 * sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": p}


def main(a):
    arms = load(a.run)
    out = {"arms": {}, "contrasts": {}}
    for name, res in sorted(arms.items()):
        tasks = sorted({t for t, _ in res})
        out["arms"][name] = {"n": len(res), "success": float(np.mean(list(res.values()))),
                             "by_task": {t: float(np.mean([v for (tt, _), v in res.items() if tt == t])) for t in tasks}}
    rng = np.random.default_rng(0)
    for pair in a.pairs.split(","):
        left, right = pair.split("-", 1) if pair.count("-") == 1 else pair.rsplit("-", 1)
        keys = sorted(set(arms[left]) & set(arms[right]))
        x = np.array([arms[left][k] for k in keys])
        y = np.array([arms[right][k] for k in keys])
        d = x.astype(float) - y
        boot = d[rng.integers(0, len(d), (5000, len(d)))].mean(1)
        out["contrasts"][pair] = {"n": len(keys), "diff": float(d.mean()), "lo": float(np.percentile(boot, 2.5)),
                                  "hi": float(np.percentile(boot, 97.5)), **mcnemar(x, y)}
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="append", required=True)
    p.add_argument("--pairs", required=True)
    p.add_argument("--out")
    main(p.parse_args())
