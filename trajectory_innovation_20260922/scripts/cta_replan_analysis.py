"""Replan-interval read-out (docs/CTA_REPLAN_INTERVAL_PROTOCOL.md). CPU only, reads closed-loop logs.

For each closed-loop run (one executed chunk length L): success per arm with root-paired contrasts; on P0's own states
the crossing share (some sibling reaches native success in the chunk, candidate 0 does not) and each scorer's capture;
on every arm's own trajectory the recovery after a missed crossing; planning time per episode. Across runs on the same
roots: difference in differences of CTAV2 - X, root bootstrap.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ti_wm.gates import BOOTSTRAP, mcnemar_exact, paired_diff  # noqa: E402

THR = 0.95


def load(run):
    shards = sorted(Path(run).glob("shard_*"))
    reports = [json.loads((s / "closed_report.json").read_text()) for s in shards]
    if not shards or any(r["status"] != "DONE" for r in reports):
        raise ValueError(f"incomplete run {run}")
    arms, names = reports[0]["arms"], reports[0]["log_scorers"]
    logs, eps = {}, {}
    for s in shards:
        for a in arms:
            with np.load(s / f"log_{a}.npz") as z:
                for k in z.files:
                    logs.setdefault(a, {}).setdefault(k, []).append(z[k])
        for line in (s / "episodes.jsonl").read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                eps[(e["arm"], e["root"])] = e
    logs = {a: {k: np.concatenate(v) for k, v in d.items()} for a, d in logs.items()}
    timing = {a: sum(r["timing"][a]["seconds"] for r in reports if a in r.get("timing", {})) for a in arms}
    n_exec = reports[0].get("n_exec", 8)
    return {"arms": arms, "names": names, "logs": logs, "eps": eps, "timing": timing, "n_exec": n_exec}


def success_vector(run, arm, roots):
    return np.array([run["eps"][(arm, r)]["success"] for r in roots], float)


def crossings(run):
    """On P0's states: crossings and each logged scorer's capture (argmax reaches native success)."""
    L = run["logs"]["P0"]
    hit = L["cov"] > THR
    cross = hit.any(1) & ~hit[:, 0]
    out = {"decisions": int(len(hit)), "crossings": int(cross.sum()), "share": float(cross.mean()),
           "roots_with_crossing": int(len(np.unique(L["root"][cross]))), "capture": {}}
    rows = np.arange(len(hit))
    for n in run["names"]:
        c = np.argmax(L[f"score_{n}"], 1)
        out["capture"][n] = float(hit[rows, c][cross].mean()) if cross.any() else float("nan")
    out["capture"]["GEOM8"] = float(hit[rows, np.argmax(L["geom"], 1)][cross].mean()) if cross.any() else float("nan")
    return out


def recovery(run, arm):
    """Roots where the arm's own choice missed a success that a sibling offered, and how many still succeeded."""
    L = run["logs"][arm]
    hit = L["cov"] > THR
    missed = hit.any(1) & ~hit[np.arange(len(hit)), L["chosen"]]
    roots = np.unique(L["root"][missed])
    return {"missed_decisions": int(missed.sum()), "roots": int(len(roots)),
            "recovered": int(sum(run["eps"][(arm, r)]["success"] for r in roots))}


def did(a15, b15, a8, b8, resamples=BOOTSTRAP, seed=0):
    """Mean over roots of (a15-b15) - (a8-b8) with a root bootstrap."""
    d = (a15 - b15) - (a8 - b8)
    idx = np.random.default_rng(seed).integers(0, len(d), (resamples, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    return {"mean": float(d.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(d))}


def summarize(run, roots, pairs):
    arms = run["arms"]
    succ = {a: success_vector(run, a, roots) for a in arms}
    out = {"n_exec": run["n_exec"], "roots": [roots[0], roots[-1], len(roots)],
           "success": {a: int(v.sum()) for a, v in succ.items()},
           "decisions_per_episode": {a: float(len(run["logs"][a]["root"]) / len(roots)) for a in arms},
           "seconds_per_episode": {a: run["timing"][a] / len(roots) for a in arms},
           "contrasts": {}, "crossings": crossings(run) if "P0" in arms else None,
           "recovery": {a: recovery(run, a) for a in arms}}
    for left, right in pairs:
        if left in succ and right in succ:
            out["contrasts"][f"{left}-{right}"] = {"success": paired_diff(succ[left], succ[right]),
                                                   "mcnemar": mcnemar_exact(succ[left], succ[right])}
    return out, succ


def main(a):
    roots = list(range(a.first, a.last + 1))
    pairs = [tuple(p.split("-")) for p in a.pairs.split(",")]
    report = {}
    succ = {}
    for label, run_dir in (("L8", a.run8), ("L15", a.run15)):
        if run_dir is None:
            continue
        run = load(run_dir)
        missing = [x for x in run["arms"] for r in roots if (x, r) not in run["eps"]]
        if missing:
            raise ValueError(f"{label}: missing episodes for {sorted(set(missing))[:3]}")
        report[label], succ[label] = summarize(run, roots, pairs)
    if len(succ) == 2:
        report["did"] = {}
        for x in a.did.split(","):
            if all(k in succ[L] for L in succ for k in ("CTAV2", x)):
                report["did"][f"CTAV2-{x}"] = did(succ["L15"]["CTAV2"], succ["L15"][x], succ["L8"]["CTAV2"], succ["L8"][x])
        for arm in ("P0", "GEOM8"):
            if arm in succ["L8"] and arm in succ["L15"]:
                report["did"][f"{arm}_L15-L8"] = paired_diff(succ["L15"][arm], succ["L8"][arm])
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "replan_summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=1), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run8", type=Path, default=None, help="closed-loop run at the native 8 actions (55666)")
    p.add_argument("--run15", type=Path, default=None, help="closed-loop run at 15 executed actions")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--first", type=int, default=2200)
    p.add_argument("--last", type=int, default=2399)
    p.add_argument("--pairs", default="CTAV2-DIRV2,CTAV2-DINOWM,CTAV2-ENDV2,CTAV2-P0,DINOWM-P0,GEOM8-P0")
    p.add_argument("--did", default="DIRV2,DINOWM,ENDV2,P0")
    main(p.parse_args())
