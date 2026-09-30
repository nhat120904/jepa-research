"""Offline evaluation of stopping rules on recorded stopping trees."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .rules import CHECKPOINTS, choose


@dataclass
class TreeRoot:
    root: int
    costs1: np.ndarray        # (30, S)
    costs2: np.ndarray        # (n, 30, S), NaN where plan 2 was not run
    p1_term: np.ndarray       # (n,)
    leaf_success: np.ndarray  # (n, n)
    leaf_dist: np.ndarray     # (n, n)
    checkpoints: tuple[int, ...]
    extra: dict


def load_root(path: Path) -> TreeRoot:
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    cps = tuple(int(k) for k in data["checkpoints"])
    return TreeRoot(root=int(path.stem.split("_")[-1]), costs1=data["costs1"],
                    costs2=data["costs2"], p1_term=data["p1_term"],
                    leaf_success=data["leaf_success"], leaf_dist=data["leaf_dist"],
                    checkpoints=cps, extra=data)


def load_task(directory: Path) -> list[TreeRoot]:
    return [load_root(p) for p in sorted(Path(directory).glob("root_*.npz"))]


def path_of(rule, tr: TreeRoot) -> tuple[int, int | None]:
    """Indices (i1, i2) the rule visits; i2 is None if plan 1 already succeeded."""
    cps = tr.checkpoints
    i1 = cps.index(choose(rule, tr.costs1, cps))
    if tr.p1_term[i1]:
        return i1, None
    i2 = cps.index(choose(rule, tr.costs2[i1], cps))
    return i1, i2


def outcome(rule, tr: TreeRoot) -> dict:
    i1, i2 = path_of(rule, tr)
    j = 0 if i2 is None else i2
    iters = tr.checkpoints[i1] + (0 if i2 is None else tr.checkpoints[i2])
    return {"success": bool(tr.leaf_success[i1, j]), "distance": float(tr.leaf_dist[i1, j]),
            "iterations": int(iters), "i1": i1, "i2": i2}


def evaluate(rule, roots: list[TreeRoot]) -> dict:
    rows = [outcome(rule, tr) for tr in roots]
    return {
        "success": np.array([r["success"] for r in rows], bool),
        "distance": np.array([r["distance"] for r in rows]),
        "iterations": np.array([r["iterations"] for r in rows], np.int32),
    }


def oracle(roots: list[TreeRoot]) -> dict:
    """Best leaf per root (finite per-plan checkpoint choices, fixed noise)."""
    succ = np.array([tr.leaf_success.any() for tr in roots], bool)
    dist = np.array([np.nanmin(tr.leaf_dist) for tr in roots])
    fixed_any = np.array([np.diag(tr.leaf_success).any() for tr in roots], bool)
    return {"any_leaf_success": succ, "min_leaf_distance": dist,
            "best_fixed_k_per_root_success": fixed_any}


def tune(candidates: list, roots: list[TreeRoot]) -> tuple[object, list[dict]]:
    """Pick the rule with the highest success; ties -> fewer mean iterations."""
    table = []
    for rule in candidates:
        res = evaluate(rule, roots)
        table.append({"rule": rule, "param": rule.param(), "success": float(res["success"].mean()),
                      "iterations": float(res["iterations"].mean()),
                      "distance": float(np.mean(res["distance"]))})
    best = max(table, key=lambda r: (r["success"], -r["iterations"], -r["distance"]))
    return best["rule"], table


def paired_bootstrap(a: np.ndarray, b: np.ndarray, draws: int = 10_000, seed: int = 0) -> dict:
    """Mean of a - b with a 95% percentile interval, resampling roots."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    diff = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(draws, len(diff)))
    boot = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": float(diff.mean()), "lo": float(lo), "hi": float(hi), "n": int(len(diff)),
            "discordant": float(np.mean(diff != 0))}


def kendall_with_zone(cost: np.ndarray, phys: np.ndarray, zone: float) -> dict:
    """Pairwise ordering accuracy counting only pairs whose physical gap exceeds zone."""
    cost = np.asarray(cost, np.float64)
    phys = np.asarray(phys, np.float64)
    i, j = np.triu_indices(len(cost), 1)
    dp = phys[i] - phys[j]
    dc = cost[i] - cost[j]
    keep = (np.abs(dp) > zone) & (dc != 0)
    agree = np.sign(dp[keep]) == np.sign(dc[keep])
    return {"pairs": int(keep.sum()), "accuracy": float(agree.mean()) if keep.any() else float("nan"),
            "abs_cost_gap": np.abs(dc[keep]), "correct": agree}
