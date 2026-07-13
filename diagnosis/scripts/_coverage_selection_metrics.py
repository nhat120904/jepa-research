"""Pure-array metrics for candidate coverage versus proxy selection.

All inputs describe the same candidate population and lower costs are better.
The module has no model, MuJoCo, pandas, or SciPy dependency so its semantics can
be tested cheaply.  It distinguishes a search failure (no good candidate was
sampled) from a selection failure (a good candidate existed but the proxy did
not select it).
"""

from __future__ import annotations

import numpy as np


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """Zero-based average ranks with stable tie handling."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    start = 0
    while start < len(x):
        stop = start + 1
        while stop < len(x) and sorted_x[stop] == sorted_x[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman_costs(x, y) -> float:
    """Spearman correlation with average ties; NaN for constant/short input."""
    a = np.asarray(x, dtype=float).reshape(-1)
    b = np.asarray(y, dtype=float).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("cost arrays must be aligned, non-empty 1-D arrays")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("cost arrays must be finite")
    if a.size < 2:
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    if np.ptp(ra) == 0 or np.ptp(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def topk_overlap(proxy, truth, k: int) -> float:
    """Intersection-over-k of the two lowest-cost sets on identical candidates."""
    p = np.asarray(proxy, dtype=float).reshape(-1)
    t = np.asarray(truth, dtype=float).reshape(-1)
    if p.shape != t.shape or p.size == 0:
        raise ValueError("cost arrays must be aligned, non-empty 1-D arrays")
    if not 1 <= int(k) <= p.size:
        raise ValueError("k must be between one and the candidate count")
    # Stable candidate-index tie break makes the metric exactly reproducible.
    ptop = set(np.argsort(p, kind="mergesort")[:k].tolist())
    ttop = set(np.argsort(t, kind="mergesort")[:k].tolist())
    return float(len(ptop & ttop) / int(k))


def coverage_selection_summary(proxy, true_progress, success_any, success_end,
                               *, topk_frac: float = 0.1) -> dict[str, float]:
    """Summarize opportunity coverage and proxy selection in one population.

    ``true_progress`` is the physical task distance (EE-to-goal for reach and
    object-to-goal for contact tasks).  Success arrays are exact simulator flags
    observed during and at the end of each candidate rollout.
    """
    p = np.asarray(proxy, dtype=float).reshape(-1)
    t = np.asarray(true_progress, dtype=float).reshape(-1)
    sa = np.asarray(success_any, dtype=bool).reshape(-1)
    se = np.asarray(success_end, dtype=bool).reshape(-1)
    if not (p.shape == t.shape == sa.shape == se.shape) or p.size == 0:
        raise ValueError("all candidate arrays must be aligned and non-empty")
    if not (np.isfinite(p).all() and np.isfinite(t).all()):
        raise ValueError("cost arrays must be finite")
    if not 0 < topk_frac <= 1:
        raise ValueError("topk_frac must be in (0, 1]")

    sel = int(np.argmin(p))
    true_best = float(np.min(t))
    selected_true = float(t[sel])
    k = max(1, int(np.ceil(topk_frac * p.size)))
    return {
        "n_candidates": int(p.size),
        "topk": int(k),
        "coverage_success_any": int(sa.any()),
        "coverage_success_end": int(se.any()),
        "n_success_any": int(sa.sum()),
        "n_success_end": int(se.sum()),
        "selected_success_any": int(sa[sel]),
        "selected_success_end": int(se[sel]),
        "best_true_progress": true_best,
        "selected_true_progress": selected_true,
        "selected_physical_regret": max(0.0, selected_true - true_best),
        "proxy_true_spearman": spearman_costs(p, t),
        "proxy_true_topk_overlap": topk_overlap(p, t, k),
        "selected_index": sel,
    }
