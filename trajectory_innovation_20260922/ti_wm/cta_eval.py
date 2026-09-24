"""Offline ranking metrics for CTA scorers on held-out decisions (docs/CTA_E2E_PROTOCOL.md).

Every decision is one bank of K siblings with labels cov8. Intervals cluster by root episode.
"""

import numpy as np
from scipy.stats import spearmanr

from ti_wm.contract import select_candidate
from ti_wm.gates import cluster_ratio


def _point(num, den):
    return {"ratio": float(num.sum() / den.sum()) if den.sum() > 0 else float("nan")}


def ranking_metrics(scores, labels, roots, ci=True):
    """scores, labels: (N, K); roots: (N,). Within-bank Spearman, retained gap vs the label oracle, default rate.

    Spearman is averaged over decisions with a label spread; a scorer that ties the whole bank there (e.g. a world
    model that predicts one code for every candidate) counts as 0, not as missing.
    """
    uniq = np.unique(roots)
    pos = {r: i for i, r in enumerate(uniq)}
    num, den, rho_sum, rho_n = (np.zeros(len(uniq)) for _ in range(4))
    chosen = np.array([select_candidate([float(x) for x in s]) for s in scores])
    for n in range(len(labels)):
        i = pos[roots[n]]
        num[i] += labels[n][chosen[n]] - labels[n][0]
        den[i] += labels[n].max() - labels[n][0]
        if np.ptp(labels[n]) > 0:
            rho_sum[i] += spearmanr(scores[n], labels[n]).statistic if np.ptp(scores[n]) > 0 else 0.0
            rho_n[i] += 1
    stat = lambda a, b: (cluster_ratio if ci else _point)(a, b) if len(a) and b.sum() > 0 else dict.fromkeys(("ratio", "lo", "hi"), float("nan"))
    keep = rho_n > 0
    return {"within_bank_spearman": stat(rho_sum[keep], rho_n[keep]), "retained_gap": stat(num, den),
            "chose_default": float(np.mean(chosen == 0)), "chosen": chosen}


def choice_agreement(a, b, labels):
    """Fraction of decisions with a label spread where two scorers pick the same candidate."""
    keep = np.ptp(labels, axis=1) > 0
    pick = lambda s: np.array([select_candidate([float(x) for x in row]) for row in s[keep]])
    return float(np.mean(pick(a) == pick(b))) if keep.any() else float("nan")


def perplexity(idx, v):
    """Mean over code positions of exp(entropy) of the empirical index distribution. idx: (N, M) ints."""
    out = []
    for m in range(idx.shape[1]):
        p = np.bincount(idx[:, m], minlength=v) / idx.shape[0]
        p = p[p > 0]
        out.append(float(np.exp(-(p * np.log(p)).sum())))
    return float(np.mean(out))


def bank_distinct(idx):
    """Fraction of sibling pairs in a bank whose codes differ. idx: (N, K, M) ints."""
    same = (idx[:, :, None] == idx[:, None, :]).all(-1)
    k = idx.shape[1]
    off = ~np.eye(k, dtype=bool)
    return float(1 - same[:, off].mean())
